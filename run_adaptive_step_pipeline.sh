#!/usr/bin/env bash
# ==============================================================================
# Shell Script: Adaptive Step Inference Pipeline (Token-Wise Reduced-Step
# Sampling) for TGM-DLM Phase 1
#
# Builds a per-token-position reduced-step schedule from a trained Phase-1
# checkpoint's validation behavior, then samples with it. Four stages:
#
#   1. reduced_step_profile.py  -- full T-step validation-loss profiling
#   2. trajectory_analysis.py gradient  -- smooth loss(logSNR), take dE/dlambda
#   3. trajectory_analysis.py monitor   -- turn gradients into a step density
#   4. trajectory_analysis.py allocate  -- discretize density into J[K, L]
#   5. text_sample.py --step_matrix_path ...  -- sample with J
#
# Stage 1 is the expensive one (T model calls per validation example) but
# only needs to run once per checkpoint. Stages 2-4 are pure numpy/scipy and
# run in seconds. See ADAPTIVE_STEP_INFERENCE.md for the full explanation.
# ==============================================================================

set -e

# ==============================================================================
# User Configuration (Edit parameters here)
# ==============================================================================
CUDA_DEVICE="0"

MODEL_PATH="/home/ee/phd/eez248435/tgm-dlm_merged/checkpoints/PLAIN_ema_0.9999_170000.pt"

# If this checkpoint was trained with Adaptive Noising (see ADAPTIVE_NOISING.md),
# point this at the matching alpha_cumprod_step_N.npy so Stage 1 profiles the
# model against the noise schedule it was actually trained on, and Stage 5
# samples with that same schedule. Set to "" or "none" to skip (plain 1-D
# schedule, e.g. a checkpoint trained without adaptive noising).
ADAPTIVE_SCHEDULE="/home/ee/phd/eez248435/tgm-dlm_merged/checkpoints/adaptive_schedule/alpha_cumprod_step_160000.npy"

RUN_NAME="adaptive_testrun1"
K=10                       # number of reduced decoding calls (must match --token_adaptive_steps below)
PROFILE_SPLIT="train_val_256"
NUM_PROFILE_EXAMPLES=2000   # validation examples used to build the schedule
MIN_ACTIVE_FRAC=0.5        # position must have non-pad coverage this often to get an adaptive schedule
DENSITY_MODE="paper"       # "paper" or "magnitude", see trajectory_analysis.py

SAMPLE_SPLIT="test"
NUM_SAMPLES=1000
BATCH_SIZE=64
SEED=108

# --- Mixed-Space Diffusion ---
# Must match how MODEL_PATH was trained (see UNIFIED_FEATURES.md), or
# reduced_step_profile.py's `model.load_state_dict()` will fail on `mean_embed`
# (Stage 1) and Stage 5 sampling will silently diverge from training-time
# behavior even if it happens to load without error.
LEARNED_MEAN_EMBED=True
DENOISE=True
DENOISE_RATE=0.2
REG_RATE=0.1

# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
SCRIPTS_DIR="${PROJECT_ROOT}/improved-diffusion/scripts"
DATASETS_DIR="${PROJECT_ROOT}/datasets/SMILES"
OUT_DIR="${PROJECT_ROOT}/reduced_step_outputs/${RUN_NAME}"
GEN_DIR="${PROJECT_ROOT}/generation_outputs"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export PYTHONPATH="${PROJECT_ROOT}/improved-diffusion:${PROJECT_ROOT}/transformers/src:${PYTHONPATH}"

mkdir -p "${OUT_DIR}" "${GEN_DIR}"
cd "${SCRIPTS_DIR}"

# Determine whether an adaptive-noising schedule was supplied.
SCHEDULE_ARG=""
if [ -n "${ADAPTIVE_SCHEDULE}" ] && [ "${ADAPTIVE_SCHEDULE}" != "none" ] && [ -f "${ADAPTIVE_SCHEDULE}" ]; then
    SCHEDULE_ARG="--adaptive_schedule_path ${ADAPTIVE_SCHEDULE}"
    echo " -> Found adaptive noising schedule, will use it for profiling and sampling: ${ADAPTIVE_SCHEDULE}"
else
    echo " -> No adaptive noising schedule supplied, using the plain 1-D noise schedule."
fi

echo "======================================================================"
echo " Stage 1: Validation-loss profiling (full T-step trajectory)"
echo "======================================================================"
python reduced_step_profile.py \
  --model_path "${MODEL_PATH}" \
  --data_dir "${DATASETS_DIR}" \
  --split "${PROFILE_SPLIT}" \
  --num_examples "${NUM_PROFILE_EXAMPLES}" \
  --out_path "${OUT_DIR}/stage1_profile.npz" \
  --seed "${SEED}" \
  --learned_mean_embed "${LEARNED_MEAN_EMBED}" \
  --denoise "${DENOISE}" \
  --denoise_rate "${DENOISE_RATE}" \
  --reg_rate "${REG_RATE}" \
  ${SCHEDULE_ARG}

echo "======================================================================"
echo " Stage 2: Smooth loss(logSNR) -> dE/dlambda"
echo "======================================================================"
python trajectory_analysis.py gradient \
  --input "${OUT_DIR}/stage1_profile.npz" \
  --output "${OUT_DIR}/stage2_gradient.npz" \
  --plot-dir "${OUT_DIR}/plots_stage2" \
  --min-active-frac "${MIN_ACTIVE_FRAC}"

echo "======================================================================"
echo " Stage 3: Gradients -> reduced-step density"
echo "======================================================================"
python trajectory_analysis.py monitor \
  --input "${OUT_DIR}/stage2_gradient.npz" \
  --output-dir "${OUT_DIR}/stage3_density" \
  --mode "${DENSITY_MODE}"

echo "======================================================================"
echo " Stage 4: Density -> discrete step matrix J[K, L]"
echo "======================================================================"
python trajectory_analysis.py allocate \
  --input "${OUT_DIR}/stage3_density/density_profile_${DENSITY_MODE}.npz" \
  --trajectory "${OUT_DIR}/stage1_profile.npz" \
  --output "${OUT_DIR}/stage4_schedule/schedule.npz" \
  --K "${K}" \
  --plot-dir "${OUT_DIR}/plots_stage4"

STEP_MATRIX="${OUT_DIR}/stage4_schedule/schedule_all_positions.csv"

echo "======================================================================"
echo " Stage 5: Sampling with the adaptive step schedule"
echo "======================================================================"
python text_sample.py \
  --model_path "${MODEL_PATH}" \
  --outputdir "${GEN_DIR}/${RUN_NAME}_adaptive_K${K}_seed${SEED}.txt" \
  --data_dir "${DATASETS_DIR}" \
  --seed "${SEED}" \
  --split "${SAMPLE_SPLIT}" \
  --num_samples "${NUM_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --clip_denoised false \
  --step_matrix_path "${STEP_MATRIX}" \
  --token_adaptive_steps "${K}" \
  --learned_mean_embed "${LEARNED_MEAN_EMBED}" \
  --denoise "${DENOISE}" \
  --denoise_rate "${DENOISE_RATE}" \
  --reg_rate "${REG_RATE}" \
  ${SCHEDULE_ARG}

echo "======================================================================"
echo " Done. Adaptive-step samples written to:"
echo "   ${GEN_DIR}/${RUN_NAME}_adaptive_K${K}_seed${SEED}.txt"
echo ""
echo " For comparison, the uniform-step DPM-Solver baseline at the same K:"
echo "   python text_sample.py --model_path ${MODEL_PATH} \\"
echo "     --outputdir ${GEN_DIR}/${RUN_NAME}_uniform_K${K}_seed${SEED}.txt \\"
echo "     --data_dir ${DATASETS_DIR} --seed ${SEED} --split ${SAMPLE_SPLIT} \\"
echo "     --num_samples ${NUM_SAMPLES} --batch_size ${BATCH_SIZE} --clip_denoised false \\"
echo "     --learned_mean_embed ${LEARNED_MEAN_EMBED} --denoise ${DENOISE} --denoise_rate ${DENOISE_RATE} --reg_rate ${REG_RATE} \\"
echo "     --use_dpm_solver true --dpm_solver_steps ${K} --dpm_solver_order 2 ${SCHEDULE_ARG}"
echo "======================================================================"