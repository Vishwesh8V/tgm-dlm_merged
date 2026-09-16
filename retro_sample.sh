#!/usr/bin/env bash
# ==============================================================================
# Shell Script: Run Inference / Sampling for TGM-DLM -- Retrosynthesis
#
# Identical to sample.sh (same text_sample.py, same ev.py evaluation at the
# end) -- only DATASETS_DIR/MODEL_PATH/ADAPTIVE_SCHEDULE point at the
# retrosynthesis dataset/checkpoints instead of ChEBI's datasets/SMILES.
#
# Recall the retro role mapping: smiles (diffusion target) = reactants,
# desc (condition) = product. So this samples reactants given a product.
# ==============================================================================

set -e

# --- Environment Setup ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMPI_MCA_btl="^openib"

# ==============================================================================
# User Configuration (Edit parameters here)
#
# Every value below can also be overridden by exporting the same-named
# environment variable before calling this script (same pattern already
# used for CUDA_VISIBLE_DEVICES/WANDB_MODE above) -- lets sweep scripts
# drive this file directly instead of duplicating its argument-building
# logic. Nothing changes if no env vars are set.
# ==============================================================================
CUDA_DEVICE="${CUDA_DEVICE:-0}"

# Point this at whichever checkpoint you want to evaluate, e.g. the latest
# PLAIN_ema_0.9999_<step>.pt written by retro_train.sh into checkpoints_retro/
MODEL_PATH="${MODEL_PATH:-/home/ee/phd/eez248435/tgm-dlm_merged/checkpoints_retro/PLAIN_ema_0.9999_170000.pt}"

# Set path to .npy schedule file, or set to "none" for standard uniform schedule
ADAPTIVE_SCHEDULE="${ADAPTIVE_SCHEDULE:-/home/ee/phd/eez248435/tgm-dlm_merged/checkpoints_retro/adaptive_schedule/alpha_cumprod_step_160000.npy}"

OUTPUT_FILE="${OUTPUT_FILE:-../../generation_outputs/retro_sampled_dpm10_170k_test.txt}"

NUM_SAMPLES="${NUM_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TIMESTEP_RESPACING="${TIMESTEP_RESPACING:-1}"

# Random Seeds: single seed "121" or multiple seeds "101,102,103"
SEEDS="${SEEDS:-108}"

# --- DPM-Solver++ (fast ODE sampler) ---
USE_DPM_SOLVER="${USE_DPM_SOLVER:-True}"
DPM_SOLVER_STEPS="${DPM_SOLVER_STEPS:-10}"
DPM_SOLVER_ORDER="${DPM_SOLVER_ORDER:-2}"
DPM_SOLVER_METHOD="${DPM_SOLVER_METHOD:-multistep}"

# --- Token-adaptive step schedule (per-position reduced-step inference) ---
# Alternative to DPM-Solver++ above -- mutually exclusive with it (text_sample.py
# will error out if both are set). Produced offline by:
#   1. reduced_step_profile.py   (Stage 1: profile the checkpoint)
#   2. trajectory_analysis.py    gradient -> monitor -> allocate  (Stages 2-4)
# Stage 4's "allocate" step writes a `<name>_J.npy` file -- point STEP_MATRIX_PATH
# at that. Set to "" or "none" to disable and use one of the samplers above instead.
STEP_MATRIX_PATH="${STEP_MATRIX_PATH:-}"
TOKEN_ADAPTIVE_STEPS="${TOKEN_ADAPTIVE_STEPS:-10}"

# --- Mixed-Space diffusion (must match what the checkpoint was trained with) ---
LEARNED_MEAN_EMBED="${LEARNED_MEAN_EMBED:-True}"
DENOISE="${DENOISE:-True}"
DENOISE_RATE="${DENOISE_RATE:-0.2}"
REG_RATE="${REG_RATE:-0.1}"

# --- Vocab (must match what the checkpoint was trained with) ---
# The retro checkpoint uses its own smaller vocab, not the default ChEBI/
# forward one text_sample.py falls back to -- without this, loading the
# checkpoint fails with a word_embedding/lm_head size mismatch.
VOCAB_PATH="${VOCAB_PATH:-/home/ee/phd/eez248435/tgm-dlm_merged/datasets/RETRO/generate_vocab.txt}"
# ==============================================================================

# --- Determine Directory Structure ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/improved-diffusion/scripts/text_sample.py" ]; then
    PROJECT_ROOT="${SCRIPT_DIR}"
elif [ -f "${SCRIPT_DIR}/scripts/text_sample.py" ]; then
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    PROJECT_ROOT="${SCRIPT_DIR}"
fi

SCRIPTS_DIR="${PROJECT_ROOT}/improved-diffusion/scripts"
DATASETS_DIR="${DATASETS_DIR:-${PROJECT_ROOT}/datasets/RETRO}"
OUT_DIR="${PROJECT_ROOT}/generation_outputs"

# --- Seed Configuration ---
CLEAN_SEEDS=$(echo "${SEEDS}" | tr ',' ' ')
read -r -a SEED_ARRAY <<< "${CLEAN_SEEDS}"

OUT_DIRNAME="$(dirname "${OUTPUT_FILE}")"
OUT_BASENAME="$(basename "${OUTPUT_FILE}")"
OUT_STEM=$(echo "${OUT_BASENAME}" | sed -E 's/_seed[0-9]+(\.txt)?$//' | sed 's/\.txt$//')

export PYTHONPATH="${PROJECT_ROOT}/improved-diffusion:${PROJECT_ROOT}/transformers/src:${PYTHONPATH}"

mkdir -p "${OUT_DIR}"
cd "${SCRIPTS_DIR}"

echo "======================================================================"
echo " Starting TGM-DLM Inference / Molecular Sampling -- RETROSYNTHESIS"
echo "======================================================================"
echo " Project Root      : ${PROJECT_ROOT}"
echo " Task               : Product -> Reactants"
echo " Model Checkpoint  : ${MODEL_PATH}"
echo " Adaptive Schedule : ${ADAPTIVE_SCHEDULE:-"(None - Uniform Baseline)"}"
echo " Data Directory    : ${DATASETS_DIR}"
echo " Vocab Path        : ${VOCAB_PATH}"
echo " Timestep Respacing: ${TIMESTEP_RESPACING}"
echo " DPM-Solver++      : ${USE_DPM_SOLVER} (steps=${DPM_SOLVER_STEPS}, order=${DPM_SOLVER_ORDER}, method=${DPM_SOLVER_METHOD})"
echo " Token-Adaptive     : ${STEP_MATRIX_PATH:-"(None - not using token-adaptive step schedule)"}"
echo " Batch Size        : ${BATCH_SIZE}"
echo " NUM_SAMPLES       : ${NUM_SAMPLES}"
echo " Target Base Name  : ${OUT_DIRNAME}/${OUT_STEM}.txt"
echo " Seeds to Run      : [${SEED_ARRAY[*]}] (Total: ${#SEED_ARRAY[@]} seeds)"
echo " CUDA Visible Devs : ${CUDA_VISIBLE_DEVICES}"
echo "======================================================================"

ADAPTIVE_FLAG="false"
SCHEDULE_ARG=""
if [ -n "${ADAPTIVE_SCHEDULE}" ] && [ "${ADAPTIVE_SCHEDULE}" != "none" ] && [ "${ADAPTIVE_SCHEDULE}" != "false" ] && [ -f "${ADAPTIVE_SCHEDULE}" ]; then
    ADAPTIVE_FLAG="true"
    SCHEDULE_ARG="--adaptive_schedule_path ${ADAPTIVE_SCHEDULE}"
    echo " -> Found adaptive schedule file! Enabling adaptive noise sampling."
else
    echo " -> Running with UNIFORM baseline schedule (Adaptive Noising: OFF)."
fi

# Determine if a token-adaptive step-matrix file is set; mutually exclusive
# with DPM-Solver++ (text_sample.py enforces this too, but fail fast here
# with a clearer message before spending time loading the model).
STEP_MATRIX_ARG=""
if [ -n "${STEP_MATRIX_PATH}" ] && [ "${STEP_MATRIX_PATH}" != "none" ] && [ "${STEP_MATRIX_PATH}" != "false" ]; then
    if [ ! -f "${STEP_MATRIX_PATH}" ]; then
        echo "ERROR: STEP_MATRIX_PATH is set to '${STEP_MATRIX_PATH}' but that file does not exist." >&2
        exit 1
    fi
    if [ "${USE_DPM_SOLVER}" = "True" ] || [ "${USE_DPM_SOLVER}" = "true" ]; then
        echo "ERROR: STEP_MATRIX_PATH and USE_DPM_SOLVER are mutually exclusive samplers -- pick one (this script defaults USE_DPM_SOLVER=True, set it to False to use the step matrix)." >&2
        exit 1
    fi
    STEP_MATRIX_ARG="--step_matrix_path ${STEP_MATRIX_PATH} --token_adaptive_steps ${TOKEN_ADAPTIVE_STEPS}"
    echo " -> Found step-matrix file! Using token-adaptive per-position step allocation (K=${TOKEN_ADAPTIVE_STEPS})."
else
    echo " -> Not using token-adaptive step schedule."
fi

GENERATED_FILES=()

for s in "${SEED_ARRAY[@]}"; do
    if [ ${#SEED_ARRAY[@]} -eq 1 ] && [[ "${OUT_BASENAME}" == *"_seed"* ]]; then
        CURRENT_OUT="${OUTPUT_FILE}"
    else
        CURRENT_OUT="${OUT_DIRNAME}/${OUT_STEM}_seed${s}.txt"
    fi
    GENERATED_FILES+=("${CURRENT_OUT}")

    echo ""
    echo "======================================================================"
    echo " [Seed ${s}] Running Sampling -> ${CURRENT_OUT}"
    echo "======================================================================"

    python text_sample.py \
      --model_path "${MODEL_PATH}" \
      --outputdir "${CURRENT_OUT}" \
      --data_dir "${DATASETS_DIR}" \
      --vocab_path "${VOCAB_PATH}" \
      --learned_mean_embed "${LEARNED_MEAN_EMBED}" \
      --denoise "${DENOISE}" \
      --denoise_rate "${DENOISE_RATE}" \
      --reg_rate "${REG_RATE}" \
      --adaptive_noising "${ADAPTIVE_FLAG}" \
      ${SCHEDULE_ARG} \
      --seed "${s}" \
      --split "test" \
      --num_samples "${NUM_SAMPLES}" \
      --batch_size "${BATCH_SIZE}" \
      --timestep_respacing "${TIMESTEP_RESPACING}" \
      --use_ddim false \
      --clip_denoised false \
      --use_dpm_solver "${USE_DPM_SOLVER}" \
      --dpm_solver_steps "${DPM_SOLVER_STEPS}" \
      --dpm_solver_order "${DPM_SOLVER_ORDER}" \
      --dpm_solver_method "${DPM_SOLVER_METHOD}" \
      ${STEP_MATRIX_ARG} \
      "$@"
done

echo ""
echo "======================================================================"
echo " All ${#SEED_ARRAY[@]} Seed Runs Finished!"
echo " Generated Files:"
for f in "${GENERATED_FILES[@]}"; do
    echo "   -> $f"
done
echo "======================================================================"

if [ -f "${PROJECT_ROOT}/ev.py" ] && [ ${#GENERATED_FILES[@]} -gt 0 ]; then
    echo ""
    echo "======================================================================"
    echo " Running Evaluation via ev.py..."
    echo "======================================================================"
    python "${PROJECT_ROOT}/ev.py" "${GENERATED_FILES[0]}"
fi