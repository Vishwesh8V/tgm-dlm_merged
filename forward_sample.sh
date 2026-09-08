#!/usr/bin/env bash
# ==============================================================================
# Shell Script: Run Inference / Sampling for TGM-DLM -- Forward Reaction Prediction
#
# Identical to sample.sh (same text_sample.py, same ev.py evaluation at the
# end) -- only DATASETS_DIR/MODEL_PATH/ADAPTIVE_SCHEDULE point at the
# forward-reaction dataset/checkpoints instead of ChEBI's datasets/SMILES.
#
# Recall the forward role mapping: smiles (diffusion target) = product,
# desc (condition) = reactants. So this samples a product given reactants.
# ==============================================================================

set -e

# --- Environment Setup ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMPI_MCA_btl="^openib"

# ==============================================================================
# User Configuration (Edit parameters here)
# ==============================================================================
CUDA_DEVICE="0"

# Point this at whichever checkpoint you want to evaluate, e.g. the latest
# PLAIN_ema_0.9999_<step>.pt written by forward_train.sh into checkpoints_forward/
MODEL_PATH="/home/ee/phd/eez248435/tgm-dlm_merged/checkpoints_forward/PLAIN_ema_0.9999_140000.pt"

# Set path to .npy schedule file, or set to "none" for standard uniform schedule
ADAPTIVE_SCHEDULE="/home/ee/phd/eez248435/tgm-dlm_merged/checkpoints_forward/adaptive_schedule/alpha_cumprod_step_130000.npy"

OUTPUT_FILE="../../generation_outputs/fp_full_140k.txt"

NUM_SAMPLES=1000
BATCH_SIZE=64
TIMESTEP_RESPACING=1

# Random Seeds: single seed "121" or multiple seeds "101,102,103"
SEEDS="108"

# --- DPM-Solver++ (fast ODE sampler) ---
USE_DPM_SOLVER=False
DPM_SOLVER_STEPS=2000
DPM_SOLVER_ORDER=2
DPM_SOLVER_METHOD="multistep"

# --- Mixed-Space diffusion (must match what the checkpoint was trained with) ---
LEARNED_MEAN_EMBED=True
DENOISE=True
DENOISE_RATE=0.2
REG_RATE=0.1

# --- Vocab (must match what the checkpoint was trained with) ---
# The forward checkpoint likely uses its own vocab, not the default ChEBI/
# text-caption one text_sample.py falls back to -- without this, loading the
# checkpoint either crashes with a word_embedding/lm_head size mismatch, or
# (if the sizes happen to coincidentally match) silently decodes garbage.
# Verify with: wc -l "${DATASETS_DIR}/generate_vocab.txt" against the
# checkpoint's word_embedding.weight size reported in any past error, minus 3
# for [PAD]/[SOS]/[EOS].
VOCAB_PATH="/home/ee/phd/eez248435/tgm-dlm_merged/datasets/FORWARD/generate_vocab.txt"
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
DATASETS_DIR="${PROJECT_ROOT}/datasets/FORWARD"
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
echo " Starting TGM-DLM Inference / Molecular Sampling -- FORWARD REACTION PREDICTION"
echo "======================================================================"
echo " Project Root      : ${PROJECT_ROOT}"
echo " Task               : Reactants -> Product"
echo " Model Checkpoint  : ${MODEL_PATH}"
echo " Adaptive Schedule : ${ADAPTIVE_SCHEDULE:-"(None - Uniform Baseline)"}"
echo " Data Directory    : ${DATASETS_DIR}"
echo " Vocab Path        : ${VOCAB_PATH}"
echo " Timestep Respacing: ${TIMESTEP_RESPACING}"
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