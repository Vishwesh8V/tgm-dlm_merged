#!/usr/bin/env bash
# ==============================================================================
# Shell Script: Run Inference / Sampling for TGM-DLM
# ==============================================================================

set -e

# --- Environment Setup ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMPI_MCA_btl="^openib"
#!/usr/bin/env bash
# ==============================================================================
# Shell Script: Run Inference / Sampling for TGM-DLM
# ==============================================================================


# ==============================================================================
# User Configuration (Edit parameters here)
# ==============================================================================
CUDA_DEVICE="0"

MODEL_PATH="/home/ee/phd/eez248435/tgm-dlm_merged/checkpoints/PLAIN_ema_0.9999_200000.pt"

# Set path to .npy schedule file, or set to "none" for standard uniform schedule
ADAPTIVE_SCHEDULE="/home/ee/phd/eez248435/tgm-dlm_merged/checkpoints/adaptive_schedule/alpha_cumprod_step_190000.npy"

OUTPUT_FILE="../../generation_outputs/sampled_smiles_200k_3108.txt"

NUM_SAMPLES=3300
BATCH_SIZE=64
TIMESTEP_RESPACING=1

# Random Seeds: single seed "121" or multiple seeds "101,102,103"
SEEDS="108,112,126,135,201"

# --- DPM-Solver++ (fast ODE sampler) ---
# When enabled, overrides the p_sample_loop/ddim sampler above; TIMESTEP_RESPACING
# is not used for this path (it works on the full 2000-step noise schedule and
# picks its own steps). Typically 10-20 steps is enough.
USE_DPM_SOLVER=False
DPM_SOLVER_STEPS=2000
DPM_SOLVER_ORDER=2
DPM_SOLVER_METHOD="multistep"

# --- Mixed-Space diffusion ---
LEARNED_MEAN_EMBED=True
DENOISE=True
DENOISE_RATE=0.2
REG_RATE=0.1
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
DATASETS_DIR="${PROJECT_ROOT}/datasets/SMILES"
OUT_DIR="${PROJECT_ROOT}/generation_outputs"

# --- Seed Configuration ---
CLEAN_SEEDS=$(echo "${SEEDS}" | tr ',' ' ')
read -r -a SEED_ARRAY <<< "${CLEAN_SEEDS}"

# Determine base output filename pattern
OUT_DIRNAME="$(dirname "${OUTPUT_FILE}")"
OUT_BASENAME="$(basename "${OUTPUT_FILE}")"
OUT_STEM=$(echo "${OUT_BASENAME}" | sed -E 's/_seed[0-9]+(\.txt)?$//' | sed 's/\.txt$//')

# Set PYTHONPATH to include improved-diffusion and transformers/src
export PYTHONPATH="${PROJECT_ROOT}/improved-diffusion:${PROJECT_ROOT}/transformers/src:${PYTHONPATH}"

# Ensure output directory exists
mkdir -p "${OUT_DIR}"

# Navigate to scripts directory containing text_sample.py
cd "${SCRIPTS_DIR}"

echo "======================================================================"
echo " Starting TGM-DLM Inference / Molecular Sampling"
echo "======================================================================"
echo " Project Root      : ${PROJECT_ROOT}"
echo " Scripts Dir       : ${SCRIPTS_DIR}"
echo " Model Checkpoint  : ${MODEL_PATH}"
echo " Adaptive Schedule : ${ADAPTIVE_SCHEDULE:-"(None - Uniform Baseline)"}"
echo " Data Directory    : ${DATASETS_DIR}"
echo " Timestep Respacing: ${TIMESTEP_RESPACING}"
echo " Batch Size        : ${BATCH_SIZE}"
echo " NUM_SAMPLES       : ${NUM_SAMPLES}"
echo " Target Base Name  : ${OUT_DIRNAME}/${OUT_STEM}.txt"
echo " Seeds to Run      : [${SEED_ARRAY[*]}] (Total: ${#SEED_ARRAY[@]} seeds)"
echo " CUDA Visible Devs : ${CUDA_VISIBLE_DEVICES}"
echo "======================================================================"

# Determine if adaptive schedule file exists
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

# Run evaluation with ev.py (auto-aggregates all matching seeds)
if [ -f "${PROJECT_ROOT}/ev.py" ] && [ ${#GENERATED_FILES[@]} -gt 0 ]; then
    echo ""
    echo "======================================================================"
    echo " Running Evaluation via ev.py..."
    echo "======================================================================"
    python "${PROJECT_ROOT}/ev.py" "${GENERATED_FILES[0]}"
fi