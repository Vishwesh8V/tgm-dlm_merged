#!/usr/bin/env bash
# ==============================================================================
# Shell Script: Generation-Only Trajectory Sampling -- 50 Test Examples
#
# Scoped-down variant of run_full_trajectory_pipeline.sh:
#   - generation task ONLY (no retro, no forward)
#   - --num_samples 50 instead of 1000 (first 50 examples of the test split
#     -- text_sample.py's start_idx defaults to 0, so num_samples=50 alone
#     already gives you exactly that; sample.sh's split is already hardcoded
#     to "test")
#   - K in {10, 5, 4, 3, 2} (no K=100)
#   - both orders (DPM=2, DDIM=1)
#
# Stages 1-4 (profiling + schedule allocation) do NOT depend on --num_samples
# -- only on the checkpoint and K -- so this script points RUN_NAME at the
# SAME reduced_step_outputs/trajectory_generation/ directory
# run_full_trajectory_pipeline.sh already populated, with REUSE_PROFILE=true,
# so Stages 1-3 and every K in {10,5,4,3,2}'s Stage 4 schedule are picked up
# as already-done and skipped entirely -- only Stage 5 (fresh sampling at
# num_samples=50) actually runs.
#
# Stage 5 output goes into a brand-new directory
# (generation_outputs/trajectory_sweep_200test/generation/), never the
# original generation_outputs/trajectory_sweep/generation*.txt files from
# the 1000-sample run -- these are a different experiment and must not
# collide with or overwrite those.
#
# PLACE THIS FILE AT THE ROOT OF tgm-dlm_reduced_step, same as
# run_full_trajectory_pipeline.sh -- it resolves paths the same way.
# ==============================================================================

set -e

# ==============================================================================
# User Configuration
# ==============================================================================
STEPS_LIST=(10 5 4 3 2)
ORDERS=(2 1)   # 2 = DPM-Solver++(2M); 1 = DDIM-equivalent

NUM_SAMPLES=50
BATCH_SIZE=64
SEEDS="108"

# Points at the SAME profiling run generation already has, so Stages 1-4
# below are found already-done and skipped. Set to a new name (and
# REUSE_PROFILE=false) if you ever want fresh profiling for this experiment
# instead.
RUN_NAME="trajectory_generation"
REUSE_PROFILE="true"

# Stage 5 output root -- deliberately NOT generation_outputs/trajectory_sweep
# (that already holds the 1000-sample results for the same K/order pairs;
# reusing it would silently overwrite a different experiment's numbers).
OUTPUT_SUBDIR="trajectory_sweep_50test"

MODEL_PATH_DEFAULT="checkpoints/PLAIN_ema_0.9999_200000.pt"
ADAPTIVE_SCHEDULE_DEFAULT="checkpoints/adaptive_schedule/alpha_cumprod_step_190000.npy"
DATA_DIR_DEFAULT="datasets/SMILES"
PROFILE_SPLIT="train_val_256"
NUM_PROFILE_EXAMPLES=256
MIN_ACTIVE_FRAC=0.5
DENSITY_MODE="paper"
SEED=108

LEARNED_MEAN_EMBED=True
DENOISE=True
DENOISE_RATE=0.2
REG_RATE=0.1
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
SCRIPTS_DIR="${PROJECT_ROOT}/improved-diffusion/scripts"

MODEL_PATH="${PROJECT_ROOT}/${MODEL_PATH_DEFAULT}"
ADAPTIVE_SCHEDULE="${PROJECT_ROOT}/${ADAPTIVE_SCHEDULE_DEFAULT}"
DATA_DIR="${PROJECT_ROOT}/${DATA_DIR_DEFAULT}"
SAMPLE_SH="${PROJECT_ROOT}/sample.sh"

OUT_DIR="${PROJECT_ROOT}/reduced_step_outputs/${RUN_NAME}"
GEN_DIR="${PROJECT_ROOT}/generation_outputs/${OUTPUT_SUBDIR}/generation"

if [ ! -f "${SAMPLE_SH}" ]; then
    echo "ERROR: ${SAMPLE_SH} not found." >&2
    exit 1
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="${PROJECT_ROOT}/improved-diffusion:${PROJECT_ROOT}/transformers/src:${PYTHONPATH}"

# This is the "new folder" -- created fresh here, distinct from any prior run.
mkdir -p "${GEN_DIR}"
mkdir -p "${OUT_DIR}"

echo "======================================================================"
echo " Generation-Only Trajectory Sampling -- 50 Test Examples"
echo " K values         : ${STEPS_LIST[*]}"
echo " Orders           : ${ORDERS[*]} (2=DPM, 1=DDIM)"
echo " Model Checkpoint : ${MODEL_PATH}"
echo " Reusing profile  : reduced_step_outputs/${RUN_NAME} (REUSE_PROFILE=${REUSE_PROFILE})"
echo " New output dir   : ${GEN_DIR}"
echo "======================================================================"

SCHEDULE_ARG=""
if [ -n "${ADAPTIVE_SCHEDULE}" ] && [ -f "${ADAPTIVE_SCHEDULE}" ]; then
    SCHEDULE_ARG="--adaptive_schedule_path ${ADAPTIVE_SCHEDULE}"
fi

STAGE1="${OUT_DIR}/stage1_profile.npz"
STAGE2="${OUT_DIR}/stage2_gradient.npz"
STAGE3_DIR="${OUT_DIR}/stage3_density"
STAGE3="${STAGE3_DIR}/density_profile_${DENSITY_MODE}.npz"

pushd "${SCRIPTS_DIR}" > /dev/null

if [ "${REUSE_PROFILE}" = "true" ] && [ -f "${STAGE1}" ]; then
    echo " -> Stage 1 already exists (${STAGE1}) -- skipping."
else
    echo "" && echo "=== Stage 1: Validation-loss profiling ==="
    python reduced_step_profile.py \
      --model_path "${MODEL_PATH}" --data_dir "${DATA_DIR}" \
      --split "${PROFILE_SPLIT}" --num_examples "${NUM_PROFILE_EXAMPLES}" \
      --out_path "${STAGE1}" --seed "${SEED}" \
      --learned_mean_embed "${LEARNED_MEAN_EMBED}" --denoise "${DENOISE}" \
      --denoise_rate "${DENOISE_RATE}" --reg_rate "${REG_RATE}" ${SCHEDULE_ARG}
fi

if [ "${REUSE_PROFILE}" = "true" ] && [ -f "${STAGE2}" ]; then
    echo " -> Stage 2 already exists (${STAGE2}) -- skipping."
else
    echo "" && echo "=== Stage 2: Gradient ==="
    python trajectory_analysis.py gradient --input "${STAGE1}" --output "${STAGE2}" \
      --plot-dir "${OUT_DIR}/plots_stage2" --min-active-frac "${MIN_ACTIVE_FRAC}"
fi

if [ "${REUSE_PROFILE}" = "true" ] && [ -f "${STAGE3}" ]; then
    echo " -> Stage 3 already exists (${STAGE3}) -- skipping."
else
    echo "" && echo "=== Stage 3: Density ==="
    python trajectory_analysis.py monitor --input "${STAGE2}" --output-dir "${STAGE3_DIR}" \
      --mode "${DENSITY_MODE}"
fi

popd > /dev/null

GENERATED_FILES=()

for K in "${STEPS_LIST[@]}"; do
    SCHED_DIR="${OUT_DIR}/stage4_schedule_K${K}"
    STEP_MATRIX="${SCHED_DIR}/schedule_all_positions.csv"

    if [ -f "${STEP_MATRIX}" ]; then
        echo ""
        echo " -> K=${K}: Stage 4 schedule already exists -- skipping."
    else
        echo "" && echo "=== K=${K}: Stage 4 -- allocate ==="
        pushd "${SCRIPTS_DIR}" > /dev/null
        python trajectory_analysis.py allocate \
          --input "${STAGE3}" --trajectory "${STAGE1}" \
          --output "${SCHED_DIR}/schedule.npz" --K "${K}" \
          --plot-dir "${OUT_DIR}/plots_stage4_K${K}"
        popd > /dev/null
    fi

    for ORDER in "${ORDERS[@]}"; do
        OUT_FILE="${GEN_DIR}/K${K}_order${ORDER}.txt"
        OUT_FILE_SEEDED="${OUT_FILE%.txt}_seed${SEEDS}.txt"

        if [ -f "${OUT_FILE_SEEDED}" ] || [ -f "${OUT_FILE}" ]; then
            echo " -> K=${K} order=${ORDER}: output already exists -- skipping."
            continue
        fi

        echo "" && echo "=== K=${K} order=${ORDER}: Stage 5 -- sampling (50 test examples) ==="

        MODEL_PATH="${MODEL_PATH}" \
        ADAPTIVE_SCHEDULE="${ADAPTIVE_SCHEDULE}" \
        DATASETS_DIR="${DATA_DIR}" \
        OUTPUT_FILE="${OUT_FILE}" \
        NUM_SAMPLES="${NUM_SAMPLES}" \
        BATCH_SIZE="${BATCH_SIZE}" \
        SEEDS="${SEEDS}" \
        USE_DPM_SOLVER="false" \
        DPM_SOLVER_ORDER="${ORDER}" \
        STEP_MATRIX_PATH="${STEP_MATRIX}" \
        TOKEN_ADAPTIVE_STEPS="${K}" \
        LEARNED_MEAN_EMBED="${LEARNED_MEAN_EMBED}" \
        DENOISE="${DENOISE}" \
        DENOISE_RATE="${DENOISE_RATE}" \
        REG_RATE="${REG_RATE}" \
        bash "${SAMPLE_SH}"

        GENERATED_FILES+=("${OUT_FILE_SEEDED}")
    done
done

echo ""
echo "======================================================================"
echo " Done. Outputs under: ${GEN_DIR}"
for f in "${GENERATED_FILES[@]}"; do
    echo "   -> $f"
done
echo "======================================================================"