#!/usr/bin/env bash
# ==============================================================================
# Shell Script: Full 5-Stage Trajectory-Inference Pipeline -- All 3 Tasks
#
# Runs, in succession, for each of {generation, retro, forward}:
#   Stage 1  reduced_step_profile.py      -- profile the checkpoint (once/task)
#   Stage 2  trajectory_analysis.py gradient  (once/task)
#   Stage 3  trajectory_analysis.py monitor   (once/task)
#   Stage 4  trajectory_analysis.py allocate  -- once per (task, K); K-only,
#            does NOT depend on --dpm_solver_order (that only changes how the
#            solver *uses* a fixed schedule, not the schedule itself), so
#            each K's schedule is built once and reused by both orders below.
#   Stage 5  <task>_sample.sh, driven via env-var overrides -- once per
#            (task, K, order)
#
# PLACE THIS FILE AT THE ROOT OF tgm-dlm_reduced_step, alongside
# forward_sample.sh / retro_sample.sh / sample.sh -- it resolves paths
# relative to its own location the same way those scripts do.
#
# Total sampling runs: 3 tasks x 6 steps x 2 orders = 36.
# Total schedule builds (Stage 4): 3 tasks x 6 steps = 18.
# Total profiling runs (Stages 1-3): 3 (one per task).
# ==============================================================================

set -e

# ==============================================================================
# User Configuration
# ==============================================================================
STEPS_LIST=(100 10 5 4 3 2)
ORDERS=(2 1)   # 2 = DPM-Solver++(2M) default; 1 = DDIM-equivalent

NUM_SAMPLES=1000
BATCH_SIZE=64
SEEDS="108"

NUM_PROFILE_EXAMPLES=256   # matches SMILES's train_val_256 size, reused as the
                           # profiling-set size for retro/forward too (see below)
MIN_ACTIVE_FRAC=0.5
DENSITY_MODE="paper"
SEED=108

# Set to "true" to reuse existing Stage-1/2/3 outputs for a task (same
# RUN_NAME) instead of recomputing them.
REUSE_PROFILE="false"

# --- Mixed-Space Diffusion, same for all three checkpoints ---
LEARNED_MEAN_EMBED=True
DENOISE=True
DENOISE_RATE=0.2
REG_RATE=0.1
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
SCRIPTS_DIR="${PROJECT_ROOT}/improved-diffusion/scripts"
GEN_DIR="${PROJECT_ROOT}/generation_outputs/trajectory_sweep"

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="${PROJECT_ROOT}/improved-diffusion:${PROJECT_ROOT}/transformers/src:${PYTHONPATH}"

mkdir -p "${GEN_DIR}"

# ==============================================================================
# run_task <task_name> <model_path> <adaptive_schedule> <data_dir> <vocab_path> \
#          <sample_sh> <profile_split> <profile_num_examples>
#
# <vocab_path> may be "" (generation's sample.sh doesn't take --vocab_path at
# all; passed to Stage 1 profiling regardless, where it's always accepted --
# empty falls back to reduced_step_profile.py's own SMILES default, which is
# correct for the generation task anyway).
# ==============================================================================
run_task() {
    local TASK="$1" MODEL_PATH="$2" ADAPTIVE_SCHEDULE="$3" DATA_DIR="$4" \
          VOCAB_PATH="$5" SAMPLE_SH="$6" PROFILE_SPLIT="$7" PROFILE_NUM="$8"

    local RUN_NAME="trajectory_${TASK}"
    local OUT_DIR="${PROJECT_ROOT}/reduced_step_outputs/${RUN_NAME}"
    mkdir -p "${OUT_DIR}"

    if [ ! -f "${SAMPLE_SH}" ]; then
        echo "ERROR: ${SAMPLE_SH} not found for task '${TASK}'." >&2
        exit 1
    fi

    local SCHEDULE_ARG=""
    if [ -n "${ADAPTIVE_SCHEDULE}" ] && [ "${ADAPTIVE_SCHEDULE}" != "none" ] && [ -f "${ADAPTIVE_SCHEDULE}" ]; then
        SCHEDULE_ARG="--adaptive_schedule_path ${ADAPTIVE_SCHEDULE}"
    fi
    local VOCAB_ARG=""
    if [ -n "${VOCAB_PATH}" ]; then
        VOCAB_ARG="--vocab_path ${VOCAB_PATH}"
    fi

    echo ""
    echo "######################################################################"
    echo "# TASK: ${TASK}"
    echo "#   Model     : ${MODEL_PATH}"
    echo "#   Schedule  : ${ADAPTIVE_SCHEDULE}"
    echo "#   Data dir  : ${DATA_DIR}"
    echo "#   Vocab     : ${VOCAB_PATH:-"(none -- Stage 1 falls back to its own default)"}"
    echo "#   Sample.sh : ${SAMPLE_SH}"
    echo "######################################################################"

    local STAGE1="${OUT_DIR}/stage1_profile.npz"
    local STAGE2="${OUT_DIR}/stage2_gradient.npz"
    local STAGE3_DIR="${OUT_DIR}/stage3_density"
    local STAGE3="${STAGE3_DIR}/density_profile_${DENSITY_MODE}.npz"

    pushd "${SCRIPTS_DIR}" > /dev/null

    if [ "${REUSE_PROFILE}" = "true" ] && [ -f "${STAGE1}" ]; then
        echo " -> [${TASK}] REUSE_PROFILE=true and Stage 1 output exists -- skipping."
    else
        echo ""
        echo "=== [${TASK}] Stage 1: Validation-loss profiling (runs ONCE) ==="
        python reduced_step_profile.py \
          --model_path "${MODEL_PATH}" \
          --data_dir "${DATA_DIR}" \
          ${VOCAB_ARG} \
          --split "${PROFILE_SPLIT}" \
          --num_examples "${PROFILE_NUM}" \
          --out_path "${STAGE1}" \
          --seed "${SEED}" \
          --learned_mean_embed "${LEARNED_MEAN_EMBED}" \
          --denoise "${DENOISE}" \
          --denoise_rate "${DENOISE_RATE}" \
          --reg_rate "${REG_RATE}" \
          ${SCHEDULE_ARG}
    fi

    if [ "${REUSE_PROFILE}" = "true" ] && [ -f "${STAGE2}" ]; then
        echo " -> [${TASK}] REUSE_PROFILE=true and Stage 2 output exists -- skipping."
    else
        echo ""
        echo "=== [${TASK}] Stage 2: Smooth loss(logSNR) -> dE/dlambda (runs ONCE) ==="
        python trajectory_analysis.py gradient \
          --input "${STAGE1}" \
          --output "${STAGE2}" \
          --plot-dir "${OUT_DIR}/plots_stage2" \
          --min-active-frac "${MIN_ACTIVE_FRAC}"
    fi

    if [ "${REUSE_PROFILE}" = "true" ] && [ -f "${STAGE3}" ]; then
        echo " -> [${TASK}] REUSE_PROFILE=true and Stage 3 output exists -- skipping."
    else
        echo ""
        echo "=== [${TASK}] Stage 3: Gradients -> reduced-step density (runs ONCE) ==="
        python trajectory_analysis.py monitor \
          --input "${STAGE2}" \
          --output-dir "${STAGE3_DIR}" \
          --mode "${DENSITY_MODE}"
    fi

    popd > /dev/null

    for K in "${STEPS_LIST[@]}"; do
        local SCHED_DIR="${OUT_DIR}/stage4_schedule_K${K}"
        local STEP_MATRIX="${SCHED_DIR}/schedule_all_positions.csv"

        if [ -f "${STEP_MATRIX}" ]; then
            echo ""
            echo " -> [${TASK}] K=${K}: Stage 4 schedule already exists -- skipping."
        else
            echo ""
            echo "=== [${TASK}] K=${K}: Stage 4 -- Density -> discrete step matrix J[K, L] ==="
            pushd "${SCRIPTS_DIR}" > /dev/null
            python trajectory_analysis.py allocate \
              --input "${STAGE3}" \
              --trajectory "${STAGE1}" \
              --output "${SCHED_DIR}/schedule.npz" \
              --K "${K}" \
              --plot-dir "${OUT_DIR}/plots_stage4_K${K}"
            popd > /dev/null
        fi

        for ORDER in "${ORDERS[@]}"; do
            local OUT_FILE="${GEN_DIR}/${TASK}_adaptive_K${K}_order${ORDER}.txt"
            local OUT_FILE_SEEDED="${OUT_FILE%.txt}_seed${SEEDS}.txt"

            if [ -f "${OUT_FILE_SEEDED}" ] || [ -f "${OUT_FILE}" ]; then
                echo ""
                echo " -> [${TASK}] K=${K} order=${ORDER}: output already exists -- skipping."
                continue
            fi

            echo ""
            echo "=== [${TASK}] K=${K} order=${ORDER}: Stage 5 -- Sampling via ${SAMPLE_SH} ==="

            MODEL_PATH="${MODEL_PATH}" \
            ADAPTIVE_SCHEDULE="${ADAPTIVE_SCHEDULE}" \
            DATASETS_DIR="${DATA_DIR}" \
            VOCAB_PATH="${VOCAB_PATH}" \
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
        done
    done
}

# ==============================================================================
# Task definitions
# ==============================================================================

run_task \
    "generation" \
    "${PROJECT_ROOT}/checkpoints/PLAIN_ema_0.9999_200000.pt" \
    "${PROJECT_ROOT}/checkpoints/adaptive_schedule/alpha_cumprod_step_190000.npy" \
    "${PROJECT_ROOT}/datasets/SMILES" \
    "" \
    "${PROJECT_ROOT}/sample.sh" \
    "train_val_256" \
    "${NUM_PROFILE_EXAMPLES}"

run_task \
    "retro" \
    "${PROJECT_ROOT}/checkpoints_retro/PLAIN_ema_0.9999_170000.pt" \
    "${PROJECT_ROOT}/checkpoints_retro/adaptive_schedule/alpha_cumprod_step_160000.npy" \
    "${PROJECT_ROOT}/datasets/RETRO" \
    "${PROJECT_ROOT}/datasets/RETRO/generate_vocab.txt" \
    "${PROJECT_ROOT}/retro_sample.sh" \
    "train" \
    "${NUM_PROFILE_EXAMPLES}"

run_task \
    "forward" \
    "${PROJECT_ROOT}/checkpoints_forward/PLAIN_ema_0.9999_140000.pt" \
    "${PROJECT_ROOT}/checkpoints_forward/adaptive_schedule/alpha_cumprod_step_130000.npy" \
    "${PROJECT_ROOT}/datasets/FORWARD" \
    "${PROJECT_ROOT}/datasets/FORWARD/generate_vocab.txt" \
    "${PROJECT_ROOT}/forward_sample.sh" \
    "train" \
    "${NUM_PROFILE_EXAMPLES}"

echo ""
echo "######################################################################"
echo "# All 3 tasks x ${#STEPS_LIST[@]} steps x ${#ORDERS[@]} orders complete."
echo "# Outputs under: ${GEN_DIR}"
echo "######################################################################"