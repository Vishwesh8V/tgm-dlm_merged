#!/usr/bin/env bash
# ==============================================================================
# Shell Script: Generation-Only UNIFORM Trajectory Sampling -- 200 Test Examples
#
# Counterpart to run_generation_200test.sh, but for uniform trajectory
# (--use_dpm_solver) instead of token-adaptive trajectory (--step_matrix_path).
# Uniform trajectory needs none of the Stage 1-4 profiling/schedule-allocation
# pipeline -- dpm_solver_sample_loop picks its own log-SNR-spaced steps
# directly from the checkpoint's noise schedule at sampling time, so this
# script is just sample.sh driven via env-var overrides, looped over K and
# order. (For an adaptive-noise-trained checkpoint like this one,
# dpm_solver_sample_loop internally routes through the same per-position-
# aware solver the token-adaptive path uses, tiled uniformly across
# positions -- see the gaussian_diffusion.py/dpm_solver.py fixes earlier in
# this conversation. Nothing extra needs doing here for that to be correct.)
#
# K in {2, 3, 4, 5, 10}, both orders (DPM=2, DDIM=1) -- same steps/orders as
# the adaptive-trajectory 200-sample run, so the two are directly comparable.
#
# Output goes to generation_outputs/uniform_sweep_200test/generation/ -- a
# new folder, distinct from both the adaptive-trajectory sweeps
# (trajectory_sweep*) and anything else.
#
# PLACE THIS FILE AT THE ROOT OF tgm-dlm_reduced_step, alongside sample.sh.
# ==============================================================================

set -e

# ==============================================================================
# User Configuration
# ==============================================================================
STEPS_LIST=(2 3 4 5 10)
ORDERS=(2 1)   # 2 = DPM-Solver++(2M); 1 = DDIM-equivalent

NUM_SAMPLES=200
BATCH_SIZE=64
SEEDS="108,208,308"   # comma-separated; skip-check below verifies EVERY seed's file individually

OUTPUT_SUBDIR="uniform_sweep_200test_3seed"

MODEL_PATH_DEFAULT="checkpoints/PLAIN_ema_0.9999_200000.pt"
ADAPTIVE_SCHEDULE_DEFAULT="checkpoints/adaptive_schedule/alpha_cumprod_step_190000.npy"
DATA_DIR_DEFAULT="datasets/SMILES"

LEARNED_MEAN_EMBED=True
DENOISE=True
DENOISE_RATE=0.2
REG_RATE=0.1
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

MODEL_PATH="${PROJECT_ROOT}/${MODEL_PATH_DEFAULT}"
ADAPTIVE_SCHEDULE="${PROJECT_ROOT}/${ADAPTIVE_SCHEDULE_DEFAULT}"
DATA_DIR="${PROJECT_ROOT}/${DATA_DIR_DEFAULT}"
SAMPLE_SH="${PROJECT_ROOT}/sample.sh"

GEN_DIR="${PROJECT_ROOT}/generation_outputs/${OUTPUT_SUBDIR}/generation"

if [ ! -f "${SAMPLE_SH}" ]; then
    echo "ERROR: ${SAMPLE_SH} not found." >&2
    exit 1
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="${PROJECT_ROOT}/improved-diffusion:${PROJECT_ROOT}/transformers/src:${PYTHONPATH}"

mkdir -p "${GEN_DIR}"

echo "======================================================================"
echo " Generation-Only UNIFORM Trajectory Sampling -- 200 Test Examples"
echo " K values         : ${STEPS_LIST[*]}"
echo " Orders           : ${ORDERS[*]} (2=DPM, 1=DDIM)"
echo " Model Checkpoint : ${MODEL_PATH}"
echo " Output dir       : ${GEN_DIR}"
echo "======================================================================"

GENERATED_FILES=()

for K in "${STEPS_LIST[@]}"; do
    for ORDER in "${ORDERS[@]}"; do
        OUT_FILE="${GEN_DIR}/K${K}_order${ORDER}.txt"

        # Check EVERY individual seed's expected file, not the whole SEEDS
        # string glued onto one filename -- sample.sh writes one file per
        # seed (K2_order2_seed108.txt, K2_order2_seed208.txt, ...), so a
        # single-file check like the old "${OUT_FILE%.txt}_seed${SEEDS}.txt"
        # never matches once SEEDS has more than one value, and would
        # neither skip correctly nor warn you it can't.
        IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"
        ALL_SEEDS_EXIST=true
        for s in "${SEED_ARR[@]}"; do
            if [ ! -f "${OUT_FILE%.txt}_seed${s}.txt" ]; then
                ALL_SEEDS_EXIST=false
                break
            fi
        done

        if [ "${ALL_SEEDS_EXIST}" = "true" ]; then
            echo " -> K=${K} order=${ORDER}: output already exists for every seed (${SEEDS}) -- skipping."
            continue
        fi

        echo "" && echo "=== K=${K} order=${ORDER}: uniform-trajectory sampling (200 test examples) ==="

        MODEL_PATH="${MODEL_PATH}" \
        ADAPTIVE_SCHEDULE="${ADAPTIVE_SCHEDULE}" \
        DATASETS_DIR="${DATA_DIR}" \
        OUTPUT_FILE="${OUT_FILE}" \
        NUM_SAMPLES="${NUM_SAMPLES}" \
        BATCH_SIZE="${BATCH_SIZE}" \
        SEEDS="${SEEDS}" \
        USE_DPM_SOLVER="true" \
        DPM_SOLVER_STEPS="${K}" \
        DPM_SOLVER_ORDER="${ORDER}" \
        STEP_MATRIX_PATH="" \
        LEARNED_MEAN_EMBED="${LEARNED_MEAN_EMBED}" \
        DENOISE="${DENOISE}" \
        DENOISE_RATE="${DENOISE_RATE}" \
        REG_RATE="${REG_RATE}" \
        bash "${SAMPLE_SH}"

        for s in "${SEED_ARR[@]}"; do
            GENERATED_FILES+=("${OUT_FILE%.txt}_seed${s}.txt")
        done
    done
done

echo ""
echo "======================================================================"
echo " Done. Outputs under: ${GEN_DIR}"
for f in "${GENERATED_FILES[@]}"; do
    echo "   -> $f"
done
echo "======================================================================"