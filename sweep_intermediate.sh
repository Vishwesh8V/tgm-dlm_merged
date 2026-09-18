#!/usr/bin/env bash
# ==============================================================================
# sweep_intermediate.sh
#
# Runs intermediate_sample.py across the full grid:
#   dataset (3: FORWARD, RETRO, SMILES)
#   x length_range (5: short, medium, long, very_long, longest)
#   x { ancestral(2000) }  U  { ddim, dpm_solver } x steps(7: 500,200,100,10,5,4,2)
#     = 1 + 7*2 = 15 (solver x step) combos
#   -> 3 * 5 * 15 = 225 total runs
#
# (ancestral is NOT crossed with the ddim/dpm_solver solver axis -- it's a
# single baseline run per (dataset, length_range), not duplicated under
# both solver tags, per your call.)
#
# token_adaptive is intentionally excluded (no J/step_matrix trajectory
# inference in this sweep).
#
# Loop order is dataset -> length_range -> (solver, steps), with dataset
# OUTERMOST: the first run for a given dataset triggers intermediate_
# sample.py's length-bucket cache build (a scan of the whole split), and
# every subsequent run for that same dataset -- regardless of solver,
# steps, or length_range -- reuses that cache instead of rescanning.
#
# Edit dataset_config.sh (checkpoint/vocab/adaptive-schedule paths per
# dataset) and milestone_config.sh (milestone spec per step count) before
# running this. Everything else can be overridden via env var, e.g.:
#   DRY_RUN=true bash sweep_intermediate.sh       # print commands, run nothing
#   NUM_SAMPLES=4 bash sweep_intermediate.sh       # quick smoke test
# ==============================================================================

set -uo pipefail   # NOTE: no -e -- one failed run must not kill the other 254

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
SCRIPTS_DIR="${PROJECT_ROOT}/improved-diffusion/scripts"
INTERMEDIATE_PY="${SCRIPTS_DIR}/intermediate_sample.py"
export PYTHONPATH="${PROJECT_ROOT}/improved-diffusion:${PROJECT_ROOT}/transformers/src:${PYTHONPATH:-}"

source "${PROJECT_ROOT}/dataset_config.sh"
source "${PROJECT_ROOT}/milestone_config.sh"

# ==============================================================================
# Shared config (same across all datasets, per your confirmation) -- override
# via env var same as the single-run wrapper.
# ==============================================================================
LEARNED_MEAN_EMBED="${LEARNED_MEAN_EMBED:-True}"
DENOISE="${DENOISE:-True}"
DENOISE_RATE="${DENOISE_RATE:-0.2}"
REG_RATE="${REG_RATE:-0.1}"

SPLIT="${SPLIT:-test}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
SEED="${SEED:-108}"
DPM_SOLVER_ORDER="${DPM_SOLVER_ORDER:-2}"
DPM_SOLVER_METHOD="${DPM_SOLVER_METHOD:-multistep}"

VIS_ROOT="${VIS_ROOT:-${PROJECT_ROOT}/visualization_gen_outputs}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/sweep_logs}"
mkdir -p "${LOG_ROOT}"
FAIL_LOG="${LOG_ROOT}/failures.tsv"
: > "${FAIL_LOG}"

DRY_RUN="${DRY_RUN:-false}"          # true -> print commands, run nothing
SKIP_EXISTING="${SKIP_EXISTING:-true}"  # true -> skip a combo (no python invocation at all)
                                         # if its _FINAL.txt already exists (resume support)
OVERWRITE="${OVERWRITE:-False}"      # passed straight to intermediate_sample.py's --overwrite.
                                         # If False (default), a python invocation that DOES run
                                         # will still refuse to clobber any existing output file
                                         # for that combo (per-file skip, not a crash) -- this is
                                         # a second, independent safety net underneath
                                         # SKIP_EXISTING, in case some but not all files for a
                                         # combo already exist (e.g. a killed job) or you reran
                                         # with SKIP_EXISTING=false on purpose. To genuinely force
                                         # a full redo of specific combos: SKIP_EXISTING=false
                                         # AND OVERWRITE=True together -- SKIP_EXISTING=false with
                                         # OVERWRITE left False just re-runs the GPU work and then
                                         # skips every write, which is safe but wasted compute.

# ==============================================================================
# Grid definition
# ==============================================================================
DATASETS=(FORWARD RETRO SMILES)
LENGTH_RANGES=(short medium long very_long longest)

DPM_STEP_COUNTS=(500 200 100 10 5 4 2)
RUN_CONFIGS=("ancestral:2000")
for s in "${DPM_STEP_COUNTS[@]}"; do
    RUN_CONFIGS+=("ddim:${s}")
    RUN_CONFIGS+=("dpm_solver:${s}")
done

TOTAL_RUNS=$(( ${#DATASETS[@]} * ${#LENGTH_RANGES[@]} * ${#RUN_CONFIGS[@]} ))
echo "======================================================================"
echo " Sweep: ${#DATASETS[@]} datasets x ${#LENGTH_RANGES[@]} length_ranges x ${#RUN_CONFIGS[@]} (solver,steps) combos = ${TOTAL_RUNS} runs"
echo " DRY_RUN=${DRY_RUN}  SKIP_EXISTING=${SKIP_EXISTING}  OVERWRITE=${OVERWRITE}"
echo " Output root: ${VIS_ROOT}"
echo " Logs:        ${LOG_ROOT}  (failures listed in ${FAIL_LOG})"
echo "======================================================================"

RUN_IDX=0
SKIPPED_DATASETS=()

for DATASET in "${DATASETS[@]}"; do
    MODEL_PATH="${MODEL_PATH_BY_DATASET[$DATASET]:-}"
    VOCAB_PATH="${VOCAB_PATH_BY_DATASET[$DATASET]:-}"
    DATA_DIR="${DATA_DIR_BY_DATASET[$DATASET]:-}"
    [ -z "${DATA_DIR}" ] && DATA_DIR="${PROJECT_ROOT}/datasets/${DATASET}"
    ADAPTIVE_SCHEDULE="${ADAPTIVE_SCHEDULE_BY_DATASET[$DATASET]:-}"

    # Skip a whole dataset cleanly if it hasn't been configured yet, rather
    # than failing 5*17=85 times into the log with the same missing-path error.
    if [ -z "${MODEL_PATH}" ] || [[ "${MODEL_PATH}" == FILL_ME_IN* ]] || { [ "${DRY_RUN}" != "true" ] && [ ! -f "${MODEL_PATH}" ]; }; then
        echo "SKIPPING dataset ${DATASET}: MODEL_PATH not configured/found (${MODEL_PATH}) -- edit dataset_config.sh"
        SKIPPED_DATASETS+=("${DATASET}")
        RUN_IDX=$(( RUN_IDX + ${#LENGTH_RANGES[@]} * ${#RUN_CONFIGS[@]} ))
        continue
    fi

    ADAPTIVE_ARGS=()
    if [ -n "${ADAPTIVE_SCHEDULE}" ]; then
        if [ "${DRY_RUN}" = "true" ] || [ -f "${ADAPTIVE_SCHEDULE}" ]; then
            ADAPTIVE_ARGS=(--adaptive_schedule_path "${ADAPTIVE_SCHEDULE}")
        else
            echo "WARNING [${DATASET}]: ADAPTIVE_SCHEDULE_BY_DATASET points at a missing file" \
                 "(${ADAPTIVE_SCHEDULE}) -- falling back to the plain uniform schedule for this dataset."
        fi
    fi

    for LENGTH_RANGE in "${LENGTH_RANGES[@]}"; do
        OUT_SUBDIR="${VIS_ROOT}/${DATASET}/${LENGTH_RANGE}"
        mkdir -p "${OUT_SUBDIR}"

        for RUN_CONFIG in "${RUN_CONFIGS[@]}"; do
            RUN_IDX=$(( RUN_IDX + 1 ))
            SOLVER="${RUN_CONFIG%%:*}"
            STEPS="${RUN_CONFIG##*:}"
            MSPEC="${MILESTONES[${STEPS}]:-4}"

            MILESTONE_ARGS=()
            if [[ "${MSPEC}" == *","* ]]; then
                MILESTONE_ARGS=(--milestone_steps "${MSPEC}")
            else
                MILESTONE_ARGS=(--num_milestones "${MSPEC}")
            fi

            SAMPLER_ARGS=()
            case "${SOLVER}" in
                ancestral|ddim)
                    # intermediate_sample.py's --timestep_respacing is a STRIDE
                    # (use_timesteps = range(0, 2000, stride)), not a total step
                    # count, so convert here: stride = 2000 / desired_steps.
                    # All values in DPM_STEP_COUNTS (plus 2000 for ancestral)
                    # divide 2000 evenly -- if you add a step count that
                    # doesn't, this integer division will silently give you a
                    # different total step count than STEPS says, so check
                    # `2000 % STEPS == 0` before adding one.
                    STRIDE=$(( 2000 / STEPS ))
                    SAMPLER_ARGS=(--sampler "${SOLVER}" --timestep_respacing "${STRIDE}")
                    ;;
                dpm_solver)
                    SAMPLER_ARGS=(--sampler dpm_solver --dpm_solver_steps "${STEPS}"
                                  --dpm_solver_order "${DPM_SOLVER_ORDER}"
                                  --dpm_solver_method "${DPM_SOLVER_METHOD}")
                    ;;
            esac

            # One subfolder per (solver, steps) combo, e.g.
            # .../FORWARD/short/ddim_500/ddim_500_m4_t1499.txt -- keeps each
            # length_range folder from accumulating hundreds of flat files,
            # while filenames inside stay self-descriptive on their own.
            COMBO_DIR="${OUT_SUBDIR}/${SOLVER}_${STEPS}"
            OUTPUTDIR="${COMBO_DIR}/${SOLVER}_${STEPS}"
            TAG="${DATASET}/${LENGTH_RANGE}/${SOLVER}_${STEPS}"
            RUN_LOG="${LOG_ROOT}/${DATASET}_${LENGTH_RANGE}_${SOLVER}_${STEPS}.log"

            if [ "${SKIP_EXISTING}" = "true" ] && [ -f "${OUTPUTDIR}_FINAL.txt" ]; then
                echo "[${RUN_IDX}/${TOTAL_RUNS}] ${TAG} -- already done, skipping (SKIP_EXISTING=true)"
                continue
            fi

            CMD=(python "${INTERMEDIATE_PY}"
                 --model_path "${MODEL_PATH}"
                 --data_dir "${DATA_DIR}"
                 --vocab_path "${VOCAB_PATH}"
                 --learned_mean_embed "${LEARNED_MEAN_EMBED}"
                 --denoise "${DENOISE}"
                 --denoise_rate "${DENOISE_RATE}"
                 --reg_rate "${REG_RATE}"
                 "${ADAPTIVE_ARGS[@]}"
                 --split "${SPLIT}"
                 --num_samples "${NUM_SAMPLES}"
                 --seed "${SEED}"
                 "${SAMPLER_ARGS[@]}"
                 "${MILESTONE_ARGS[@]}"
                 --length_range "${LENGTH_RANGE}"
                 --overwrite "${OVERWRITE}"
                 --outputdir "${OUTPUTDIR}")

            echo "[${RUN_IDX}/${TOTAL_RUNS}] ${TAG}  (milestones=${MSPEC})"

            if [ "${DRY_RUN}" = "true" ]; then
                printf '    DRY_RUN:'; printf ' %q' "${CMD[@]}"; printf '\n'
                continue
            fi

            START_TS=$(date +%s)
            "${CMD[@]}" > "${RUN_LOG}" 2>&1
            STATUS=$?
            ELAPSED=$(( $(date +%s) - START_TS ))

            if [ ${STATUS} -ne 0 ]; then
                echo "    -> FAILED (exit ${STATUS}, ${ELAPSED}s) -- see ${RUN_LOG}"
                printf '%s\t%s\t%s\n' "${TAG}" "${STATUS}" "${RUN_LOG}" >> "${FAIL_LOG}"
            else
                echo "    -> ok (${ELAPSED}s)"
            fi
        done
    done
done

echo "======================================================================"
if [ ${#SKIPPED_DATASETS[@]} -gt 0 ]; then
    echo " Skipped datasets (not configured): ${SKIPPED_DATASETS[*]}"
fi
if [ -s "${FAIL_LOG}" ]; then
    N_FAIL=$(wc -l < "${FAIL_LOG}")
    echo " ${N_FAIL} run(s) FAILED -- see ${FAIL_LOG} for the list and per-run logs."
else
    echo " All executed runs completed successfully."
fi
echo " Outputs under: ${VIS_ROOT}"
echo "======================================================================"