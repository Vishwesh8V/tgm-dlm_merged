#!/usr/bin/env bash
# ==============================================================================
# Shell Script: Intermediate-Step Sampling Wrapper for intermediate_sample.py
#
# Same idea as forward_sample.sh / retro_sample.sh / sample.sh: edit the
# variables below (or override any of them via env var, same pattern used
# throughout this repo), then just run:
#
#   bash intermediate_sample.sh
#
# Every value can also be overridden without touching this file, e.g.:
#   SAMPLER=token_adaptive STEP_MATRIX_PATH=/path/to/schedule.csv \
#   NUM_MILESTONES=6 LENGTH_RANGE=long \
#   bash intermediate_sample.sh
#
# IMPORTANT: MIXED_SPACE / ADAPTIVE-NOISING FLAGS BELOW MUST MATCH HOW THE
# CHECKPOINT YOU POINT MODEL_PATH AT WAS ACTUALLY TRAINED. If you get these
# wrong (e.g. leave LEARNED_MEAN_EMBED/DENOISE off for a checkpoint that
# was trained with them on, or skip ADAPTIVE_SCHEDULE for one trained with
# a per-position adaptive schedule), inference silently mismatches training
# and sample quality degrades -- there is no error, it just quietly gets
# worse. The defaults below are pre-filled to match forward_sample.sh's own
# defaults for the FORWARD checkpoint; double check against retro_sample.sh
# / sample.sh's defaults if you point this at the retro/generation
# checkpoints instead.
# ==============================================================================

set -e

# --- Environment Setup ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# ==============================================================================
# User Configuration (edit here, or override via env var -- see header)
# ==============================================================================

# --- Checkpoint / data (the three things you said you keep changing) ---
MODEL_PATH="${MODEL_PATH:-/home/ee/phd/eez248435/tgm-dlm_merged/checkpoints_forward/PLAIN_ema_0.9999_140000.pt}"
DATA_DIR="${DATA_DIR:-/home/ee/phd/eez248435/tgm-dlm_merged/datasets/FORWARD}"
VOCAB_PATH="${VOCAB_PATH:-/home/ee/phd/eez248435/tgm-dlm_merged/datasets/FORWARD/generate_vocab.txt}"

# --- Mixed-Space Diffusion (must match what MODEL_PATH was trained with --
# see forward_sample.sh's defaults for the forward checkpoint) ---
LEARNED_MEAN_EMBED="${LEARNED_MEAN_EMBED:-True}"
DENOISE="${DENOISE:-True}"
DENOISE_RATE="${DENOISE_RATE:-0.2}"
REG_RATE="${REG_RATE:-0.1}"

# --- Adaptive noising schedule (per-position). Set to "none" to disable
# and use the plain uniform schedule instead. ---
ADAPTIVE_SCHEDULE="${ADAPTIVE_SCHEDULE:-/home/ee/phd/eez248435/tgm-dlm_merged/checkpoints_forward/adaptive_schedule/alpha_cumprod_step_130000.npy}"

# --- Sampling run ---
SPLIT="${SPLIT:-test}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
SEED="${SEED:-108}"
OUTPUTDIR="${OUTPUTDIR:-../../generation_outputs/forward_process/intermediate}"

# --- Sampler: ancestral | ddim | dpm_solver | token_adaptive ---
SAMPLER="${SAMPLER:-dpm_solver}"
TIMESTEP_RESPACING="${TIMESTEP_RESPACING:-1}"          # stride; used by 'ancestral'/'ddim' only (1 = full 2000 steps)
DPM_SOLVER_STEPS="${DPM_SOLVER_STEPS:-10}"             # used by 'dpm_solver' only
DPM_SOLVER_ORDER="${DPM_SOLVER_ORDER:-2}"              # used by 'dpm_solver' and 'token_adaptive'
DPM_SOLVER_METHOD="${DPM_SOLVER_METHOD:-multistep}"    # used by 'dpm_solver' only
STEP_MATRIX_PATH="${STEP_MATRIX_PATH:-}"               # required for 'token_adaptive'
TOKEN_ADAPTIVE_STEPS="${TOKEN_ADAPTIVE_STEPS:-10}"     # used by 'token_adaptive' only

# --- Milestones: EITHER an evenly-spaced count, OR an explicit list ---
# Leave MILESTONE_STEPS empty to use NUM_MILESTONES (evenly spaced, e.g. 4
# -> [total_calls, .75x, .5x, .25x, 0]). Set MILESTONE_STEPS (e.g. "7,6,2")
# to pick exact calls-remaining labels instead -- this takes priority.
NUM_MILESTONES="${NUM_MILESTONES:-4}"
MILESTONE_STEPS="${MILESTONE_STEPS:-}"

# --- Length-range filter: '' = no filtering, else one or more of
# short,medium,long,very_long,longest (comma-separated) ---
LENGTH_RANGE="${LENGTH_RANGE:-}"
LENGTH_CACHE_DIR="${LENGTH_CACHE_DIR:-}"               # '' = default cache location
NO_LENGTH_CACHE="${NO_LENGTH_CACHE:-False}"
OVERWRITE="${OVERWRITE:-False}"        # False (default) = never clobber an existing output
                                         # file for this outputdir prefix; skip with a message instead.

# ==============================================================================
# Resolve paths (same pattern as forward_sample.sh)
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/improved-diffusion/scripts/intermediate_sample.py" ]; then
    PROJECT_ROOT="${SCRIPT_DIR}"
elif [ -f "${SCRIPT_DIR}/scripts/intermediate_sample.py" ]; then
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    PROJECT_ROOT="${SCRIPT_DIR}"
fi
SCRIPTS_DIR="${PROJECT_ROOT}/improved-diffusion/scripts"

export PYTHONPATH="${PROJECT_ROOT}/improved-diffusion:${PROJECT_ROOT}/transformers/src:${PYTHONPATH}"

mkdir -p "$(dirname "${OUTPUTDIR}")" 2>/dev/null || true
cd "${SCRIPTS_DIR}"

# ==============================================================================
# Validate sampler-specific requirements up front (fail fast, clear message)
# ==============================================================================
if [ "${SAMPLER}" = "token_adaptive" ]; then
    if [ -z "${STEP_MATRIX_PATH}" ] || [ ! -f "${STEP_MATRIX_PATH}" ]; then
        echo "ERROR: --sampler token_adaptive requires STEP_MATRIX_PATH to point at an" >&2
        echo "       existing schedule file (from trajectory_analysis.py's 'allocate' stage)." >&2
        exit 1
    fi
fi

SCHEDULE_ARG=""
if [ -n "${ADAPTIVE_SCHEDULE}" ] && [ "${ADAPTIVE_SCHEDULE}" != "none" ] && [ -f "${ADAPTIVE_SCHEDULE}" ]; then
    SCHEDULE_ARG="--adaptive_schedule_path ${ADAPTIVE_SCHEDULE}"
elif [ -n "${ADAPTIVE_SCHEDULE}" ] && [ "${ADAPTIVE_SCHEDULE}" != "none" ]; then
    echo "WARNING: ADAPTIVE_SCHEDULE=${ADAPTIVE_SCHEDULE} does not exist -- falling back" >&2
    echo "         to the plain (non-adaptive) noise schedule. If MODEL_PATH was trained" >&2
    echo "         with a per-position adaptive schedule, this WILL degrade results." >&2
fi

MILESTONE_ARG=""
if [ -n "${MILESTONE_STEPS}" ]; then
    MILESTONE_ARG="--milestone_steps ${MILESTONE_STEPS}"
fi

LENGTH_RANGE_ARG=""
if [ -n "${LENGTH_RANGE}" ]; then
    LENGTH_RANGE_ARG="--length_range ${LENGTH_RANGE}"
fi
LENGTH_CACHE_ARG=""
if [ -n "${LENGTH_CACHE_DIR}" ]; then
    LENGTH_CACHE_ARG="--length_cache_dir ${LENGTH_CACHE_DIR}"
fi
NO_CACHE_ARG=""
if [ "${NO_LENGTH_CACHE}" = "True" ] || [ "${NO_LENGTH_CACHE}" = "true" ]; then
    NO_CACHE_ARG="--no_length_cache True"
fi

# ==============================================================================
# Echo the run config so a mismatch is obvious before you wait on the GPU
# ==============================================================================
echo "======================================================================"
echo " Intermediate-Step Sampling"
echo "======================================================================"
echo " Model Checkpoint  : ${MODEL_PATH}"
echo " Data Directory    : ${DATA_DIR}"
echo " Vocab Path        : ${VOCAB_PATH}"
echo " Mixed-Space        : learned_mean_embed=${LEARNED_MEAN_EMBED} denoise=${DENOISE}"\
     "denoise_rate=${DENOISE_RATE} reg_rate=${REG_RATE}"
echo " Adaptive Schedule  : ${ADAPTIVE_SCHEDULE:-"(none -- uniform baseline)"}"
echo " Sampler            : ${SAMPLER}"
case "${SAMPLER}" in
  ancestral)      echo "   timestep_respacing (stride) : ${TIMESTEP_RESPACING}" ;;
  ddim)           echo "   timestep_respacing (stride) : ${TIMESTEP_RESPACING}" ;;
  dpm_solver)     echo "   steps=${DPM_SOLVER_STEPS} order=${DPM_SOLVER_ORDER} method=${DPM_SOLVER_METHOD}" ;;
  token_adaptive) echo "   step_matrix=${STEP_MATRIX_PATH} K=${TOKEN_ADAPTIVE_STEPS} order=${DPM_SOLVER_ORDER}" ;;
esac
if [ -n "${MILESTONE_STEPS}" ]; then
    echo " Milestones         : manual = ${MILESTONE_STEPS}"
else
    echo " Milestones         : ${NUM_MILESTONES} evenly-spaced (+final)"
fi
echo " Length Range       : ${LENGTH_RANGE:-"(none -- no filtering)"}"
echo " Num Samples        : ${NUM_SAMPLES}   Seed: ${SEED}   Split: ${SPLIT}"
echo " Output Prefix      : ${OUTPUTDIR}   (overwrite=${OVERWRITE})"
echo "======================================================================"

python intermediate_sample.py \
  --model_path "${MODEL_PATH}" \
  --data_dir "${DATA_DIR}" \
  --vocab_path "${VOCAB_PATH}" \
  --learned_mean_embed "${LEARNED_MEAN_EMBED}" \
  --denoise "${DENOISE}" \
  --denoise_rate "${DENOISE_RATE}" \
  --reg_rate "${REG_RATE}" \
  ${SCHEDULE_ARG} \
  --split "${SPLIT}" \
  --num_samples "${NUM_SAMPLES}" \
  --seed "${SEED}" \
  --sampler "${SAMPLER}" \
  --timestep_respacing "${TIMESTEP_RESPACING}" \
  --dpm_solver_steps "${DPM_SOLVER_STEPS}" \
  --dpm_solver_order "${DPM_SOLVER_ORDER}" \
  --dpm_solver_method "${DPM_SOLVER_METHOD}" \
  --token_adaptive_steps "${TOKEN_ADAPTIVE_STEPS}" \
  $([ -n "${STEP_MATRIX_PATH}" ] && echo "--step_matrix_path ${STEP_MATRIX_PATH}") \
  --num_milestones "${NUM_MILESTONES}" \
  ${MILESTONE_ARG} \
  ${LENGTH_RANGE_ARG} \
  ${LENGTH_CACHE_ARG} \
  ${NO_CACHE_ARG} \
  --overwrite "${OVERWRITE}" \
  --outputdir "${OUTPUTDIR}"

echo ""
echo "Done. See ${OUTPUTDIR}_m*.txt and ${OUTPUTDIR}_FINAL.txt"