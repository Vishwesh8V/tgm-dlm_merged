#!/usr/bin/env bash
# ==============================================================================
# Shell Script: Train / Resume TGM-DLM with Adaptive Noise Scheduling
# ==============================================================================

set -e

# ==============================================================================
# User Configuration (Edit your parameters here)
# ==============================================================================

CUDA_DEVICE="0"

# --- Resume paths ---
# Set RESUME_CHECKPOINT to the model .pt file to resume from, or "" to train from scratch.
RESUME_CHECKPOINT=""
# RESUME_CHECKPOINT="/home/ee/phd/eez248435/tgm-dlm_adanoise1/checkpoints/PLAIN_model130000.pt"

# Set ADAPTIVE_SCHEDULE to the corresponding alpha_cumprod_step_N.npy file, or "" to start fresh.
# Companion loss_history_step_N.npy / loss_count_step_N.npy in the same folder are auto-loaded.
ADAPTIVE_SCHEDULE=""
# ADAPTIVE_SCHEDULE="/home/ee/phd/eez248435/tgm-dlm_adanoise1/checkpoints/adaptive_schedule/alpha_cumprod_step_130000.npy"

# Number of steps to linearly warm up the LR after a resume.
# The optimizer state (Adam m/v) is not saved, so moment estimates start at zero
# on resume. This ramps LR from 10% → 100% of the correct annealed value over
# this many steps, re-stabilising Adam before returning to normal decay.
# Set to 0 to disable. Has no effect on fresh-start runs.
RESUME_WARMUP_STEPS=0

# --- Training hyperparameters ---
BATCH_SIZE=64
LR=0.00005
LR_ANNEAL_STEPS=200000
SAVE_INTERVAL=10000
LOG_INTERVAL=20

# --- Adaptive noise schedule ---
ADAPTIVE_NOISE=True
TOKEN_MAX_LENGTH=256
PAD_TOK_ID=0
LOSS_UPDATE_GRANU=50
SCHEDULE_UPDATE_STRIDE=10000

# --- Mixed-Space diffusion ---
LEARNED_MEAN_EMBED=True
DENOISE=True
DENOISE_RATE=0.2
REG_RATE=0.0

# ==============================================================================

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_SILENT=true
export OMPI_MCA_btl="^openib"
export PYTHONUNBUFFERED=1

# --- Determine directory structure FIRST (everything else depends on ROOT_DIR) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "${SCRIPT_DIR}/improved-diffusion/scripts" ]; then
    ROOT_DIR="${SCRIPT_DIR}"
    SCRIPTS_DIR="${SCRIPT_DIR}/improved-diffusion/scripts"
elif [ -f "${SCRIPT_DIR}/train.py" ]; then
    SCRIPTS_DIR="${SCRIPT_DIR}"
    ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
    echo "Error: Cannot locate training directory."
    exit 1
fi

# --- Build all paths as absolutes from ROOT_DIR ---
# This avoids any dependency on the shell's current working directory,
# which is why the old relative ../../checkpoints caused Permission denied.
CHECKPOINT_PATH="${ROOT_DIR}/checkpoints"
SAVE_DIR="${ROOT_DIR}/checkpoints/adaptive_schedule"
DATASET_DIR="${ROOT_DIR}/datasets/SMILES"

export PYTHONPATH="${ROOT_DIR}/improved-diffusion:${ROOT_DIR}/transformers/src:${ROOT_DIR}/transformers:${PYTHONPATH}"

# Check and auto-install local packages if missing
if ! python -c "import transformers" &>/dev/null; then
    echo "[Setup] Installing local custom transformers package in editable mode..."
    pip install -e "${ROOT_DIR}/transformers" --no-deps || true
fi
if ! python -c "import improved_diffusion" &>/dev/null; then
    echo "[Setup] Installing local improved-diffusion package in editable mode..."
    pip install -e "${ROOT_DIR}/improved-diffusion" --no-deps || true
fi

# Ensure output directories exist
mkdir -p "${SAVE_DIR}"

# Navigate to scripts directory where train.py lives
cd "${SCRIPTS_DIR}"

# Preprocess text embeddings if missing
if [ ! -f "${DATASET_DIR}/train_val_256_desc_states.pt" ]; then
    echo "[Data Setup] Preprocessing train_val_256 text descriptions with SciBERT..."
    python process_text.py -i train_val_256
fi
if [ ! -f "${DATASET_DIR}/test_desc_states.pt" ]; then
    echo "[Data Setup] Preprocessing test text descriptions with SciBERT..."
    python process_text.py -i test
fi

echo "======================================================================"
echo " Starting TGM-DLM Training / Resume (Adaptive Noise)"
echo "======================================================================"
echo " Project root       : ${ROOT_DIR}"
echo " Scripts dir        : ${SCRIPTS_DIR}"
echo " Resume checkpoint  : ${RESUME_CHECKPOINT:-"(None - training from scratch)"}"
echo " Adaptive schedule  : ${ADAPTIVE_SCHEDULE:-"(None - initialising default)"}"
echo " Resume LR warm-up  : ${RESUME_WARMUP_STEPS} steps (0 = disabled)"
echo " Batch size         : ${BATCH_SIZE}"
echo " LR                 : ${LR}  (anneal over ${LR_ANNEAL_STEPS} steps)"
echo " Save interval      : ${SAVE_INTERVAL}"
echo " Adaptive noise     : ${ADAPTIVE_NOISE}  (stride=${SCHEDULE_UPDATE_STRIDE}, granu=${LOSS_UPDATE_GRANU})"
echo " Learned Mean Embed : ${LEARNED_MEAN_EMBED}"
echo " Denoise (Discrete) : ${DENOISE} (rate=${DENOISE_RATE})"
echo " Reg Rate           : ${REG_RATE}"
echo " Checkpoint path    : ${CHECKPOINT_PATH}"
echo " Dataset dir        : ${DATASET_DIR}"
echo " CUDA device        : ${CUDA_VISIBLE_DEVICES}"
echo "======================================================================"

python train.py \
    --adaptive_noise    "${ADAPTIVE_NOISE}" \
    --token_max_length  "${TOKEN_MAX_LENGTH}" \
    --pad_tok_id        "${PAD_TOK_ID}" \
    --loss_update_granu "${LOSS_UPDATE_GRANU}" \
    --schedule_update_stride "${SCHEDULE_UPDATE_STRIDE}" \
    --learned_mean_embed "${LEARNED_MEAN_EMBED}" \
    --denoise           "${DENOISE}" \
    --denoise_rate      "${DENOISE_RATE}" \
    --reg_rate          "${REG_RATE}" \
    --save_dir          "${SAVE_DIR}" \
    --checkpoint_path   "${CHECKPOINT_PATH}" \
    --resume_checkpoint "${RESUME_CHECKPOINT}" \
    --adaptive_schedule_path "${ADAPTIVE_SCHEDULE}" \
    --batch_size        "${BATCH_SIZE}" \
    --lr                "${LR}" \
    --lr_anneal_steps   "${LR_ANNEAL_STEPS}" \
    --save_interval     "${SAVE_INTERVAL}" \
    --log_interval      "${LOG_INTERVAL}" \
    --resume_warmup_steps "${RESUME_WARMUP_STEPS}" \
    "$@"
