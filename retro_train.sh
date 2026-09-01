#!/usr/bin/env bash
# ==============================================================================
# Shell Script: Train TGM-DLM on the Phase-1 retrosynthesis dataset
#
# Identical training pipeline/diffusion math to train.sh (see
# RETRO_ADAPTATION_PLAN.md) -- the only functional difference is which
# dataset directory train.py and process_text.py read from, via --data_dir /
# --dataset_dir (previously hardcoded to datasets/SMILES; see train.py /
# process_text.py diffs). Point RAW_RETRO_INPUT at your source file once and
# this builds datasets/RETRO the first time it's run.
# ==============================================================================

set -e

CUDA_DEVICE="0"

# --- Path to your raw retrosynthesis source file, and its format ---
# format: molinstructions | uspto50k  (see build_retro_dataset.py --help)
RAW_RETRO_INPUT="/home/ee/phd/eez248435/tgm-dlm_merged/datasets/RETRO/retrosynthesis.json"
RAW_RETRO_FORMAT="molinstructions"

RESUME_CHECKPOINT=""
ADAPTIVE_SCHEDULE=""
RESUME_WARMUP_STEPS=0

BATCH_SIZE=64
LR=0.00005
LR_ANNEAL_STEPS=200000
SAVE_INTERVAL=10000
LOG_INTERVAL=20

ADAPTIVE_NOISE=True
TOKEN_MAX_LENGTH=256
PAD_TOK_ID=0
LOSS_UPDATE_GRANU=50
SCHEDULE_UPDATE_STRIDE=10000

LEARNED_MEAN_EMBED=True
DENOISE=True
DENOISE_RATE=0.2
REG_RATE=0.0

# Which split (of train/validation/test.txt in DATASET_DIR) to train on.
# Retro data has real train/validation/test splits from metadata.split -- unlike
# ChEBI captioning's train_val_256, there's no pre-merged split to default to.
TRAIN_SPLIT="train"

# ==============================================================================

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_SILENT=true
export OMPI_MCA_btl="^openib"
export PYTHONUNBUFFERED=1

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

CHECKPOINT_PATH="${ROOT_DIR}/checkpoints_retro"
SAVE_DIR="${ROOT_DIR}/checkpoints_retro/adaptive_schedule"
DATASET_DIR="${ROOT_DIR}/datasets/RETRO"

export PYTHONPATH="${ROOT_DIR}/improved-diffusion:${ROOT_DIR}/transformers/src:${ROOT_DIR}/transformers:${PYTHONPATH}"

if ! python -c "import transformers" &>/dev/null; then
    pip install -e "${ROOT_DIR}/transformers" --no-deps || true
fi
if ! python -c "import improved_diffusion" &>/dev/null; then
    pip install -e "${ROOT_DIR}/improved-diffusion" --no-deps || true
fi

mkdir -p "${SAVE_DIR}"
cd "${SCRIPTS_DIR}"

# --- One-time dataset build: raw retro file -> DATASET_DIR/{train,validation,test}.txt + vocab ---
if [ ! -f "${DATASET_DIR}/train.txt" ]; then
    if [ -z "${RAW_RETRO_INPUT}" ]; then
        echo "Error: ${DATASET_DIR}/train.txt does not exist and RAW_RETRO_INPUT is unset."
        echo "Set RAW_RETRO_INPUT to your raw retrosynthesis file (and RAW_RETRO_FORMAT) at the top of this script."
        exit 1
    fi
    echo "[Data Setup] Building ${DATASET_DIR} from ${RAW_RETRO_INPUT} (format=${RAW_RETRO_FORMAT})..."
    python build_retro_dataset.py --input "${RAW_RETRO_INPUT}" --format "${RAW_RETRO_FORMAT}" --out_dir "${DATASET_DIR}"
    python build_retro_vocab.py --dataset_dir "${DATASET_DIR}"
fi

# --- Preprocess text embeddings if missing (SciBERT-encodes the DESC column, i.e. product SMILES) ---
if [ ! -f "${DATASET_DIR}/${TRAIN_SPLIT}_desc_states.pt" ]; then
    echo "[Data Setup] Preprocessing ${TRAIN_SPLIT} product-SMILES conditioning with SciBERT..."
    python process_text.py -i "${TRAIN_SPLIT}" --dataset_dir "${DATASET_DIR}"
fi
if [ -f "${DATASET_DIR}/test.txt" ] && [ ! -f "${DATASET_DIR}/test_desc_states.pt" ]; then
    echo "[Data Setup] Preprocessing test product-SMILES conditioning with SciBERT..."
    python process_text.py -i test --dataset_dir "${DATASET_DIR}"
fi

echo "======================================================================"
echo " Starting TGM-DLM Training (Phase 1 Retrosynthesis)"
echo "======================================================================"
echo " Dataset dir        : ${DATASET_DIR}"
echo " Train split         : ${TRAIN_SPLIT}"
echo " Resume checkpoint  : ${RESUME_CHECKPOINT:-"(None - training from scratch)"}"
echo " Batch size         : ${BATCH_SIZE}"
echo " LR                 : ${LR}  (anneal over ${LR_ANNEAL_STEPS} steps)"
echo " Checkpoint path    : ${CHECKPOINT_PATH}"
echo "======================================================================"

python train.py \
    --data_dir           "${DATASET_DIR}/" \
    --train_split        "${TRAIN_SPLIT}" \
    --adaptive_noise     "${ADAPTIVE_NOISE}" \
    --token_max_length   "${TOKEN_MAX_LENGTH}" \
    --pad_tok_id         "${PAD_TOK_ID}" \
    --loss_update_granu  "${LOSS_UPDATE_GRANU}" \
    --schedule_update_stride "${SCHEDULE_UPDATE_STRIDE}" \
    --learned_mean_embed "${LEARNED_MEAN_EMBED}" \
    --denoise            "${DENOISE}" \
    --denoise_rate       "${DENOISE_RATE}" \
    --reg_rate           "${REG_RATE}" \
    --save_dir           "${SAVE_DIR}" \
    --checkpoint_path    "${CHECKPOINT_PATH}" \
    --resume_checkpoint  "${RESUME_CHECKPOINT}" \
    --adaptive_schedule_path "${ADAPTIVE_SCHEDULE}" \
    --batch_size         "${BATCH_SIZE}" \
    --lr                 "${LR}" \
    --lr_anneal_steps    "${LR_ANNEAL_STEPS}" \
    --save_interval      "${SAVE_INTERVAL}" \
    --log_interval       "${LOG_INTERVAL}" \
    --resume_warmup_steps "${RESUME_WARMUP_STEPS}" \
    "$@"