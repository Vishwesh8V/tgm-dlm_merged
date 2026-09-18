#!/usr/bin/env bash
# ==============================================================================
# dataset_config.sh
#
# Sourced by sweep_intermediate.sh. Fill in the checkpoint/vocab/adaptive-
# schedule path for each of the three datasets below -- these are the only
# things that differ per dataset; mixed-space settings (learned_mean_embed/
# denoise/denoise_rate/reg_rate) are shared across all three (edit those in
# sweep_intermediate.sh itself if that ever stops being true).
#
# DATA_DIR is auto-derived as <project_root>/datasets/<NAME> unless you
# override it here. Set ADAPTIVE_SCHEDULE_PATH to "" (empty) for a dataset
# that should use the plain uniform schedule instead of a per-position one.
# ==============================================================================

declare -A MODEL_PATH_BY_DATASET
declare -A VOCAB_PATH_BY_DATASET
declare -A DATA_DIR_BY_DATASET
declare -A ADAPTIVE_SCHEDULE_BY_DATASET

# --- FORWARD (reaction product prediction) ---
MODEL_PATH_BY_DATASET[FORWARD]="/home/ee/phd/eez248435/tgm-dlm_merged/checkpoints_forward/PLAIN_ema_0.9999_140000.pt"
VOCAB_PATH_BY_DATASET[FORWARD]="/home/ee/phd/eez248435/tgm-dlm_merged/datasets/FORWARD/generate_vocab.txt"
DATA_DIR_BY_DATASET[FORWARD]="/home/ee/phd/eez248435/tgm-dlm_reduced_step/datasets/FORWARD"   # "" -> auto: <project_root>/datasets/FORWARD
ADAPTIVE_SCHEDULE_BY_DATASET[FORWARD]="/home/ee/phd/eez248435/tgm-dlm_merged/checkpoints_forward/adaptive_schedule/alpha_cumprod_step_130000.npy"

# --- RETRO (retrosynthesis: reactant prediction) ---
MODEL_PATH_BY_DATASET[RETRO]="/home/ee/phd/eez248435/tgm-dlm_reduced_step/checkpoints_retro/PLAIN_ema_0.9999_170000.pt"
VOCAB_PATH_BY_DATASET[RETRO]="/home/ee/phd/eez248435/tgm-dlm_reduced_step/datasets/RETRO/generate_vocab.txt"
DATA_DIR_BY_DATASET[RETRO]="/home/ee/phd/eez248435/tgm-dlm_reduced_step/datasets/RETRO"     # "" -> auto: <project_root>/datasets/RETRO
ADAPTIVE_SCHEDULE_BY_DATASET[RETRO]="/home/ee/phd/eez248435/tgm-dlm_reduced_step/checkpoints_retro/adaptive_schedule/alpha_cumprod_step_160000.npy"   # "" -> plain uniform schedule (fill in if RETRO has one)

# --- SMILES (unconditional / ChEBI text-to-molecule generation) ---
MODEL_PATH_BY_DATASET[SMILES]="/home/ee/phd/eez248435/tgm-dlm_reduced_step/checkpoints/PLAIN_ema_0.9999_200000.pt"
VOCAB_PATH_BY_DATASET[SMILES]="/home/ee/phd/eez248435/tgm-dlm_reduced_step/datasets/SMILES/generate_vocab.txt"
DATA_DIR_BY_DATASET[SMILES]="/home/ee/phd/eez248435/tgm-dlm_reduced_step/datasets/SMILES"   # "" -> auto: <project_root>/datasets/SMILES
ADAPTIVE_SCHEDULE_BY_DATASET[SMILES]="/home/ee/phd/eez248435/tgm-dlm_reduced_step/checkpoints/adaptive_schedule/alpha_cumprod_step_190000.npy"  # "" -> plain uniform schedule (fill in if SMILES has one)