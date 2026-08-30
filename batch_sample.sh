#!/usr/bin/env bash
# ==============================================================================
# Shell Script: Batch Inference / Molecular Sampling for All Checkpoints
# ==============================================================================

set -e

# --- Environment Setup ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMPI_MCA_btl="^openib"

# --- Determine Directory Structure ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "${SCRIPT_DIR}/sample.sh" ]; then
    PROJECT_ROOT="${SCRIPT_DIR}"
else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

# Checkpoint Directory argument (Default: ${PROJECT_ROOT}/checkpoints)
CHECKPOINT_DIR="${1:-${PROJECT_ROOT}/checkpoints}"

# Shift first argument if provided so remaining args can be forwarded to sample.sh
if [ $# -gt 0 ] && [ -d "$1" ]; then
    shift
fi

OUT_BATCH_DIR="${PROJECT_ROOT}/generation_outputs_batch"
mkdir -p "${OUT_BATCH_DIR}"

echo "======================================================================"
echo " Starting TGM-DLM Batch Inference for All Checkpoints"
echo "======================================================================"
echo " Project Root      : ${PROJECT_ROOT}"
echo " Checkpoint Dir    : ${CHECKPOINT_DIR}"
echo " Output Batch Dir  : ${OUT_BATCH_DIR}"
echo " CUDA Visible Devs : ${CUDA_VISIBLE_DEVICES}"
echo "======================================================================"

if [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo "Error: Checkpoint directory does not exist: ${CHECKPOINT_DIR}"
    exit 1
fi

# Execute Python script to pair checkpoints with schedules and run sample.sh sequentially
python -c "
import os, sys, glob, re, subprocess

project_root = '${PROJECT_ROOT}'
ckpt_dir = '${CHECKPOINT_DIR}'
out_batch_dir = '${OUT_BATCH_DIR}'
extra_args = sys.argv[1:]

sched_dir = os.path.join(ckpt_dir, 'adaptive_schedule')

# 1. Discover all model checkpoint (.pt) files
pt_files = glob.glob(os.path.join(ckpt_dir, '*.pt'))
if not pt_files:
    print(f'Error: No .pt model checkpoint files found in {ckpt_dir}')
    sys.exit(1)

# Group .pt files by step number, preferring EMA checkpoint if available
pt_by_step = {}
for pf in pt_files:
    m = re.search(r'(\d+)\.pt$', os.path.basename(pf))
    if m:
        step = int(m.group(1))
        is_ema = 'ema' in os.path.basename(pf)
        if step not in pt_by_step or is_ema:
            pt_by_step[step] = pf

# 2. Discover all schedule (.npy) files
sched_by_step = {}
if os.path.exists(sched_dir):
    sched_files = glob.glob(os.path.join(sched_dir, '*.npy'))
    for sf in sched_files:
        m = re.search(r'alpha_cumprod_step_(\d+)\.npy$', os.path.basename(sf))
        if m:
            step = int(m.group(1))
            sched_by_step[step] = sf

steps = sorted(pt_by_step.keys())
print(f'Found {len(steps)} distinct checkpoint steps to process: {steps}')

sample_sh_path = os.path.join(project_root, 'sample.sh')

# 3. Process each checkpoint step sequentially
for idx, step in enumerate(steps, 1):
    model_path = pt_by_step[step]
    
    # Matching schedule formula:
    # Priority 1: step - 500 (e.g. 119,500.npy for step 120,000)
    # Priority 2: step.npy (e.g. 120,000.npy)
    # Priority 3: highest schedule step <= step
    target_sched_step = step - 500
    matched_sched_path = ''
    if target_sched_step in sched_by_step:
        matched_sched_path = sched_by_step[target_sched_step]
    elif step in sched_by_step:
        matched_sched_path = sched_by_step[step]
    else:
        valid_steps = [s for s in sched_by_step.keys() if s <= step]
        if valid_steps:
            matched_sched_path = sched_by_step[max(valid_steps)]
            
    # Format output filename (e.g. sampled_smiles_120k.txt)
    step_str = f'{step//1000}k' if step >= 1000 and step % 1000 == 0 else str(step)
    out_file = os.path.join(out_batch_dir, f'sampled_smiles_{step_str}.txt')
    
    print('\n' + '='*70)
    print(f'[{idx}/{len(steps)}] Processing Checkpoint Step {step}')
    print(f'  Model Weights:     {os.path.basename(model_path)}')
    print(f'  Matching Schedule: {os.path.basename(matched_sched_path) if matched_sched_path else \"None (default schedule)\"}')
    print(f'  Output File:       {out_file}')
    print('='*70 + '\n')
    
    env = os.environ.copy()
    env['MODEL_PATH'] = model_path
    env['ADAPTIVE_SCHEDULE'] = matched_sched_path
    env['OUTPUT_FILE'] = out_file
    
    cmd = ['bash', sample_sh_path] + extra_args
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        print(f'Warning: Sampling for checkpoint step {step} exited with code {res.returncode}')

print('\n' + '='*70)
print(f' Batch Inference Complete!')
print(f' All outputs saved in directory: {out_batch_dir}')
print('='*70)
" "$@"
