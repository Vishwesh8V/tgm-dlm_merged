"""
reduced_step_profile.py

Per-position, per-timestep validation-loss profiling for TGM-DLM's Phase-1
diffusion model. Stage 1 of the adaptive-step-inference pipeline.

Ported from `reduced_step.py` (a sibling DiffuSeq-derived codebase built for
a molecule -> caption task) and adapted here for TGM-DLM's caption -> SMILES
direction. The reference profiler only tracked the generated "caption" half
of a concatenated [source | target] sequence (a fixed range of positions),
because the source half was preserved input the model never denoised.
TGM-DLM has no such split: the caption is external cross-attention
conditioning (desc_state/desc_mask), and every position of the generated
SMILES sequence is a genuine model output -- so this profiler tracks every
sequence position, excluding only padding.

For each validation example we run the *full* ancestral reverse trajectory
(`GaussianDiffusion.p_sample_loop_progressive`, one model call per
t = 0..T-1) and record:
    pred_xstart_list[0]      -> predicted x0 at t = T-1
    ...
    pred_xstart_list[T-1]    -> predicted x0 at t = 0

This is what makes a per-timestep loss curve at every t possible, and is why
this profiling pass is expensive (T model calls per example) even though
normal sampling never runs this many steps -- it only needs to be run once
per checkpoint.

Each predicted x0 is compared with the ground-truth clean embedding
x_start = model.get_embeds(input_ids), giving per-position validation MSE
aligned by raw timestep t:
    mean_loss[t, i]
    logsnr[t, i]
    alpha_cumprod[t, i]

No smoothing, no derivative, no step allocation happens here -- that is
`trajectory_analysis.py`'s job (Stage 2), unchanged in spirit from the
reference.

Usage:
    python reduced_step_profile.py \
        --model_path diffusion_models/<phase1_checkpoint>.pt \
        --split validation --num_examples 200 \
        --out_path reduced_step_outputs/<run_name>/validation_profile.npz
"""

import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
IMPROVED_DIFFUSION_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
TRANSFORMERS_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, "transformers/src"))

for p in [TRANSFORMERS_SRC, IMPROVED_DIFFUSION_DIR]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import argparse
from pathlib import Path

import numpy as np
import torch as th

try:
    from transformers import set_seed
except ImportError:
    def set_seed(seed):
        import random
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(seed)

from improved_diffusion import gaussian_diffusion as gd
from improved_diffusion import dist_util, logger
from improved_diffusion.transformer_model2 import TransformerNetModel2
from improved_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
from mydatasets import ChEBIdataset
from mytokenizers import regexTokenizer


def compute_logsnr(alpha_cumprod, eps=1e-12):
    """Compute full log-SNR from alpha_bar."""
    alpha = np.asarray(alpha_cumprod, dtype=np.float64)
    alpha = np.clip(alpha, eps, 1.0 - eps)
    return np.log(alpha) - np.log1p(-alpha)


class ValidationLossProfiler:
    def __init__(self, diffusion, seq_len):
        self.T = diffusion.num_timesteps
        self.L = seq_len

        alpha = np.asarray(diffusion.alphas_cumprod, dtype=np.float64)
        if alpha.ndim == 1:
            alpha = np.repeat(alpha[:, None], seq_len, axis=1)
        if alpha.shape != (self.T, self.L):
            raise ValueError(
                f"Expected alphas_cumprod shape {(self.T, self.L)}, got {alpha.shape}"
            )

        self.alpha_cumprod = alpha
        self.logsnr = compute_logsnr(alpha)

        self.loss_sum = np.zeros((self.T, self.L), dtype=np.float64)
        self.count = np.zeros((self.T, self.L), dtype=np.float64)

    @th.no_grad()
    def add_example(self, pred_xstart_list, x_start, valid_mask):
        """
        Accumulate one validation example (batch size 1).

        pred_xstart_list:
            list length T, ordered [T-1, ..., 0], each entry [1, L, D].
        x_start:
            ground-truth clean embedding [1, L, D].
        valid_mask:
            [1, L] float/bool, 1 for positions to include (non-pad),
            0 for positions to exclude (padding).
        """
        if len(pred_xstart_list) != self.T:
            raise ValueError(
                f"Expected {self.T} predictions, got {len(pred_xstart_list)}. "
                "Run full-step decoding (p_sample_loop_progressive over the "
                "un-respaced base diffusion, not a SpacedDiffusion / DPM-Solver "
                "subsample)."
            )

        mask = valid_mask.to(x_start.device).float()

        for traj_idx, pred_x0 in enumerate(pred_xstart_list):
            # traj_idx = 0     -> t = T-1
            # traj_idx = T-1   -> t = 0
            t = self.T - 1 - traj_idx

            token_mse = (
                (pred_x0.float() - x_start.float()) ** 2
            ).mean(dim=-1)  # [1, L]
            token_mse = token_mse * mask

            self.loss_sum[t] += token_mse.sum(dim=0).double().cpu().numpy()
            self.count[t] += mask.sum(dim=0).double().cpu().numpy()

    def save(self, output_path):
        """Save aligned timestep/logSNR/loss arrays."""
        mean_loss = np.full_like(self.loss_sum, np.nan)
        np.divide(self.loss_sum, self.count, out=mean_loss, where=self.count > 0)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        steps = np.arange(self.T, dtype=np.int64)
        np.savez_compressed(
            output_path,
            steps=steps,
            mean_loss=mean_loss,
            logsnr=self.logsnr,
            alpha_cumprod=self.alpha_cumprod,
            valid_count=self.count,
        )

        print(f"### Saved validation profile: {output_path}")
        print(f"### mean_loss shape: {mean_loss.shape}")
        print(f"### logsnr shape:    {self.logsnr.shape}")
        print(f"### decoding order:  {self.T - 1} -> 0")

        return str(output_path)


def create_argparser():
    defaults = dict(
        model_path="",
        out_path="reduced_step_outputs/validation_profile.npz",
        split="train_val_256",
        num_examples=200,
        start_idx=0,
        seed=121,
        token_max_length=256,
        adaptive_schedule_path='',
        data_dir='',
        clip_denoised=False,
        # Mixed-Space Diffusion (must match how this checkpoint was trained,
        # or model.load_state_dict()/the forward process will silently diverge
        # from training -- see UNIFIED_FEATURES.md).
        learned_mean_embed=False,
        denoise=False,
        denoise_rate=0.2,
        reg_rate=0.0,
        vocab_path='',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


@th.no_grad()
def main():
    args = create_argparser().parse_args()
    set_seed(args.seed)
    logger.configure()

    logger.log("### Creating model and diffusion for validation profiling...")
    dataset_dir = args.data_dir if args.data_dir and os.path.exists(args.data_dir) else (
        '/home/ee/phd/eez248435/tgm-dlm/datasets/SMILES/'
        if os.path.exists('/home/ee/phd/eez248435/tgm-dlm/datasets/SMILES/')
        else '../../datasets/SMILES/'
    )
    # Vocab resolution is opt-in via --vocab_path only. It must NOT be
    # inferred from data_dir: a checkpoint's training data_dir and its vocab
    # file are not reliably the same path (confirmed by a regression --
    # inferring it from data_dir silently swapped in the wrong vocab for a
    # checkpoint that was working fine before). Default is regexTokenizer's
    # own original hardcoded default; only override when explicitly asked
    # to, e.g. for a checkpoint trained on a different vocab (retro).
    vocab_path = getattr(args, 'vocab_path', '') or '../../datasets/SMILES/generate_vocab.txt'
    tokenizer = regexTokenizer(path=vocab_path, max_len=args.token_max_length)
    model = TransformerNetModel2(
        in_channels=32,
        model_channels=128,
        dropout=0.1,
        use_checkpoint=False,
        config_name='bert-base-uncased',
        training_mode='e2e',
        vocab_size=len(tokenizer),
        experiment_mode='lm',
        logits_mode=1,
        hidden_size=1024,
        num_attention_heads=16,
        num_hidden_layers=12,
        learned_mean_embed=getattr(args, 'learned_mean_embed', False),
    )

    # Plain (un-respaced) GaussianDiffusion: the full T-step trajectory is
    # exactly what we need to profile, not a subsampled schedule. This
    # mirrors how `text_sample.py` builds `dpm_solver_diffusion` for the
    # same reason.
    #
    # If an adaptive-noising schedule is going to be loaded below, the
    # diffusion object must be constructed with adaptive_noising=True (and
    # token_max_length/pad_tok_id) up front so its schedule arrays are
    # already (T, S) -- load_adaptive_schedule()/update_time_discretized_
    # parameters() only overwrite an existing 2-D array in place, they do
    # not expand a 1-D one.
    use_adaptive_noise = bool(
        args.adaptive_schedule_path and os.path.exists(args.adaptive_schedule_path)
    )
    diffusion = gd.GaussianDiffusion(
        betas=gd.get_named_beta_schedule('sqrt', 2000),
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=gd.ModelVarType.FIXED_LARGE,
        loss_type=gd.LossType.E2E_MSE,
        rescale_timesteps=True,
        model_arch='transformer',
        training_mode='e2e',
        reg_rate=getattr(args, 'reg_rate', 0.0),
        denoise=getattr(args, 'denoise', False),
        denoise_rate=getattr(args, 'denoise_rate', 0.2),
        adaptive_noising=use_adaptive_noise,
        token_max_length=tokenizer.max_len if use_adaptive_noise else None,
        pad_tok_id=tokenizer.toktoid['[PAD]'] if use_adaptive_noise else None,
        save_dir=os.path.dirname(args.out_path) or './generation_outputs',
    )

    if use_adaptive_noise:
        logger.log(f"### Loading adaptive schedule from {args.adaptive_schedule_path}...")
        diffusion.load_adaptive_schedule(args.adaptive_schedule_path)

    if not args.model_path:
        raise ValueError("--model_path is required.")
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    val_dataset = ChEBIdataset(
        dir=dataset_dir, smi_tokenizer=tokenizer, split=args.split, replace_desc=False
    )

    end_idx = (
        min(args.start_idx + args.num_examples, len(val_dataset))
        if args.num_examples > 0 else len(val_dataset)
    )
    logger.log(
        f"### Profiling on {args.split}[{args.start_idx}:{end_idx}] "
        f"({end_idx - args.start_idx} examples, {diffusion.num_timesteps} "
        "model calls each -- this is slow by design, see module docstring)"
    )

    profiler = ValidationLossProfiler(diffusion, seq_len=tokenizer.max_len)
    pad_id = tokenizer.toktoid['[PAD]']

    from tqdm import tqdm
    for idx in tqdm(range(args.start_idx, end_idx)):
        item = val_dataset[idx]
        input_ids = item['tok_smiles'].to(dist_util.dev())          # [1, L]
        desc_state = item['desc_state'].to(dist_util.dev())
        desc_mask = item['desc_mask'].to(dist_util.dev())

        x_start = model.get_embeds(input_ids)                       # [1, L, D]
        valid_mask = (input_ids != pad_id).float()                  # [1, L]

        shape = (1, tokenizer.max_len, model.in_channels)
        pred_xstart_list = []
        for out in diffusion.p_sample_loop_progressive(
            model,
            shape,
            clip_denoised=args.clip_denoised,
            model_kwargs={},
            progress=False,
            desc=(desc_state, desc_mask),
        ):
            pred_xstart_list.append(out["pred_xstart"].detach())

        profiler.add_example(pred_xstart_list, x_start, valid_mask)

    profiler.save(args.out_path)


if __name__ == "__main__":
    main()