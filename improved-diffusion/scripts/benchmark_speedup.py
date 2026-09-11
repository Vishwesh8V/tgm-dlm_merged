"""
benchmark_speedup.py

Paper-ready wall-clock benchmark: "normal" ancestral sampling (T=2000,
p_sample_loop) vs. DPM-Solver++ (dpm_solver_sample_loop) at a handful of
step budgets. Produces a table of

    method | steps | mean_sec_per_smiles | std_sec_per_smiles | n_repeats | speedup_vs_normal

with mean +/- std computed over `--n_repeats` repeated *timing-only* runs,
after a discarded warm-up run per config. Saves CSV + a LaTeX table.

SPLIT ACROSS SESSIONS (important -- normal/2000-step timing is slow):
  Every config (the normal baseline, and each dpm-steps value) writes its
  raw per-repeat times to its OWN checkpoint file under
  `<out_dir>/raw/<method>_<steps>.json`, appending after EVERY single
  repeat -- not just at the end. That means:
    - `--methods normal` runs only the 2000-step baseline this session.
    - `--methods dpm` runs only the DPM-Solver++ configs this session.
    - If a session dies mid-way through --n_repeats (killed, walltime
      limit, whatever), the NEXT invocation reads the checkpoint file,
      sees how many repeats are already banked, and only runs the
      remainder -- it does not redo completed repeats or lose them.
    - `--report_only` skips loading the model/GPU entirely and just
      rebuilds the CSV/LaTeX/Markdown table from whatever checkpoint
      files already exist on disk (normal baseline + any dpm configs
      you've collected so far, even across several separate sessions).

Design notes (read before trusting the numbers):
  - Timing is per-SMILES-string: batch_size is fixed to 1 so "sec_per_smiles"
    is a direct wall-clock measurement, not a batch average.
  - Every config times the SAME real (desc_state, desc_mask) conditioning
    example and the SAME noise seed (offset per repeat index so repeats
    aren't literally identical calls). This isolates pure compute-time
    variance (scheduler jitter, clock noise) rather than mixing in
    sample-to-sample variance from different inputs -- that is a
    separate, legitimate thing to report but is NOT what this script
    measures.
  - One warm-up call is run and discarded at the START OF EVERY PROCESS
    (not just the first time a config is ever run) -- a new session is a
    new process, so CUDA context / cuDNN autotune / kernel JIT has to be
    paid again regardless of how many repeats are already checkpointed.
  - torch.cuda.synchronize() brackets every timed call; without this,
    CUDA kernels are launched asynchronously and time.perf_counter()
    would measure almost nothing.
  - progress=False everywhere during timing -- tqdm bars measurably
    slow down tight loops and would bias the comparison.

--adaptive_schedule_path is REQUIRED alongside --model_path for any
checkpoint trained with a per-position (token-wise) adaptive noising
schedule -- which, per this repo's convention, is every checkpoint. It's
loaded into whichever diffusion object is actually sampling (the
ancestral SpacedDiffusion for 'normal', the plain GaussianDiffusion for
DPM-Solver++) via load_adaptive_schedule(), exactly once, outside the
timed loop -- never mix a checkpoint trained with an adaptive schedule
with a diffusion object that didn't load one, the reverse process will
silently diverge from training.

Usage:
    # Session 1 (just the slow baseline, resumable):
    python benchmark_speedup.py --model_path <ckpt> \
        --adaptive_schedule_path <schedule.npz> --methods normal \
        --normal_steps 2000 --n_repeats 5 --out_dir ./benchmark_outputs

    # Session 2, later, maybe on a different machine (fast DPM configs):
    python benchmark_speedup.py --model_path <ckpt> \
        --adaptive_schedule_path <schedule.npz> --methods dpm \
        --dpm_steps 2 10 100 --n_repeats 5 --out_dir ./benchmark_outputs

    # Any time after, no GPU needed, just assembles what's on disk so far:
    python benchmark_speedup.py --report_only --out_dir ./benchmark_outputs
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
import time
import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th


def _fallback_set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Checkpointed per-config timing: read-existing / run-remainder / write-after
# -every-repeat. This is the piece that makes the benchmark resumable and
# splittable across sessions.
# ---------------------------------------------------------------------------

def _raw_path(raw_dir, name, steps, tag=None):
    suffix = f"_{tag}" if tag else ""
    return Path(raw_dir) / f"{name}_{steps}{suffix}.json"


def load_checkpoint(raw_dir, name, steps, tag=None):
    path = _raw_path(raw_dir, name, steps, tag)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    state = {"method": name, "steps": steps, "raw_times": []}
    if tag is not None:
        state["tag"] = tag
    return state


def save_checkpoint(raw_dir, name, steps, state, tag=None):
    path = _raw_path(raw_dir, name, steps, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def run_config_resumable(name, steps, timed_fn, n_repeats, base_seed, raw_dir, tag=None, extra_fields=None):
    """Run up to `n_repeats` timed repeats for one (method, steps[, tag])
    config, picking up from whatever is already checkpointed on disk and
    writing the checkpoint file after EVERY repeat (not just at the end),
    so a session that dies mid-way loses at most the one in-flight repeat.

    `tag` namespaces the checkpoint filename (e.g. "ex7" for a
    content-invariance run against example_idx=7) without touching the
    original untagged filenames the primary normal/dpm sweep already
    uses -- existing checkpoints from earlier sessions keep resolving
    exactly as before. `extra_fields` (e.g. {"example_idx": 7}) is merged
    into the saved state so later reporting knows which input a checkpoint
    belongs to."""
    label = f"{name} steps={steps}" + (f" tag={tag}" if tag else "")
    state = load_checkpoint(raw_dir, name, steps, tag)
    if extra_fields:
        for k, v in extra_fields.items():
            state.setdefault(k, v)
    times = state["raw_times"]
    already = len(times)
    remaining = n_repeats - already

    if remaining <= 0:
        print(f"[{label}] {already}/{n_repeats} repeats already banked -- skipping.")
        return times

    print(f"[{label}] {already}/{n_repeats} banked, running {remaining} more this session...")
    # Warm-up once per PROCESS (a resumed session is a fresh process, so
    # CUDA/cuDNN cold-start cost has to be paid again regardless of how
    # many repeats are already on disk). Discarded, not checkpointed.
    timed_fn(seed=base_seed)

    for r in range(remaining):
        idx = already + r
        seed = base_seed + idx
        dt = timed_fn(seed=seed)
        times.append(dt)
        print(f"    [{label}] repeat {idx + 1}/{n_repeats}: {dt:.4f}s")
        state["raw_times"] = times
        save_checkpoint(raw_dir, name, steps, state, tag)  # checkpoint after EVERY repeat

    return times


def summarize(name, steps, times, n_repeats_target):
    times = np.array(times, dtype=np.float64)
    n = len(times)
    return {
        "method": name,
        "steps": steps,
        "mean_sec_per_smiles": times.mean() if n > 0 else np.nan,
        "std_sec_per_smiles": times.std(ddof=1) if n > 1 else 0.0,
        "n_repeats": n,
        "n_repeats_target": n_repeats_target,
        "complete": n >= n_repeats_target,
    }


# ---------------------------------------------------------------------------
# Model / diffusion / data setup
# ---------------------------------------------------------------------------

def build_model_and_tokenizer(args):
    from improved_diffusion import dist_util
    from improved_diffusion.transformer_model2 import TransformerNetModel2
    from mytokenizers import regexTokenizer

    vocab_path = args.vocab_path or '../../datasets/SMILES/generate_vocab.txt'
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
        learned_mean_embed=args.learned_mean_embed,
    )
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    model.eval()
    return model, tokenizer


def _use_adaptive_noise(args):
    return bool(args.adaptive_schedule_path and os.path.exists(args.adaptive_schedule_path))


def build_diffusion(args, tokenizer):
    """Plain, un-respaced GaussianDiffusion, used for DPM-Solver++.

    Every inference run in this repo pairs --model_path with
    --adaptive_schedule_path: the checkpoint was trained against a
    per-position (token-wise) noising schedule, not the plain 1-D beta
    schedule, so the diffusion object doing the sampling has to be built
    with adaptive_noising=True + the matching token_max_length/pad_tok_id
    *and* have load_adaptive_schedule() called on it -- otherwise the
    reverse process silently diverges from what the model was trained on
    (see reduced_step_profile.py / text_sample.py, same pattern).
    load_adaptive_schedule() only overwrites an existing (T, S) array in
    place, so adaptive_noising=True has to be passed at construction time,
    not bolted on after."""
    from improved_diffusion import gaussian_diffusion as gd
    from improved_diffusion import logger

    use_adaptive = _use_adaptive_noise(args)
    diffusion = gd.GaussianDiffusion(
        betas=gd.get_named_beta_schedule('sqrt', 2000),
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=gd.ModelVarType.FIXED_LARGE,
        loss_type=gd.LossType.E2E_MSE,
        rescale_timesteps=True,
        model_arch='transformer',
        training_mode='e2e',
        reg_rate=args.reg_rate,
        denoise=args.denoise,
        denoise_rate=args.denoise_rate,
        adaptive_noising=use_adaptive,
        token_max_length=tokenizer.max_len if use_adaptive else None,
        pad_tok_id=tokenizer.toktoid['[PAD]'] if use_adaptive else None,
    )
    if use_adaptive:
        logger.log(f"### Loading adaptive schedule from {args.adaptive_schedule_path} (DPM-Solver++ diffusion)...")
        diffusion.load_adaptive_schedule(args.adaptive_schedule_path)
    elif args.adaptive_schedule_path:
        logger.log(f"### WARNING: --adaptive_schedule_path={args.adaptive_schedule_path} does not exist, "
                    "falling back to the plain (non-adaptive) noise schedule.")
    return diffusion


def build_normal_diffusion(args, tokenizer):
    """SpacedDiffusion for the ancestral 'normal' sampler, built ONCE
    (outside the timing loop, and outside run_config_resumable's per-repeat
    call) so that loading the adaptive schedule from disk -- a real file
    read -- never happens inside a timed repeat and never gets re-read on
    every one of --n_repeats calls. Same adaptive-noising requirement as
    build_diffusion() above; kept as a separate function because
    SpacedDiffusion needs the respacing/use_timesteps args at construction
    time that GaussianDiffusion doesn't."""
    from improved_diffusion import gaussian_diffusion as gd
    from improved_diffusion import logger
    from improved_diffusion.respace import SpacedDiffusion

    # Mirrors text_sample.py's own stride construction rather than
    # space_timesteps() with a string -- see time_normal's docstring for
    # why: this fork's space_timesteps() string-parsing branch is
    # commented out.
    stride = max(1, 2000 // args.normal_steps)
    use_timesteps = list(range(0, 2000, stride))

    use_adaptive = _use_adaptive_noise(args)
    spaced = SpacedDiffusion(
        use_timesteps=use_timesteps,
        betas=gd.get_named_beta_schedule('sqrt', 2000),
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=gd.ModelVarType.FIXED_LARGE,
        loss_type=gd.LossType.E2E_MSE,
        rescale_timesteps=True,
        model_arch='transformer',
        training_mode='e2e',
        reg_rate=args.reg_rate,
        denoise=args.denoise,
        denoise_rate=args.denoise_rate,
        adaptive_noising=use_adaptive,
        token_max_length=tokenizer.max_len if use_adaptive else None,
        pad_tok_id=tokenizer.toktoid['[PAD]'] if use_adaptive else None,
        save_dir=args.out_dir,
    )
    if use_adaptive:
        logger.log(f"### Loading adaptive schedule from {args.adaptive_schedule_path} (normal/ancestral diffusion)...")
        spaced.load_adaptive_schedule(args.adaptive_schedule_path)
    elif args.adaptive_schedule_path:
        logger.log(f"### WARNING: --adaptive_schedule_path={args.adaptive_schedule_path} does not exist, "
                    "falling back to the plain (non-adaptive) noise schedule.")
    return spaced


def build_dataset(args, tokenizer):
    """Build the ChEBIdataset exactly ONCE. This reads the whole split
    file and loads the precomputed desc_states tensor from disk -- doing
    that per example_idx (the old get_one_example behavior) is fine for a
    handful of molecules but becomes real, needless wall-clock overhead
    once you're indexing into hundreds/thousands of examples for a
    content-invariance sweep over the full test set."""
    from mydatasets import ChEBIdataset

    dataset_dir = args.data_dir if args.data_dir and os.path.exists(args.data_dir) else (
        '/home/ee/phd/eez248435/tgm-dlm/datasets/SMILES/'
        if os.path.exists('/home/ee/phd/eez248435/tgm-dlm/datasets/SMILES/')
        else '../../datasets/SMILES/'
    )
    return ChEBIdataset(dir=dataset_dir, smi_tokenizer=tokenizer, split=args.split, replace_desc=False)


def example_from_dataset(ds, idx):
    from improved_diffusion import dist_util
    item = ds[idx]
    return item['desc_state'].to(dist_util.dev()), item['desc_mask'].to(dist_util.dev())


def get_one_example(args, tokenizer, idx=None):
    """Convenience wrapper for a single one-off lookup (used for the
    primary example). Prefer build_dataset() + example_from_dataset() when
    pulling many examples -- see their docstrings."""
    if idx is None:
        idx = args.example_idx
    ds = build_dataset(args, tokenizer)
    return example_from_dataset(ds, idx)


@th.no_grad()
def time_normal(spaced, model, shape, desc, seed, args, set_seed):
    """Ancestral p_sample_loop on a pre-built SpacedDiffusion (see
    build_normal_diffusion) -- construction, including the adaptive
    schedule's disk read, happens once outside this function so it never
    contaminates the timed region or gets repeated on every one of
    --n_repeats calls."""
    set_seed(seed)
    if th.cuda.is_available():
        th.cuda.synchronize()
    t0 = time.perf_counter()
    spaced.p_sample_loop(
        model, shape, clip_denoised=args.clip_denoised, model_kwargs={},
        top_p=args.top_p, progress=False, desc=desc,
    )
    if th.cuda.is_available():
        th.cuda.synchronize()
    return time.perf_counter() - t0


@th.no_grad()
def time_dpm(diffusion, model, shape, desc, steps, order, method, seed, args, set_seed):
    set_seed(seed)
    if th.cuda.is_available():
        th.cuda.synchronize()
    t0 = time.perf_counter()
    diffusion.dpm_solver_sample_loop(
        model, shape, clip_denoised=args.clip_denoised, model_kwargs={},
        progress=False, desc=desc, steps=steps, order=order, method=method,
    )
    if th.cuda.is_available():
        th.cuda.synchronize()
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Report assembly -- runs against whatever's on disk, no GPU required.
# ---------------------------------------------------------------------------

def _to_markdown_table(df):
    """Minimal GitHub-flavored-markdown table writer, so the report doesn't
    depend on the optional `tabulate` package (pandas.to_markdown() raises
    ImportError without it -- and this runs AFTER the expensive GPU timing
    is already done and checkpointed, so it shouldn't be able to crash the
    whole session over a cosmetic export step)."""
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _gather_config_rows(raw_dir):
    """Scan a directory of checkpoint JSON files and return one summary
    row per file that has at least one completed repeat."""
    checkpoint_files = sorted(glob.glob(str(Path(raw_dir) / "*.json")))
    rows = []
    for fp in checkpoint_files:
        with open(fp) as f:
            state = json.load(f)
        times = state.get("raw_times", [])
        if len(times) == 0:
            continue
        rows.append({
            "method": state["method"],
            "steps": state["steps"],
            "example_idx": state.get("example_idx"),
            "mean_sec_per_smiles": float(np.mean(times)),
            "std_sec_per_smiles": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
            "n_repeats": len(times),
        })
    return rows


def assemble_report(raw_dir, out_dir):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _gather_config_rows(raw_dir)
    if not rows:
        checkpoint_files = list(glob.glob(str(raw_dir / "*.json")))
        if not checkpoint_files:
            print(f"### No checkpoint files found under {raw_dir} -- nothing to report yet.")
        else:
            print(f"### Checkpoint files exist under {raw_dir} but none have completed repeats yet.")
        return None

    df = pd.DataFrame(rows).sort_values(["method", "steps"]).reset_index(drop=True)
    df = df.drop(columns=["example_idx"])  # always None for the primary (untagged) sweep

    normal_rows = df[df["method"] == "normal"]
    if len(normal_rows) == 0:
        print("### WARNING: no completed 'normal' baseline checkpoint found yet -- "
              "speedup_vs_normal will be left blank until you run --methods normal.")
        df["speedup_vs_normal"] = np.nan
    else:
        normal_mean = normal_rows["mean_sec_per_smiles"].iloc[0]
        df["speedup_vs_normal"] = normal_mean / df["mean_sec_per_smiles"]

    print("\n### Analyzing results...")
    print(df.to_string(index=False))

    csv_path = out_dir / "speedup_results.csv"
    try:
        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"### WARNING: failed to write {csv_path}: {e}")

    # Paper-ready formatted table: "mean ± std" as a single string column.
    paper_df = df.copy()
    paper_df["time_per_smiles_s"] = [
        f"{m:.3f} $\\pm$ {s:.3f}" for m, s in zip(df["mean_sec_per_smiles"], df["std_sec_per_smiles"])
    ]
    paper_df["speedup"] = [
        f"{x:.1f}$\\times$" if pd.notna(x) else "--" for x in df["speedup_vs_normal"]
    ]
    paper_df = paper_df[["method", "steps", "time_per_smiles_s", "speedup", "n_repeats"]]

    # Every export below is independently wrapped: none of these can crash
    # the session after the (expensive, GPU-bound) timing work above is
    # already checkpointed. A failure in one format (e.g. a missing
    # optional dependency) should not prevent the others from being saved.
    latex_path = out_dir / "speedup_table.tex"
    try:
        with open(latex_path, "w") as f:
            f.write(paper_df.to_latex(
                index=False, escape=False,
                caption="Wall-clock time per SMILES string and speedup vs. 2000-step ancestral sampling.",
                label="tab:dpm_speedup",
            ))
    except Exception as e:
        print(f"### WARNING: failed to write {latex_path}: {e}")

    md_path = out_dir / "speedup_table.md"
    try:
        with open(md_path, "w") as f:
            f.write(_to_markdown_table(paper_df))
    except Exception as e:
        print(f"### WARNING: failed to write {md_path}: {e}")

    print(f"\n### Saved: {csv_path}")
    print(f"### Saved: {latex_path}")
    print(f"### Saved: {md_path}")
    print("\n### Paper-ready table:")
    print(paper_df.to_string(index=False))
    return df


def assemble_invariance_report(invariance_raw_dir, raw_dir, out_dir, primary_example_idx):
    """Combine the primary (single-example) sweep with the extra
    per-example_idx checkpoints under invariance_raw_dir, and for every
    (method, steps) tested against 2+ distinct examples, report how much
    the timing actually moves across different input molecules --
    separate from (and not mixed into) the main speedup table, since that
    one is deliberately anchored to one fixed example."""
    invariance_raw_dir = Path(invariance_raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    primary_rows = _gather_config_rows(raw_dir)
    for r in primary_rows:
        r["example_idx"] = primary_example_idx

    invariance_rows = _gather_config_rows(invariance_raw_dir) if invariance_raw_dir.exists() else []

    all_rows = primary_rows + invariance_rows
    if not all_rows:
        print(f"### No content-invariance checkpoints found under {invariance_raw_dir} yet.")
        return None

    per_example_df = pd.DataFrame(all_rows).sort_values(
        ["method", "steps", "example_idx"]
    ).reset_index(drop=True)

    print("\n### Content-invariance check (same config, different molecules)...")
    print(per_example_df.to_string(index=False))

    summary_rows = []
    for (method, steps), grp in per_example_df.groupby(["method", "steps"]):
        n_examples = grp["example_idx"].nunique()
        cross_example_mean = grp["mean_sec_per_smiles"].mean()
        cross_example_std = grp["mean_sec_per_smiles"].std(ddof=1) if n_examples > 1 else 0.0
        rel_pct = (cross_example_std / cross_example_mean * 100) if cross_example_mean else float("nan")
        summary_rows.append({
            "method": method,
            "steps": steps,
            "n_examples_tested": n_examples,
            "mean_sec_per_smiles_across_examples": cross_example_mean,
            "std_sec_per_smiles_across_examples": cross_example_std,
            "relative_std_pct": rel_pct,
        })
    summary_df = pd.DataFrame(summary_rows).sort_values(["method", "steps"]).reset_index(drop=True)

    print("\n### Cross-example summary (low relative_std_pct => timing is content-invariant):")
    print(summary_df.to_string(index=False))

    per_example_path = out_dir / "content_invariance_per_example.csv"
    summary_path = out_dir / "content_invariance_summary.csv"
    md_path = out_dir / "content_invariance_summary.md"
    try:
        per_example_df.to_csv(per_example_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        with open(md_path, "w") as f:
            f.write(_to_markdown_table(summary_df))
        print(f"\n### Saved: {per_example_path}")
        print(f"### Saved: {summary_path}")
        print(f"### Saved: {md_path}")
    except Exception as e:
        print(f"### WARNING: failed to write one of the content-invariance export files: {e}")

    return summary_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = create_argparser().parse_args()
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    invariance_raw_dir = out_dir / "invariance_raw"

    if args.report_only:
        assemble_report(raw_dir, out_dir)
        if invariance_raw_dir.exists() and any(invariance_raw_dir.glob("*.json")):
            assemble_invariance_report(invariance_raw_dir, raw_dir, out_dir, args.example_idx)
        return

    from improved_diffusion import logger
    try:
        from transformers import set_seed
    except ImportError:
        set_seed = _fallback_set_seed

    logger.configure()

    print("### Loading model + tokenizer...")
    model, tokenizer = build_model_and_tokenizer(args)

    # Built once, reused for the primary example AND every invariance
    # example_idx below -- see build_dataset()'s docstring for why this
    # matters once you're indexing into hundreds/thousands of examples.
    dataset = build_dataset(args, tokenizer)
    if len(args.invariance_example_idxs) == 1 and args.invariance_example_idxs[0] == -1:
        # Sentinel: "-1" means "every example in this split", so you don't
        # have to type or shell out 1000 explicit indices.
        args.invariance_example_idxs = list(range(len(dataset)))
        print(f"### --invariance_example_idxs -1 given -> expanded to all "
              f"{len(dataset)} examples in split='{args.split}'.")

    desc_state, desc_mask = example_from_dataset(dataset, args.example_idx)
    shape = (1, tokenizer.max_len, model.in_channels)

    spaced = None
    diffusion = None

    if args.methods in ("normal", "all"):
        spaced = build_normal_diffusion(args, tokenizer)
        run_config_resumable(
            "normal", args.normal_steps,
            lambda seed: time_normal(spaced, model, shape, (desc_state, desc_mask),
                                      seed, args, set_seed),
            args.n_repeats, args.seed, raw_dir,
        )

    if args.methods in ("dpm", "all"):
        diffusion = build_diffusion(args, tokenizer)
        for steps in args.dpm_steps:
            run_config_resumable(
                "dpm", steps,
                lambda seed, steps=steps: time_dpm(diffusion, model, shape, (desc_state, desc_mask),
                                                    steps, args.dpm_solver_order, args.dpm_solver_method,
                                                    seed, args, set_seed),
                args.n_repeats, args.seed, raw_dir,
            )

    # Always assemble whatever's on disk at the end of a session -- if the
    # normal baseline hasn't been run yet, this still saves the dpm-only
    # numbers (with speedup left blank) so nothing is wasted.
    assemble_report(raw_dir, out_dir)

    # ------------------------------------------------------------------
    # Optional content-invariance check: same configs, different molecules.
    # Reuses the already-built `spaced`/`diffusion` objects (adaptive
    # schedule, model config -- none of that depends on which example is
    # being conditioned on) and the already-loaded model; only the
    # (desc_state, desc_mask) pair changes per example_idx. Each
    # (method, steps, example_idx) combination gets its own resumable
    # checkpoint file under out_dir/invariance_raw/, so this survives a
    # lost GPU session exactly like the primary sweep does.
    # ------------------------------------------------------------------
    extra_idxs = [i for i in args.invariance_example_idxs if i != args.example_idx]
    if extra_idxs:
        print(f"\n### Content-invariance check across example_idx={extra_idxs} "
              f"(primary example_idx={args.example_idx} already covered above)...")
        for idx in extra_idxs:
            ex_desc_state, ex_desc_mask = example_from_dataset(dataset, idx)
            tag = f"ex{idx}"

            if args.methods in ("normal", "all"):
                if spaced is None:
                    spaced = build_normal_diffusion(args, tokenizer)
                run_config_resumable(
                    "normal", args.normal_steps,
                    lambda seed: time_normal(spaced, model, shape, (ex_desc_state, ex_desc_mask),
                                              seed, args, set_seed),
                    args.invariance_repeats, args.seed, invariance_raw_dir,
                    tag=tag, extra_fields={"example_idx": idx},
                )

            if args.methods in ("dpm", "all"):
                if diffusion is None:
                    diffusion = build_diffusion(args, tokenizer)
                for steps in args.dpm_steps:
                    run_config_resumable(
                        "dpm", steps,
                        lambda seed, steps=steps: time_dpm(diffusion, model, shape,
                                                            (ex_desc_state, ex_desc_mask),
                                                            steps, args.dpm_solver_order,
                                                            args.dpm_solver_method, seed, args, set_seed),
                        args.invariance_repeats, args.seed, invariance_raw_dir,
                        tag=tag, extra_fields={"example_idx": idx},
                    )

        assemble_invariance_report(invariance_raw_dir, raw_dir, out_dir, args.example_idx)


def create_argparser():
    from improved_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser

    defaults = dict(
        model_path="",
        data_dir='',
        split='train_val_256',
        example_idx=0,
        token_max_length=256,
        clip_denoised=False,
        top_p=1.0,
        seed=121,
        n_repeats=5,
        normal_steps=2000,
        dpm_steps=[2, 10, 100],
        dpm_solver_order=2,
        dpm_solver_method='multistep',
        learned_mean_embed=False,
        denoise=False,
        denoise_rate=0.2,
        reg_rate=0.0,
        vocab_path='',
        out_dir='./benchmark_outputs',
        adaptive_schedule_path='',
    )
    # dpm_steps is a list -- add_dict_to_argparser (shared with the rest of
    # the repo's scripts) infers argparse `type` straight from the default
    # value's type and has no list/nargs support, so it must be added by
    # hand with nargs='+' rather than routed through that helper.
    dpm_steps_default = defaults.pop("dpm_steps")
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--dpm_steps", type=int, nargs="+", default=dpm_steps_default)
    parser.add_argument("--invariance_example_idxs", type=int, nargs="+", default=[],
                         help="Extra dataset example indices (besides --example_idx) to run "
                              "the SAME configs against, to check that timing doesn't depend "
                              "on which molecule is being generated. Empty (default) disables "
                              "this check entirely. Results go to a separate "
                              "content_invariance_*.csv, not into the main speedup table.")
    parser.add_argument("--invariance_repeats", type=int, default=1,
                         help="Repeats per extra example_idx in the content-invariance check "
                              "(kept low by default since this is a secondary sanity check, "
                              "not the primary timing table -- raise it if you want tighter "
                              "error bars on the cross-example comparison too).")
    parser.add_argument("--methods", choices=["normal", "dpm", "all"], default="all",
                         help="Which config group to run THIS session. Run 'normal' and "
                              "'dpm' in separate sessions if the 2000-step baseline is too "
                              "slow to fit alongside the DPM configs in one sitting.")
    parser.add_argument("--report_only", action="store_true",
                         help="Skip loading the model/GPU entirely; just rebuild the "
                              "CSV/LaTeX/Markdown table from whatever checkpoint files "
                              "already exist under <out_dir>/raw/.")
    return parser


if __name__ == "__main__":
    main()