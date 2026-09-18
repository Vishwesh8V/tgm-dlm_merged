"""
intermediate_sample.py
=======================

Like text_sample.py, but instead of only saving the final decoded SMILES,
this dumps the decoded SMILES at a handful of evenly-spaced points across
the reverse diffusion trajectory.

Works with all three samplers already in this repo:
  --sampler ancestral        full/uniform-stride reverse diffusion (p_sample_loop)
  --sampler dpm_solver        DPM-Solver++ fast ODE sampler (few steps)
  --sampler token_adaptive    per-position reduced-step schedule (step_matrix)

At every point we decode `pred_xstart` (the model's current best guess of the
clean embedding), not the raw noisy latent `x_t` -- decoding x_t through the
LM head is meaningless noise until very late in the trajectory, whereas
pred_xstart is a legitimate "preview" at any point.

--- How milestones are chosen (--num_milestones) ---

Rather than asking for specific raw diffusion timesteps (2000, 1000, 500, 1),
which for a fast sampler with only e.g. 10 model calls almost never land
exactly where you asked (and worse, can silently mislabel/collapse onto the
wrong capture when they don't), you instead just say how many checkpoints
you want, `--num_milestones N`, and this picks N+1 evenly-spaced ACTUAL
model calls out of however many the sampler really makes (N spaced calls,
plus the true final output):

    total_calls = however many model calls this sampler actually makes
                  (== diffusion.num_timesteps for ancestral, == dpm_solver_steps
                  for dpm_solver, == token_adaptive_steps (K) for token_adaptive)
    spacing = total_calls // N                      (floor division)
    labels  = [total_calls, total_calls - spacing, total_calls - 2*spacing, ...,
               (N of these)] + [0]

e.g. total_calls=100, N=4  -> spacing=25 -> labels = [100, 75, 50, 25, 0]
     total_calls=10,  N=4  -> spacing=2  -> labels = [10, 8, 6, 4, 0]
     total_calls=3,   N=4  -> can't subdivide 3 calls into 4 intervals -> every
                               call is used instead: labels = [3, 2, 1, 0]

--- Or pick the exact calls yourself (--milestone_steps) ---

If you'd rather name the exact calls-remaining labels instead of an evenly
spaced count, pass --milestone_steps "7,6,2" (comma-separated, any order).
This takes priority over --num_milestones whenever it's non-empty. Label 0
(final output) is NOT added automatically here -- add it yourself if you
want it, e.g. "7,6,2,0". A label outside [0, total_calls] for however many
calls this run actually made gets clamped to the nearest valid one, with a
warning printed -- it's never silently dropped.

Each "label" is a 1-indexed countdown of calls-remaining, NOT a raw diffusion
timestep -- label 0 is always the true final decoded output; label
`total_calls` is always the very first model call the sampler makes (the
noisiest point). This is exact and rounding-free regardless of sampler or
skip_type, because it indexes by call *count*, never by continuous time.
For your own bearings, the script still prints the actual raw diffusion
timestep each captured call happened to land on -- informational only, it
plays no part in selecting which calls get saved.

Usage (forward task example, mirrors forward_sample.sh's flags):

    python intermediate_sample.py \
        --model_path checkpoints_forward/PLAIN_ema_0.9999_140000.pt \
        --data_dir ../../datasets/FORWARD \
        --vocab_path ../../datasets/FORWARD/generate_vocab.txt \
        --split test --num_samples 8 --seed 108 \
        --learned_mean_embed True --denoise True --denoise_rate 0.2 --reg_rate 0.1 \
        --sampler ancestral --num_milestones 4 \
        --outputdir ../../generation_outputs/forward_process/intermediate

    # fast DPM-Solver++ inference instead:
    python intermediate_sample.py ... \
        --sampler dpm_solver --dpm_solver_steps 10 --dpm_solver_order 2 \
        --num_milestones 4 \
        --outputdir ../../generation_outputs/forward_process/intermediate_dpm10

    # token-adaptive (per-position) schedule:
    python intermediate_sample.py ... \
        --sampler token_adaptive --step_matrix_path <schedule_all_positions.csv or .npy> \
        --token_adaptive_steps 10 --dpm_solver_order 2 \
        --num_milestones 4 \
        --outputdir ../../generation_outputs/forward_process/intermediate_tok10

Output: one file per milestone label, `<outputdir>_m<LABEL>_t<ACTUAL_RAW_T>.txt`,
in the same "smiles   ||   reference" line format text_sample.py uses.
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
from improved_diffusion.respace import SpacedDiffusion
from improved_diffusion import dist_util, logger
from improved_diffusion.transformer_model2 import TransformerNetModel2
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    add_dict_to_argparser,
)
from mydatasets import ChEBIdataset
from mytokenizers import regexTokenizer


# --------------------------------------------------------------------------
# Length-range filtering (--length_range)
# --------------------------------------------------------------------------
# You can't pick meaningful fixed token-count thresholds ("short = under 20
# tokens") without knowing this dataset's actual distribution -- FORWARD,
# RETRO, and SMILES/generation each have very different typical molecule
# sizes (reaction products vs. single reactants vs. arbitrary ChEBI
# molecules), and hand-picked numbers tuned for one will silently be wrong
# for another (e.g. "long" for RETRO's short reactant fragments might not
# even be reachable, while it's mid-pack for FORWARD's larger products).
#
# So instead of guessing static ranges, this measures the REAL tokenized
# length of the target SMILES for every example in the split you're
# sampling from, right now, and buckets by quantile (20/40/60/80th
# percentile cutoffs -> 5 equal-population buckets: short, medium, long,
# very_long, longest). This self-calibrates to whatever dataset/split you
# point it at, and the buckets always have (roughly) equal numbers of
# candidates in them, whatever the underlying length distribution looks
# like (uniform, long-tailed, bimodal, ...). The computed cutoffs and each
# bucket's [min, max] token count and population are always printed, so
# it's fully auditable -- nothing is hidden or hardcoded.
LENGTH_CATEGORIES = ["short", "medium", "long", "very_long", "longest"]


def token_length(tokenizer, smiles):
    """
    True tokenized length of a SMILES string under this run's tokenizer:
    number of regex-matched SMILES tokens (atoms/bonds/ring-closures/etc.)
    plus the [SOS]/[EOS] special tokens, capped at tokenizer.max_len (which
    is where encode_one() would truncate it during real sampling). This
    does NOT count padding -- it's the real information content length.
    """
    n = len(tokenizer.rg.findall(smiles)) + 2  # + [SOS], [EOS]
    return min(n, tokenizer.max_len)


def _length_cache_path(cache_dir, data_dir, split, vocab_path, max_len):
    import hashlib
    key = f"{os.path.abspath(data_dir)}|{split}|{os.path.abspath(vocab_path)}|{max_len}"
    h = hashlib.md5(key.encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"lengths_{split}_{h}.npy")


def compute_length_buckets(train_dataset, tokenizer, cache_dir=None, vocab_path='', use_cache=True):
    """
    Scan every example's raw SMILES in this split (cheap -- just regex over
    strings, no tensors touched) and assign each to one of LENGTH_CATEGORIES
    by quantile. Returns (lengths, bucket_of_index, cutoffs).

    The scan itself is fast (a 'test' split is ~1000 lines -- sub-second),
    but a 'train' split can be 100k+ lines, and re-scanning on every single
    invocation is wasted work if you're launching this repeatedly (e.g. a
    sweep script). If `cache_dir` is given, the computed lengths are cached
    to disk keyed on (data_dir, split, vocab_path, max_len) and reused on
    later runs, invalidated automatically if the dataset size changes
    (e.g. you regenerated the split file).
    """
    n = len(train_dataset)
    cache_path = _length_cache_path(cache_dir, train_dataset.dir, train_dataset.split,
                                     vocab_path, tokenizer.max_len) if (cache_dir and use_cache) else None

    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached.shape[0] == n:
            print(f"[LengthRange] using cached token lengths from {cache_path} "
                  f"(skipping rescan of {n} examples)")
            lengths = cached
            cutoffs = np.percentile(lengths, [20, 40, 60, 80])
            bucket_of_index = np.searchsorted(cutoffs, lengths, side='right')
            return lengths, bucket_of_index, cutoffs
        print(f"[LengthRange] cache at {cache_path} has {cached.shape[0]} entries "
              f"but this split has {n} -- stale, recomputing.")

    lengths = np.array([
        token_length(tokenizer, train_dataset.ori_data[i][1])
        for i in range(n)
    ])

    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_path, lengths)
        print(f"[LengthRange] cached token lengths to {cache_path} for future runs")

    cutoffs = np.percentile(lengths, [20, 40, 60, 80])
    bucket_of_index = np.searchsorted(cutoffs, lengths, side='right')  # 0..4
    return lengths, bucket_of_index, cutoffs


def print_length_buckets(lengths, bucket_of_index, cutoffs):
    print(f"\n[LengthRange] token-length quantile cutoffs over this split "
          f"(20/40/60/80th pct): {cutoffs.tolist()}")
    for b, name in enumerate(LENGTH_CATEGORIES):
        mask = bucket_of_index == b
        n = int(mask.sum())
        if n == 0:
            print(f"    {name:>10}: 0 examples")
            continue
        lo, hi = int(lengths[mask].min()), int(lengths[mask].max())
        print(f"    {name:>10}: {n:6d} examples, token length range [{lo}, {hi}]")


def select_indices_by_length(train_dataset, tokenizer, length_range, num_samples,
                              cache_dir=None, vocab_path='', use_cache=True):
    """
    Return the first `num_samples` dataset indices (in original order, for
    reproducibility) whose target SMILES falls in one of the requested
    LENGTH_CATEGORIES buckets. Prints the bucket table either way so you
    can see what you're drawing from.
    """
    requested = [c.strip().lower() for c in length_range.split(',') if c.strip()]
    for c in requested:
        if c not in LENGTH_CATEGORIES:
            raise ValueError(f"--length_range: unknown category '{c}', "
                              f"must be one of {LENGTH_CATEGORIES}")

    lengths, bucket_of_index, cutoffs = compute_length_buckets(
        train_dataset, tokenizer, cache_dir=cache_dir, vocab_path=vocab_path, use_cache=use_cache)
    print_length_buckets(lengths, bucket_of_index, cutoffs)

    wanted_bucket_ids = {LENGTH_CATEGORIES.index(c) for c in requested}
    matching = [i for i in range(len(train_dataset)) if bucket_of_index[i] in wanted_bucket_ids]

    if len(matching) < num_samples:
        print(f"[LengthRange] WARNING: only {len(matching)} example(s) match "
              f"{requested} but --num_samples={num_samples} was requested -- "
              f"using all {len(matching)} available instead.")
    chosen = matching[:num_samples]
    print(f"[LengthRange] selected {len(chosen)} example(s) from categor"
          f"{'y' if len(requested) == 1 else 'ies'} {requested}")
    return chosen


# --------------------------------------------------------------------------
# Call recording + call-index-based milestone selection
# --------------------------------------------------------------------------
class CallRecorder:
    """
    Records every real model call the sampler makes, in visiting order
    (noisiest -> cleanest): each entry is (raw_diffusion_t, pred_xstart).
    Milestone selection later happens purely by COUNTING entries in this
    list -- never by matching a requested raw timestep -- so it is exact
    and rounding-free no matter how coarse the sampler's schedule is.
    """

    def __init__(self):
        self.calls = []  # list of (raw_t, x_batch), in visiting order

    def record(self, raw_t, x_batch):
        self.calls.append((int(raw_t), x_batch.detach().clone()))


def choose_call_labels(total_calls, num_milestones):
    """
    Pick `num_milestones` evenly-spaced 1-indexed "calls remaining" labels
    out of `total_calls` real model calls, always including the final
    output (label 0). Uses floor division for spacing, so the labels are
    exact integers -- no rounding onto a call that doesn't exist.

    total_calls=100, num_milestones=4 -> spacing=25 -> [100, 75, 50, 25, 0]
    total_calls=10,  num_milestones=4 -> spacing=2  -> [10, 8, 6, 4, 0]
    total_calls=3,   num_milestones=4 -> too few calls to subdivide ->
                                          every call instead -> [3, 2, 1, 0]
    """
    if num_milestones <= 0:
        return [0]
    if total_calls <= num_milestones:
        # Not enough real calls to fill the requested number of milestones
        # -- fall back to reporting every single call that actually exists.
        return list(range(total_calls, -1, -1))
    spacing = total_calls // num_milestones
    labels = [total_calls - i * spacing for i in range(num_milestones)]
    labels.append(0)
    # de-dup while preserving descending order (spacing=0 can't happen here
    # since total_calls > num_milestones >= 1, but guard anyway)
    seen, out = set(), []
    for l in labels:
        if l not in seen:
            seen.add(l)
            out.append(l)
    return out


def resolve_manual_labels(raw_labels, total_calls):
    """
    Validate and clean up a user-supplied list of "calls remaining" labels
    (e.g. --milestone_steps "7,6,2"). Same convention as choose_call_labels:
    label 0 = final output, label total_calls = the very first model call.

    Anything outside [0, total_calls] is out of range for this sampler's
    actual number of calls -- clamped to the nearest valid label (never
    silently dropped, so you always get SOME saved output for a bad
    request) with a loud warning so you know it happened.
    """
    seen, out = set(), []
    for raw in sorted(set(int(x) for x in raw_labels), reverse=True):
        clamped = max(0, min(raw, total_calls))
        if clamped != raw:
            print(f"[Milestones] WARNING: requested label {raw} is out of "
                  f"range for this run (only {total_calls} real model "
                  f"call(s) were made) -- clamped to {clamped}.")
        if clamped not in seen:
            seen.add(clamped)
            out.append(clamped)
    return out


def decode_latents_to_smiles(model, tokenizer, x):
    with th.no_grad():
        logits = model.get_logits(x.cuda())
        cands = th.topk(logits, k=1, dim=-1)
        ids = cands.indices.squeeze(-1)
    smiles = tokenizer.decode(ids)
    return [
        s.replace('[PAD]', '').replace('[EOS]', '').replace('[SOS]', '').strip()
        for s in smiles
    ]


def safe_write_lines(out_path, lines, overwrite):
    """
    Single guarded write path for every output file this script produces.
    Never silently overwrites: if out_path already exists and overwrite is
    False, the write is skipped (with a clear message) rather than
    clobbering whatever's there -- e.g. a previous sweep run with different
    --dpm_solver_order/--seed/--milestone_steps that happens to land on the
    same outputdir prefix. Pass --overwrite explicitly to allow replacing.
    """
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    if os.path.exists(out_path) and not overwrite:
        print(f"    -> SKIPPED (already exists, not overwriting): {out_path}")
        print(f"       re-run with --overwrite to replace it")
        return False
    if os.path.exists(out_path) and overwrite:
        print(f"    -> overwriting existing file: {out_path}")
    with open(out_path, 'w') as f:
        f.writelines(lines)
    print(f"    -> wrote {out_path}")
    return True


def write_milestones(outputdir, recorder, final_sample, num_milestones, model, tokenizer, answers,
                      manual_labels=None, overwrite=False):
    total_calls = len(recorder.calls)
    if manual_labels is not None:
        labels = resolve_manual_labels(manual_labels, total_calls)
        print(f"\n[Milestones] sampler made {total_calls} real model call(s); "
              f"using manually-requested labels (calls remaining, 0=final): {labels}")
    else:
        labels = choose_call_labels(total_calls, num_milestones)
        print(f"\n[Milestones] sampler made {total_calls} real model call(s); "
              f"requested {num_milestones} milestone(s) -> using labels (calls "
              f"remaining, 0=final): {labels}")

    for label in labels:
        if label == 0:
            x_batch = final_sample
            raw_t = "final"
        else:
            # label = calls remaining INCLUDING this one -> 1-indexed call
            # number is (total_calls - label + 1).
            call_idx0 = total_calls - label  # 0-indexed into recorder.calls
            raw_t, x_batch = recorder.calls[call_idx0]
        smiles = decode_latents_to_smiles(model, tokenizer, x_batch)
        if smiles:
            print(f"    label={label:>5}  (raw t={raw_t}):  sample0 = {smiles[0]}")
        out_path = f"{outputdir}_m{label}_t{raw_t}.txt"
        lines = [s + '   ||   ' + (answers[i] if i < len(answers) else "") + '\n'
                 for i, s in enumerate(smiles)]
        safe_write_lines(out_path, lines, overwrite)


# --------------------------------------------------------------------------
# Sampler 1: ancestral (full / uniform-stride reverse diffusion)
# --------------------------------------------------------------------------
def run_ancestral(diffusion, model, sample_shape, desc, args):
    recorder = CallRecorder()
    indices = list(range(diffusion.num_timesteps))[::-1]
    tmap = getattr(diffusion, 'timestep_map', None)

    gen = diffusion.p_sample_loop_progressive(
        model,
        sample_shape,
        clip_denoised=args.clip_denoised,
        denoised_fn=None,
        model_kwargs={},
        progress=True,
        top_p=args.top_p,
        desc=desc,
    )

    last_out = None
    for local_i, out in zip(indices, gen):
        raw_t = int(tmap[local_i]) if tmap is not None else int(local_i)
        recorder.record(raw_t, out["pred_xstart"])
        last_out = out

    final_sample = last_out["sample"]
    return final_sample, recorder


# --------------------------------------------------------------------------
# Sampler 1b: DDIM (deterministic reverse ODE -- same SpacedDiffusion
# machinery as ancestral, just diffusion.ddim_sample_loop_progressive
# instead of p_sample_loop_progressive)
# --------------------------------------------------------------------------
def run_ddim(diffusion, model, sample_shape, desc, args):
    recorder = CallRecorder()
    indices = list(range(diffusion.num_timesteps))[::-1]
    tmap = getattr(diffusion, 'timestep_map', None)

    gen = diffusion.ddim_sample_loop_progressive(
        model,
        sample_shape,
        clip_denoised=args.clip_denoised,
        denoised_fn=None,
        model_kwargs={},
        progress=True,
        eta=0.0,
        desc=desc,
    )

    last_out = None
    for local_i, out in zip(indices, gen):
        raw_t = int(tmap[local_i]) if tmap is not None else int(local_i)
        recorder.record(raw_t, out["pred_xstart"])
        last_out = out

    final_sample = last_out["sample"]
    return final_sample, recorder


# --------------------------------------------------------------------------
# Sampler 2: plain DPM-Solver++ (few-step ODE solver, global schedule)
# --------------------------------------------------------------------------
def run_dpm_solver(dpm_solver_diffusion, model, sample_shape, desc, args, device):
    from improved_diffusion.dpm_solver import NoiseScheduleVP, DPM_Solver, expand_dims

    img = th.randn(*sample_shape, device=device)
    if getattr(model, 'mean_embed', None) is not None:
        img = img + model.mean_embed[None, None].to(device)

    desc_state, desc_mask = desc[0].to(device), desc[1].to(device)

    noise_schedule = NoiseScheduleVP(
        schedule='discrete',
        alphas_cumprod=th.tensor(dpm_solver_diffusion.alphas_cumprod, dtype=th.float32, device=device),
    )

    diffusion = dpm_solver_diffusion
    total_N = noise_schedule.total_N

    def get_model_input_time(t_continuous):
        return th.clamp(th.round(t_continuous * total_N - 1), min=0, max=total_N - 1).long()

    recorder = CallRecorder()

    # Re-derived from improved_diffusion.dpm_solver.tgmdlm_model_wrapper, but
    # additionally records the recovered pred_xstart at every model call
    # (tgmdlm_model_wrapper only returns the eps-space value the solver
    # needs internally, so we invert its last step to get pred_xstart back).
    # Every real call is recorded, in order -- which of these actually get
    # saved to disk is decided afterwards purely by call COUNT (see
    # choose_call_labels), never by matching this raw_t to anything.
    def model_fn(x, t_continuous):
        t_idx = get_model_input_time(t_continuous).expand(x.shape[0])
        t_input = diffusion._scale_timesteps(t_idx)
        model_output = model(x, t_input, desc_state, desc_mask)
        if diffusion.model_var_type.name in ("LEARNED", "LEARNED_RANGE"):
            C = x.size(-1)
            model_output, _ = th.split(model_output, C, dim=-1)
        if diffusion.model_mean_type.name == "START_X":
            pred_xstart = model_output
        elif diffusion.model_mean_type.name == "EPSILON":
            pred_xstart = diffusion._predict_xstart_from_eps(x_t=x, t=t_idx, eps=model_output)
        else:
            raise NotImplementedError(diffusion.model_mean_type)
        if args.clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)

        raw_t = int(t_idx[0].item())
        recorder.record(raw_t, pred_xstart)

        alpha_t = expand_dims(noise_schedule.marginal_alpha(t_continuous), x.dim())
        sigma_t = expand_dims(noise_schedule.marginal_std(t_continuous), x.dim())
        eps = (x - alpha_t * pred_xstart) / sigma_t
        return eps

    correcting_xt_fn = None
    mean_embed = getattr(model, 'mean_embed', None)
    if dpm_solver_diffusion.denoise and mean_embed is not None:
        mean_embed_dev = mean_embed.to(device)

        def correcting_xt_fn(x_t, t, step):
            sigma_t = noise_schedule.marginal_std(t)
            mask_rate = (sigma_t * dpm_solver_diffusion.denoise_rate).clamp(0.0, 1.0)
            mask_rate_expand = mask_rate.view(1, 1).expand(x_t.shape[0], x_t.shape[1])
            random_mask = mask_rate_expand.bernoulli()[..., None].expand(x_t.shape)
            mean_embed_expand = mean_embed_dev[None, None].expand(x_t.shape)
            return th.where(random_mask == 0, x_t, mean_embed_expand)

    dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++",
                             correcting_xt_fn=correcting_xt_fn)

    with th.no_grad():
        sample = dpm_solver.sample(
            img,
            steps=args.dpm_solver_steps,
            order=args.dpm_solver_order,
            method=args.dpm_solver_method,
            skip_type="time_uniform",
            lower_order_final=True,
        )

    # `sample` is the ODE's true final endpoint (there's one more update
    # after the last model call that doesn't get a fresh network call --
    # see dpm_solver.py's multistep loop) -- record it as the final output,
    # NOT necessarily identical to recorder.calls[-1].
    return sample, recorder


# --------------------------------------------------------------------------
# Sampler 3: token-adaptive DPM-Solver++ (per-position step_matrix)
# --------------------------------------------------------------------------
def run_token_adaptive(token_adaptive_diffusion, model, sample_shape, desc, step_matrix, args, device):
    recorder = CallRecorder()
    call_count = {"i": 0}

    def callback(x_t, pred_xstart):
        r = call_count["i"]
        call_count["i"] += 1
        # Representative scalar timestep for this call, same convention the
        # library itself uses internally (median over positions) -- purely
        # for the printed raw_t label, plays no part in call selection.
        rep_t = int(np.median(step_matrix[r]))
        recorder.record(rep_t, pred_xstart)

    sample = token_adaptive_diffusion.token_adaptive_dpm_solver_sample_loop(
        model,
        sample_shape,
        step_matrix=step_matrix,
        clip_denoised=args.clip_denoised,
        denoised_fn=None,
        model_kwargs={},
        progress=True,
        desc=desc,
        order=args.dpm_solver_order,
        callback=callback,
    )
    return sample, recorder


def main():
    args = create_argparser().parse_args()
    set_seed(int(args.seed))
    logger.configure()

    vocab_path = args.vocab_path or '../../datasets/SMILES/generate_vocab.txt'
    tokenizer = regexTokenizer(path=vocab_path, max_len=args.token_max_length)

    manual_labels = None
    if args.milestone_steps.strip():
        manual_labels = [int(x) for x in args.milestone_steps.split(',') if x.strip() != '']

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
    device = dist_util.dev()

    use_adaptive_noise = bool(args.adaptive_schedule_path and os.path.exists(args.adaptive_schedule_path))

    # Dataset
    train_dataset = ChEBIdataset(dir=args.data_dir, smi_tokenizer=tokenizer, split=args.split, replace_desc=False)
    if args.length_range.strip():
        cache_dir = args.length_cache_dir or os.path.join(PROJECT_ROOT, "generation_outputs", ".length_cache")
        indices = select_indices_by_length(
            train_dataset, tokenizer, args.length_range, args.num_samples,
            cache_dir=cache_dir, vocab_path=vocab_path, use_cache=not args.no_length_cache)
    else:
        indices = list(range(min(args.num_samples, len(train_dataset))))
    desc_list = [(train_dataset[i]['desc_state'], train_dataset[i]['desc_mask'], train_dataset[i]['smiles'])
                 for i in indices]
    answers = [d[2] for d in desc_list]
    desc_state = th.cat([d[0] for d in desc_list], dim=0)
    desc_mask = th.cat([d[1] for d in desc_list], dim=0)
    sample_shape = (len(desc_list), tokenizer.max_len, model.in_channels)
    print(f"Sampling {len(desc_list)} example(s), shape={sample_shape}, num_milestones={args.num_milestones}")

    if args.sampler in ("ancestral", "ddim"):
        # --timestep_respacing is a STRIDE: use_timesteps = range(0, 2000, stride),
        # so stride must divide 2000 evenly to land on the step count you want
        # (e.g. stride=4 -> 500 steps, stride=1 -> full 2000-step trajectory).
        step_stride = int(args.timestep_respacing) if args.timestep_respacing.isdigit() else 1
        use_timesteps = list(range(0, 2000, step_stride))
        diffusion = SpacedDiffusion(
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
            adaptive_noising=use_adaptive_noise,
            token_max_length=tokenizer.max_len if use_adaptive_noise else None,
            pad_tok_id=tokenizer.toktoid['[PAD]'] if use_adaptive_noise else None,
        )
        if use_adaptive_noise:
            diffusion.load_adaptive_schedule(args.adaptive_schedule_path)
        desc_dev = (desc_state.to(device), desc_mask.to(device))
        if args.sampler == "ancestral":
            final_sample, recorder = run_ancestral(diffusion, model, sample_shape, desc_dev, args)
        else:
            final_sample, recorder = run_ddim(diffusion, model, sample_shape, desc_dev, args)
    else:
        base_diffusion = gd.GaussianDiffusion(
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
            adaptive_noising=use_adaptive_noise,
            token_max_length=tokenizer.max_len if use_adaptive_noise else None,
            pad_tok_id=tokenizer.toktoid['[PAD]'] if use_adaptive_noise else None,
        )
        if use_adaptive_noise:
            base_diffusion.load_adaptive_schedule(args.adaptive_schedule_path)
        desc_dev = (desc_state.to(device), desc_mask.to(device))

        if args.sampler == "dpm_solver":
            final_sample, recorder = run_dpm_solver(base_diffusion, model, sample_shape, desc_dev, args, device)
        elif args.sampler == "token_adaptive":
            from improved_diffusion.dpm_solver import load_step_matrix
            step_matrix = load_step_matrix(args.step_matrix_path, K=args.token_adaptive_steps, seq_len=tokenizer.max_len)
            final_sample, recorder = run_token_adaptive(base_diffusion, model, sample_shape, desc_dev,
                                                          step_matrix, args, device)
        else:
            raise ValueError(f"unknown --sampler {args.sampler}")

    write_milestones(args.outputdir, recorder, final_sample, args.num_milestones, model, tokenizer, answers,
                      manual_labels=manual_labels, overwrite=args.overwrite)

    final_smiles = decode_latents_to_smiles(model, tokenizer, final_sample)
    final_lines = [s + '   ||   ' + (answers[i] if i < len(answers) else "") + '\n'
                   for i, s in enumerate(final_smiles)]
    safe_write_lines(args.outputdir + "_FINAL.txt", final_lines, args.overwrite)


def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=8,
        batch_size=8,
        seed=121,
        model_path="",
        model_arch='transformer',
        out_dir="generation_outputs",
        data_dir='',
        vocab_path='',
        split='test',
        length_range='',                # '' = no filtering; else one or more of
                                         # short,medium,long,very_long,longest
        length_cache_dir='',            # '' = default to <project_root>/generation_outputs/.length_cache
        no_length_cache=False,          # force a fresh rescan even if a cache file exists
        top_p=1.0,
        token_max_length=256,
        learned_mean_embed=False,
        denoise=False,
        denoise_rate=0.2,
        reg_rate=0.0,
        adaptive_schedule_path='',
        timestep_respacing='1',         # stride for ancestral/ddim (1 = full 2000-step trajectory)
        sampler='ancestral',            # ancestral | ddim | dpm_solver | token_adaptive
        num_milestones=4,               # how many evenly-spaced checkpoints (+final) to save
        milestone_steps='',             # manual override, e.g. "7,6,2" -- "calls remaining" labels;
                                         # takes priority over num_milestones when non-empty
        dpm_solver_steps=10,
        dpm_solver_order=2,
        dpm_solver_method='multistep',
        step_matrix_path='',
        token_adaptive_steps=10,
        outputdir='../../generation_outputs/forward_process/intermediate',
        overwrite=False,                # if False (default), never clobber an existing
                                         # output file -- skip it with a message instead.
                                         # Pass True to intentionally replace old results.
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()