# Adaptive Per-Token Noise Scheduling in TGM-DLM

> Ported and adapted from SeqDiffuSeq into TGM-DLM for molecular (SMILES) generation.

---

## Table of Contents

1. Background and Motivation
2. Core Idea
3. Architecture Overview
4. File-by-File Changes
   - 4.1 gaussian_diffusion.py -- __init__
   - 4.2 gaussian_diffusion.py -- _expand_schedule_to_2d
   - 4.3 gaussian_diffusion.py -- update_time_discretized_parameters
   - 4.4 gaussian_diffusion.py -- _loss_history_update
   - 4.5 gaussian_diffusion.py -- _refit_schedule
   - 4.6 gaussian_diffusion.py -- _get_fixed_large_variance
   - 4.7 gaussian_diffusion.py -- p_mean_variance
   - 4.8 gaussian_diffusion.py -- training_losses_e2e
   - 4.9 gaussian_diffusion.py -- _extract_into_tensor
   - 4.10 respace.py -- SpacedDiffusion.__init__
   - 4.11 train_util.py -- forward_backward
   - 4.12 script_util.py -- create_gaussian_diffusion
5. The Algorithm End to End
6. Logging Every 20 Steps
7. Molecule-Specific Observations
8. How to Enable / Disable
9. Backward Compatibility
10. Key Invariants
11. Checkpoint Resume — Bugs Fixed
    - 11.1 Bug 1: Warmup counter resets → immediate post-resume refit
    - 11.2 Bug 2: Loss history not saved/restored across restarts
    - 11.3 Bug 3: train.sh resume block was commented out
    - 11.4 train.sh rewrite

---

## 1. Background and Motivation

### Standard diffusion noise schedule

In vanilla DDPM/DiffuSeq-style text diffusion, the noise schedule is a **1-D array**:

```
alphas_cumprod: shape (T,)
```

At timestep `t`, **every token position** in a sequence of length `S` gets the **same** noise coefficient. The forward process is:

```
x_t = sqrt(a_t) * x_0  +  sqrt(1 - a_t) * e,    e ~ N(0, I)
```

This is wasteful: if some positions are easy (common atoms `C`, `N`) and others are hard (ring-closure digits, stereocenters), the model spends **equal training budget everywhere regardless of difficulty**.

### The adaptive idea

Replace the shared 1-D schedule with a **2-D per-position schedule**:

```
alphas_cumprod: shape (T, S)
```

Now position `s` can be at a **different signal level** than position `s'` at the same timestep `t`. The schedule is **re-fit every N steps** from empirical per-token reconstruction loss:

- **Hard positions**: steeper loss curve -> more timesteps at high noise
- **Easy positions**: shallower curve -> compressed timestep budget

Re-fitting uses **non-parametric monotone interpolation** (np.interp) -- no architecture changes, no extra parameters.

### Why this matters for molecules

SMILES strings are structurally uneven:
- Common atoms (`C`, `N`, `O`) -- easy
- Ring-closure digits (`1`, `2`, `3`) -- rare and structurally critical
- Stereocenters (`@`, `@@`) -- very hard
- Charged atoms (`[NH4+]`, `[O-]`) -- hard

The adaptive schedule discovers and encodes this difficulty map automatically.

---

## 2. Core Idea

| Aspect | Before | After |
|--------|--------|-------|
| `alphas_cumprod` shape | `(T,)` | `(T, S)` per position |
| Noise in `q_sample` | Same for all positions | Different per position |
| Schedule | Fixed | Re-fit every N steps |
| Training cost | None | ~1 all_gather + numpy/step |
| Inference | Unchanged | Unchanged |
| Backward compat | -- | `adaptive_noise=False` = zero change |

Key insight: spend equal timesteps at equal **loss increments per position**, not equal noise increments. Diffusion budget becomes proportional to difficulty.

---

## 3. Architecture Overview

```
Training Step
    |
    v
train_util.py -> forward_backward()
    |  passes training_step=self.step (NEW)
    v
training_losses_e2e(training_step=N)
    |
    +-- [Normal loss -- unchanged]
    |       q_sample uses 2-D sqrt_alphas_cumprod  <- KEY CHANGE
    |
    +-- [Adaptive noise block -- th.no_grad()]  (NEW)
            per_token_mse = mean((target - output)^2, dim=-1) -> (B, S)
            pad_mask = (input_ids != pad_tok_id)              -> (B, S)
            |
            +-- [every 20 steps] log snapshot (rank 0)
            +-- _loss_history_update(t, per_token_mse, pad_mask, step)
                    +-- all_gather across GPUs
                    +-- bin into buckets, accumulate _loss_history
                    +-- [every stride steps after warmup] _refit_schedule()
                            +-- monotone interp -> new (T, S) alphas_cumprod
                            +-- update_time_discretized_parameters()
```

---

## 4. File-by-File Changes


### 4.1 `gaussian_diffusion.py` -- `__init__`

**What changed:** 6 new keyword args with safe defaults; removed `assert len(betas.shape) == 1`; added 2-D init block at end.

**Why the assert is removed:** After `_expand_schedule_to_2d()`, `self.betas` becomes `(T, S)`. The meaningful constraint `(betas > 0).all()` is kept.

**Why 1-D computation runs first:** `SpacedDiffusion` creates a temporary `GaussianDiffusion` with `adaptive_noise=False` for timestep spacing. That instance needs 1-D `alphas_cumprod`. Expanding to 2-D only at the end of `__init__` keeps the temp instance in 1-D. See section 4.10.

**Why the ramp prior for `_loss_history`:** Starting at zeros gives a flat curve for the first refit. A ramp `linspace(0, 0.5, num_buckets)` ensures the first refit produces a reasonable monotone schedule even when some buckets have few observations.

```python
def __init__(
    self, *, betas, model_mean_type, model_var_type,
    loss_type, rescale_timesteps=False, model_arch=None, training_mode='emb',
    # NEW parameters (all default to off -- zero change for existing callers)
    adaptive_noise=False,
    token_max_length=None,
    pad_tok_id=None,
    loss_update_granu=50,        # T//50 = 40 buckets for T=2000
    schedule_update_stride=500,  # refit every 500 steps
    save_dir=None,
):
    # ... all 1-D schedule computation runs here unchanged ...

    self.adaptive_noise = adaptive_noise
    self.token_max_length = token_max_length
    self.pad_tok_id = pad_tok_id
    self.save_dir = save_dir
    self.maxt = -1

    # NEW: Conditional 2-D expansion at end of __init__
    if self.adaptive_noise:
        self._loss_interp_granu = int(loss_update_granu)
        self._loss_history_update_stride = schedule_update_stride
        num_buckets = self.num_timesteps // self._loss_interp_granu

        # shape (num_buckets, S) -- ramp prior so first refit has a sensible start
        self._loss_history = (
            np.ones((num_buckets, token_max_length))
            * np.linspace(0, 0.5, num_buckets)[:, None]
        )
        self._loss_history_count = np.ones((num_buckets, token_max_length))

        self._expand_schedule_to_2d()   # (T,) -> (T, S) for all 12 arrays
```

---

### 4.2 `gaussian_diffusion.py` -- `_expand_schedule_to_2d`

**What changed:** New method. Tiles all 12 schedule arrays `(T,) -> (T, S)`.

**How this makes q_sample position-dependent without any other code change:**

All downstream functions call `_extract_into_tensor(arr, timesteps, broadcast_shape)`.

With `arr` shape `(T, S)` and `timesteps` shape `(B,)`:
- `arr[timesteps]` uses NumPy advanced integer indexing on axis 0 -> shape `(B, S)`
- The unsqueeze loop adds one dim: `(B, S) -> (B, S, 1)`
- `.expand(B, S, D)` broadcasts to full embedding shape

Result: each token position `s` gets its own noise coefficient from `alphas_cumprod[t, s]`. **No other code needs to change.**

```python
def _expand_schedule_to_2d(self):
    S = self.token_max_length
    # np.tile(arr[:, None], (1, S)):  (T,) -> (T, 1) -> (T, S)
    self.alphas_cumprod                = np.tile(self.alphas_cumprod[:, None],                (1, S))
    self.alphas_cumprod_prev           = np.tile(self.alphas_cumprod_prev[:, None],           (1, S))
    self.alphas_cumprod_next           = np.tile(self.alphas_cumprod_next[:, None],           (1, S))
    self.sqrt_alphas_cumprod           = np.tile(self.sqrt_alphas_cumprod[:, None],           (1, S))
    self.sqrt_one_minus_alphas_cumprod = np.tile(self.sqrt_one_minus_alphas_cumprod[:, None], (1, S))
    self.log_one_minus_alphas_cumprod  = np.tile(self.log_one_minus_alphas_cumprod[:, None],  (1, S))
    self.sqrt_recip_alphas_cumprod     = np.tile(self.sqrt_recip_alphas_cumprod[:, None],     (1, S))
    self.sqrt_recipm1_alphas_cumprod   = np.tile(self.sqrt_recipm1_alphas_cumprod[:, None],   (1, S))
    self.posterior_variance            = np.tile(self.posterior_variance[:, None],            (1, S))
    self.posterior_mean_coef1          = np.tile(self.posterior_mean_coef1[:, None],          (1, S))
    self.posterior_mean_coef2          = np.tile(self.posterior_mean_coef2[:, None],          (1, S))
    self.betas                         = np.tile(self.betas[:, None],                         (1, S))

    # posterior_log_variance_clipped: re-derived with vstack (not np.append)
    # because posterior_variance[1] is now shape (S,), not a scalar
    self.posterior_log_variance_clipped = np.log(
        np.vstack([self.posterior_variance[1:2, :], self.posterior_variance[1:, :]])
    )  # (T, S)
```

---

### 4.3 `gaussian_diffusion.py` -- `update_time_discretized_parameters`

**What changed:** New method. Hot-swaps the schedule in-place given a new `(T, S)` `alphas_cumprod`.

**Why pin column 0 (BOS):** If every position got a new schedule simultaneously in early training, the model could lose calibration of "what does timestep t mean?" before enough statistics are collected. Column 0 is always the original 1-D schedule, giving a stable anchor.

**Why recompute ALL posteriors:** The reverse process mean uses `posterior_mean_coef1` and `posterior_mean_coef2`, both depending on `betas` re-derived from the new `alphas_cumprod`. Stale coefficients would produce incorrect denoising steps.

```python
def update_time_discretized_parameters(self, alphas_cumprod_new):
    # Pin col 0 (BOS) to original schedule -- stability anchor
    self.alphas_cumprod[:, 1:] = alphas_cumprod_new[:, 1:]

    # Re-derive betas: alpha_t = a_t / a_{t-1},  beta_t = 1 - alpha_t
    alphas = np.zeros_like(self.alphas_cumprod)
    for i in range(len(self.alphas_cumprod)):
        alphas[i] = (self.alphas_cumprod[i] if i == 0
                     else self.alphas_cumprod[i] / self.alphas_cumprod[i - 1])
    self.betas = 1.0 - alphas  # (T, S)

    # Recompute all downstream arrays -- must stay in sync
    self.alphas_cumprod_prev = np.vstack(
        [np.ones((1, self.token_max_length)), self.alphas_cumprod[:-1]]
    )
    self.sqrt_alphas_cumprod           = np.sqrt(self.alphas_cumprod)
    self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
    self.posterior_variance = (
        self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    )
    self.posterior_log_variance_clipped = np.log(
        np.vstack([self.posterior_variance[1:2, :], self.posterior_variance[1:, :]])
    )
    self.posterior_mean_coef1 = (
        self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    )
    self.posterior_mean_coef2 = (
        (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
    )
```


### 4.4 `gaussian_diffusion.py` -- `_loss_history_update`

**What changed:** New method. Accumulates per-token MSE into a bucket table and triggers refits.

**Why coarse buckets (not per-timestep):** With T=2000 and batch size 64, each specific `t` appears only `64*500/2000 = 16` times after 500 steps -- too few. With 40 buckets: ~2400 observations each -- reliable for refit.

**Why sum + count separately:** Allows correct incremental averaging. Storing running means directly would require tracking the old count to weight new observations correctly.

**Why all_gather:** Each rank has a different data shard. Rank-0-only accumulation biases loss history toward rank-0's data. `all_gather` gives the full dataset view.

```python
def _loss_history_update(self, ts, per_token_mse, pad_mask, training_step):
    # Gather from all GPUs
    all_losses = all_gather_numpy(per_token_mse)    # (B*W, S)
    all_masks  = all_gather_numpy(pad_mask.float()) # (B*W, S)
    all_ts     = all_gather_numpy(ts).astype(int)   # (B*W,)

    # Bin: timestep 347, granularity=50 -> bucket 6  (347 // 50)
    bucketed_ts = all_ts // self._loss_interp_granu

    # Accumulate sum and count (NOT running average)
    for t_bucket, loss_row, mask_row in zip(bucketed_ts, all_losses, all_masks):
        self._loss_history[t_bucket]       += loss_row   # MSE sum
        self._loss_history_count[t_bucket] += mask_row   # non-pad count

    # Every 20 steps: diagnostic log (rank 0)
    if training_step % 20 == 0 and dist_mod.get_rank() == 0:
        avg_loss = self._loss_history / np.maximum(self._loss_history_count, 1e-8)
        # logs: table shape, mean/hard/easy (bucket,pos), bucket fill, schedule spread

    # Trigger refit after 3x stride warmup
    warmup = self._loss_history_update_stride * 3
    if training_step >= warmup and training_step % self._loss_history_update_stride == 0:
        self._refit_schedule(training_step)
```

---

### 4.5 `gaussian_diffusion.py` -- `_refit_schedule`

**What changed:** New method. The core adaptive scheduling algorithm.

**The interpolation intuition -- worked example for position s=72 (ring-closure digit `1`):**

```
Empirical loss after 1500 steps:
  bucket  0 (t=0..49):    loss=0.05  (easy: near-zero noise)
  bucket 10 (t=500..549): loss=0.30
  bucket 20 (t=1000..49): loss=0.58
  bucket 39 (t=1950..99): loss=0.80  (hard: high noise)

np.interp inverts the curve: "for loss=X, what alpha corresponds?"
  loss_val = linspace(0.05, 0.80, 2000)  -- equal loss increments
  Because the curve is steep at high t, those alpha values (low a, high noise)
  get more of the 2000 timestep slots.

Result: model trains ring-closure digit reconstruction more at high noise
        -- exactly where it needs the most practice.
```

**Why reset accumulators after each refit:** Track the model's *current* difficulty, not its performance 3000 steps ago. Fresh windows give the schedule responsiveness.

```python
def _refit_schedule(self, training_step):
    # Step 1: Average accumulated loss
    loss_dist = self._loss_history / np.maximum(self._loss_history_count, 1e-8)
    # shape: (num_buckets, S)

    # Step 2: Enforce monotonicity -- loss MUST increase with t
    # +1e-5 ensures STRICT increase, required for np.interp correctness
    for i in range(1, loss_dist.shape[0]):
        loss_dist[i, :] = np.maximum(loss_dist[i, :], loss_dist[i - 1, :] + 1e-5)

    # Step 3: Extrapolate endpoints (prevents flat constant at boundaries)
    loss_dist = np.vstack([
        loss_dist[:1, :]  - (loss_dist[1:2, :] - loss_dist[:1, :])  / 2,
        loss_dist,
        loss_dist[-1:, :] + (loss_dist[-1:, :] - loss_dist[-2:-1, :]) / 2,
    ])

    # Step 4: Per-position interpolation (core of the algorithm)
    interp_alpha_cumprod = []
    for s in range(loss_dist.shape[1]):
        # 2000 equally-spaced loss targets
        loss_val = np.linspace(loss_dist[:, s].min() - 1e-5,
                               loss_dist[:, s].max() + 1e-5,
                               self.num_timesteps)
        # Coarsen current schedule to bucket resolution
        alpha_coarse = np.concatenate([
            [np.max(self.alphas_cumprod[:, s])],
            np.mean(self.alphas_cumprod[:, s].reshape(-1, self._loss_interp_granu), axis=1),
            [np.min(self.alphas_cumprod[:, s])],
        ])
        # For each target loss, find the corresponding alpha
        interp_alpha_cumprod.append(
            np.interp(loss_val, loss_dist[:, s], alpha_coarse)
        )
    new_alphas_cumprod = np.stack(interp_alpha_cumprod).T  # (T, S)

    # Step 5: Save .npy snapshots, hot-swap, reset accumulators
    np.save(f"{self.save_dir}/alpha_cumprod_step_{training_step}.npy", new_alphas_cumprod)
    self.update_time_discretized_parameters(new_alphas_cumprod)
    self._loss_history = (np.ones((num_buckets, S))
                          * np.linspace(0, 0.5, num_buckets)[:, None])
    self._loss_history_count = np.ones((num_buckets, S))
```

---

### 4.6 `gaussian_diffusion.py` -- `_get_fixed_large_variance`

**What changed:** New helper. Fixes the broken `np.append` pattern in `p_mean_variance` for 2-D arrays.

**The bug without this fix:**
```python
# When posterior_variance is (T, S):
np.append(self.posterior_variance[1], self.betas[1:])
# posterior_variance[1] is shape (S,)  <- row vector, NOT a scalar
# betas[1:] is shape (T-1, S)
# np.append flattens -> 1-D of length T*S -> WRONG shape -> garbage variances
```

**The fix:**
```python
def _get_fixed_large_variance(self):
    if self.adaptive_noise and len(self.betas.shape) == 2:
        # 2-D path: vstack preserves (T, S) shape
        mv = np.vstack([self.posterior_variance[1:2, :], self.betas[1:, :]])  # (T, S)
    else:
        # 1-D standard path: unchanged
        mv = np.append(self.posterior_variance[1], self.betas[1:])            # (T,)
    return mv, np.log(mv)
```

**Why FIXED_LARGE uses this formula:** At `t=0`, posterior variance is 0 (log = -inf). The convention clips to `posterior_variance[1]` (smallest non-zero variance). For `t > 0`, it uses `betas` (per-step marginal variances). This gives better decoder log-likelihood.

---

### 4.7 `gaussian_diffusion.py` -- `p_mean_variance`

**What changed:** FIXED_LARGE/FIXED_SMALL selection now uses the helper instead of the hardcoded dict.

```python
# BEFORE (breaks silently for 2-D):
model_variance, model_log_variance = {
    ModelVarType.FIXED_LARGE: (
        np.append(self.posterior_variance[1], self.betas[1:]),      # <- BREAKS for 2-D
        np.log(np.append(self.posterior_variance[1], self.betas[1:])),
    ),
    ModelVarType.FIXED_SMALL: (self.posterior_variance,
                                self.posterior_log_variance_clipped),
}[self.model_var_type]

# AFTER (handles both 1-D and 2-D):
if self.model_var_type == ModelVarType.FIXED_LARGE:
    model_variance_arr, model_log_variance_arr = self._get_fixed_large_variance()
else:  # FIXED_SMALL
    model_variance_arr     = self.posterior_variance
    model_log_variance_arr = self.posterior_log_variance_clipped
model_variance     = _extract_into_tensor(model_variance_arr,     t, x.shape)
model_log_variance = _extract_into_tensor(model_log_variance_arr, t, x.shape)
```

This is one of two critical correctness fixes. Without it, inference with `adaptive_noise=True` uses corrupted variance arrays.

---

### 4.8 `gaussian_diffusion.py` -- `training_losses_e2e`

**What changed:** New `training_step=0` parameter; adaptive noise block inserted after `terms["loss"]`.

**New signature:**
```python
def training_losses_e2e(self, model, micro, t, model_kwargs=None, noise=None,
                        training_step=0):
```

**The adaptive noise block:**
```python
if self.adaptive_noise:
    with th.no_grad():   # ZERO gradient impact -- purely observational

        # Per-token MSE: reduce over embedding dim D only -> (B, S)
        # terms["mse"] uses mean_flat -> reduces over S and D -> (B,) (too coarse)
        per_token_mse = th.mean((target - model_output) ** 2, dim=-1)

        # At t=0, diffusion MSE is near-zero (almost no noise added).
        # Use anchor loss (x_start_mean vs model_out_x_start) instead --
        # it captures actual reconstruction difficulty per position.
        t0_per_token = th.mean((x_start_mean - model_out_x_start) ** 2, dim=-1)
        per_token_mse = th.where(
            t0_mask.view(-1, 1).expand_as(per_token_mse),
            t0_per_token, per_token_mse
        )

        # Padding mask: exclude pad positions from schedule
        # Pad always has the same dummy embedding -- its "difficulty" is
        # meaningless and would skew the schedule for real positions
        pad_mask = (input_ids != self.pad_tok_id)
        per_token_mse = per_token_mse * pad_mask.float()

        # Every 20 steps: log snapshot to terminal (rank 0 only)
        if training_step % 20 == 0 and dist_mod.get_rank() == 0:
            mean_per_pos = per_token_mse.mean(dim=0).cpu().numpy()  # (S,)
            top5_hard = np.argsort(mean_per_pos)[-5:][::-1]
            top5_easy = np.argsort(mean_per_pos)[:5]
            # prints: batch stats, top5 hard/easy positions, model std, noise std

        # Feed into adaptive schedule accumulator -- the key call
        self._loss_history_update(t, per_token_mse, pad_mask, training_step)
```

**Why `th.no_grad()`:** Without it, `.numpy()` calls on graph tensors fail, and in-place ops could corrupt gradients. This block is observational -- it has zero effect on `terms["loss"].backward()`.

**Why `dim=-1` for per_token_mse:** `mean_flat` reduces over ALL non-batch dims -> `(B,)`, losing position info. We reduce over only the embedding dim `D` -> `(B, S)`, preserving position-level granularity.

---

### 4.9 `gaussian_diffusion.py` -- `_extract_into_tensor`

**What changed:** Logic unchanged. Docstring updated to document 2-D support.

```python
# With arr shape (T,):    arr[timesteps] -> (B,)   -> (B,1,1) -> expand (B,S,D)
# With arr shape (T, S):  arr[timesteps] -> (B, S) -> (B,S,1) -> expand (B,S,D)
#
# NumPy advanced integer indexing on axis 0 handles the 2-D case automatically.
# No code change required. This is the key mechanism making the whole thing work.
res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
while len(res.shape) < len(broadcast_shape):
    res = res[..., None]
return res.expand(broadcast_shape)
```

This is the most elegant aspect: numpy's indexing semantics are exactly right for the 2-D case with zero special-casing.

---

### 4.10 `respace.py` -- `SpacedDiffusion.__init__`

**What changed:** Base diffusion used for spacing computation is forced to `adaptive_noise=False`.

**The bug without this fix:**
```python
# Without the fix: base_diffusion = GaussianDiffusion(**kwargs)
# If adaptive_noise=True:
#   alphas_cumprod is (T, S)
#   loop iterates: alpha_cumprod is (S,) row, not a scalar
#   new_betas becomes [(S,), (S,), ...] -> np.array -> (T_spaced, S)
#   kwargs["betas"] = (T_spaced, S) goes into super().__init__ BEFORE 1-D phase
#   -> broken schedule entirely
```

**The fix:**
```python
# Force 1-D for the temporary base diffusion used ONLY for spacing computation.
# The actual super().__init__(**kwargs) below retains adaptive_noise=True
# and correctly expands to 2-D at the end of its __init__.
base_kwargs = {**kwargs, 'adaptive_noise': False}
base_diffusion = GaussianDiffusion(**base_kwargs)
# ... spacing loop uses 1-D alpha_cumprod scalars correctly ...
# ... then super().__init__(**kwargs) expands to 2-D as designed
```

---

### 4.11 `train_util.py` -- `forward_backward`

**What changed:** Added `training_step=self.step` to `functools.partial`; added every-20-step diagnostic banner.

**Threading training_step:**
```python
# BEFORE:
compute_losses = functools.partial(
    self.diffusion.training_losses, self.ddp_model, micro, t, model_kwargs=None,
)

# AFTER:
compute_losses = functools.partial(
    self.diffusion.training_losses, self.ddp_model, micro, t, model_kwargs=None,
    training_step=self.step,    # threads step counter to training_losses_e2e
)
# Flows through: forward_backward -> training_losses(**kwargs) -> training_losses_e2e()
```

**Every-20-step diagnostic banner:**
```python
if self.step % 20 == 0 and self.rank == 0:
    total_grad_norm = sum(
        p.grad.detach().data.norm(2).item() ** 2
        for p in self.model_params if p.grad is not None
    ) ** 0.5
    t_np = t.cpu().numpy()
    print(f"[TrainLoop | step {self.step}]")
    print(f"  Loss: {loss.item():.6f}  LR: {self.opt.param_groups[0]['lr']:.3e}")
    print(f"  Grad norm (L2): {total_grad_norm:.5f}")
    print(f"  t: min={t_np.min()}, max={t_np.max()}, mean={t_np.mean():.1f}")
    print(f"  t histogram (5 bins): {np.histogram(t_np, bins=5)[0].tolist()}")
    if self.diffusion.adaptive_noise:
        spread = self.diffusion.alphas_cumprod[T//2].std()
        print(f"  Schedule spread at t=T//2: {spread:.6f}  (0=uniform, >0=adapted)")
```

---

### 4.12 `script_util.py` -- `create_gaussian_diffusion`

**What changed:** 6 new parameters added; forwarded to `SpacedDiffusion`.

```python
def create_gaussian_diffusion(
    *, steps=1000, ...,    # all existing params unchanged

    # NEW: all default to off -- zero change for existing callers
    adaptive_noise=False,
    token_max_length=None,
    pad_tok_id=None,
    loss_update_granu=50,
    schedule_update_stride=500,
    save_dir=None,
):
    return SpacedDiffusion(
        ...,   # existing kwargs unchanged
        adaptive_noise=adaptive_noise,
        token_max_length=token_max_length,
        pad_tok_id=pad_tok_id,
        loss_update_granu=loss_update_granu,
        schedule_update_stride=schedule_update_stride,
        save_dir=save_dir,
    )
    # SpacedDiffusion passes **kwargs to super().__init__ -> GaussianDiffusion
    # New params ride along in **kwargs at no cost. Only respace.py change
    # needed was the base_kwargs fix (Section 4.10).
```


---

## 5. The Algorithm End to End

### Phase 1: Warmup (steps 0 to 3*stride-1, default 0-1499)

- Every step: per-token MSE binned into 40 buckets
- Every 20 steps: 3-block diagnostic log
- No schedule changes -- model trains on uniform-tiled initial schedule
- After 1500 steps at batch=64: ~2400 observations per bucket (sufficient for refit)

### Phase 2: First Refit (step 1500)

```
Input: _loss_history (40, 128), _loss_history_count (40, 128)

1. avg = sum / count                            -> (40, 128)
2. Monotonicity: loss[i] > loss[i-1] + 1e-5    -> (40, 128)
3. Endpoint extrapolation                       -> (42, 128)
4. For each position s:
     loss_val = linspace(min_s, max_s, 2000)
     new_alpha[:,s] = np.interp(loss_val, loss_curve_s, alpha_coarse_s)
5. new_alphas_cumprod = stack                   -> (2000, 128)
6. Save .npy snapshots
7. update_time_discretized_parameters()
     - pin col 0; recompute betas, posteriors in-place
8. Reset accumulators to ramp prior
```

After step 1500: `alphas_cumprod[:, s].std() > 0` -- positions have diverged.

### Phase 3: Active Adaptation (step 1500+)

- Model trains with position-dependent noise
- Every 500 steps: fresh refit using new window
- Schedule converges to a stable difficulty map over a few refits
- Hard positions (ring closures, stereocenters) get more timesteps at high noise

---

## 6. Logging Every 20 Steps

Three distinct log sections appear every 20 steps:

### Block 1: Training Loop Banner (train_util.py)
```
============================================================
[TrainLoop | step 20]
  Loss (weighted mean):   0.482301
  MSE component:          0.361204
  Learning rate:          1.000e-04
  Grad norm (L2):         3.24512
  t sampled: min=3, max=1998, mean=987.4, std=576.1
  t histogram (5 bins):   [12, 14, 13, 11, 14]
  Adaptive noise:         ON
  Schedule status:        WARMUP (1480 steps remaining)
  Schedule spread at t=T//2: 0.000000  (0=uniform, >0=adapted)
============================================================
```

| What to watch | Meaning |
|--------------|---------|
| Grad norm spike | Training instability |
| Grad norm near 0 | Vanishing gradients |
| t histogram gap | Sampler bug |
| Schedule spread | 0 during warmup, grows after first refit |

### Block 2: Per-Token MSE Snapshot (training_losses_e2e)
```
............................................................
[E2E Loss | step 20] Per-token MSE snapshot:
  Batch:  B=64, S=128, D=16
  t sampled (first 8):     [134, 872, 1501, 44, 1998, 667, 312, 1100]
  a at sampled t (first 8): [0.9103, 0.4821, 0.0934, 0.9871, ...]
  Non-pad tokens/sample:   [45, 32, 67, 58, 41, 71, 29, 55]
  Global per-token MSE:    0.118432
  Per-pos MSE [pos 0-15]:  [0.12, 0.09, 0.14, 0.21, 0.08, ...]
  Top-5 HARDEST positions: [72, 31, 88, 14, 55] (losses: [0.31, ...])
  Top-5 EASIEST positions: [0, 1, 3, 127, 126]  (losses: [0.01, ...])
  model_output std:        0.41203
  target std:              0.48912
  Approx noise std:        0.34420
............................................................
```

| What to watch | Meaning |
|--------------|---------|
| Top-5 HARDEST | Should converge to structural SMILES positions (ring closures, stereocenters) |
| model_output std | If collapses toward 0: mode collapse (model predicts mean) |
| Approx noise std | sqrt(per_token_mse) -- effective noise amplitude |

### Block 3: Loss History Diagnostics (_loss_history_update)
```
------------------------------------------------------------
[AdaptiveNoise | step 20] Loss History Diagnostics
  Loss history shape:  (40, 128)
  Mean loss:           0.234512
  Hardest: bucket=38, pos=72, loss=0.812341
  Easiest: bucket=0,  pos=0,  loss=0.000123
  Bucket fill (mean/min/max): 16.0 / 0 / 48
  a at t=0    (mean/std): 0.9999 / 0.000000
  a at t=T//2 (mean/std): 0.2341 / 0.000000  <- grows post-refit
  a at t=T-1  (mean/std): 0.000012 / 0.000000
  Steps to next refit: 1480
------------------------------------------------------------
```

### Block 4: Refit Event (step 1500, 2000, 2500, ...)
```
============================================================
[AdaptiveNoise | step 1500] *** REFITTING SCHEDULE ***
  Monotonicity violations corrected: 47
  New schedule shape: (2000, 128)
  Max alpha change: 0.043210
  Mean alpha change: 0.006341
  a at t=T//2 std: 0.038241  (NON-ZERO -- positions diverged!)
  a at t=T//2 min/max: 0.18420 / 0.29341
  Saved .npy to: checkpoints/adaptive_schedule/
  Loss history reset for next window.
============================================================
```

---

## 7. Molecule-Specific Observations

| SMILES token | Example | Expected | Why |
|-------------|---------|----------|-----|
| Common atoms | `C`, `N`, `O` | **Easy** | High frequency, strong context |
| Hydrogen counts | `[NH2]` | Medium | Predictable from valence |
| Ring-closure digits | `1`, `2`, `3` | **Hard** | Must recall distant context |
| Branch delimiters | `(`, `)` | Medium-easy | Follow local grammar |
| Stereocenters | `@`, `@@` | **Hard** | Context-dependent, rare |
| Charged atoms | `[NH4+]`, `[O-]` | **Hard** | Rare + charge state |
| Aromatic atoms | `c`, `n` | Medium | More predictable in rings |
| Padding | `<pad>` | **Masked** | Excluded from schedule entirely |

---

## 8. How to Enable / Disable

### Enable adaptive noising:
```python
from improved_diffusion.script_util import create_gaussian_diffusion

diffusion = create_gaussian_diffusion(
    adaptive_noise=True,
    token_max_length=128,              # your SMILES max sequence length
    pad_tok_id=0,                      # your tokenizer's padding token id
    loss_update_granu=50,              # T=2000 -> 40 buckets (default)
    schedule_update_stride=500,        # first refit at step 1500 (default)
    save_dir='checkpoints/adaptive_schedule',
)
```

### Disable (default -- zero change):
```python
diffusion = create_gaussian_diffusion()
# adaptive_noise=False by default -> identical to original code
```

### Analysing saved schedules:
```python
import numpy as np
import matplotlib.pyplot as plt

# Load schedule snapshots
ac = np.load('checkpoints/adaptive_schedule/alpha_cumprod_step_5000.npy')  # (2000, 128)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Heatmap: alpha_cumprod(t, s)
axes[0].imshow(ac.T, aspect='auto', cmap='viridis')
axes[0].set_xlabel('Timestep t'); axes[0].set_ylabel('Position s')
axes[0].set_title('Adaptive schedule at step 5000')

# Divergence: std across positions vs t
axes[1].plot(ac.std(axis=1))
axes[1].set_xlabel('Timestep t'); axes[1].set_ylabel('std across positions')
axes[1].set_title('Schedule Divergence (0=uniform, >0=adapted)')

plt.tight_layout()
plt.savefig('schedule_analysis.png')
```

---

## 9. Backward Compatibility

All 6 new parameters default to `False` or `None`:

| Parameter | Default | Behaviour when default |
|-----------|---------|------------------------|
| `adaptive_noise` | `False` | All arrays stay 1-D `(T,)`, zero behaviour change |
| `token_max_length` | `None` | Not accessed |
| `pad_tok_id` | `None` | Not accessed |
| `loss_update_granu` | `50` | Not accessed |
| `schedule_update_stride` | `500` | Not accessed |
| `save_dir` | `None` | Not accessed |

**Existing training scripts require zero changes** to run as before.

---

## 10. Key Invariants and Correctness Notes

**`alphas_cumprod[:, s]` must be strictly decreasing for each `s`**

The interpolation guarantees this: `loss_dist[:, s]` is strictly increasing (enforced by `+1e-5`), and `alpha_coarse` is strictly decreasing. `np.interp` on increasing `xp` with decreasing `fp` produces a strictly decreasing output. This ensures `betas = 1 - alpha_t / alpha_{t-1} > 0` for all `t > 0`.

**Column 0 is always pinned to the original global schedule**

`update_time_discretized_parameters` writes only `[:, 1:]`. Column 0 retains the original 1-D-derived values. Prevents position 0 from receiving an extreme schedule before sufficient statistics are gathered.

**`th.no_grad()` cannot affect gradients**

The adaptive noise block computes `per_token_mse` inside `th.no_grad()`. It does not modify `model_output`, `target`, or any tensor in `terms`. `loss.backward()` sees exactly the same computation graph as before.

**`all_gather` is safe in single-GPU mode**

`dist_mod.get_world_size()` returns `1`. Gather creates a 1-element list, concatenates it, returns the original tensor. No penalty, no bug.

**The `+1e-5` ensures strict increase for np.interp**

`np.maximum(loss_dist[i], loss_dist[i-1] + 1e-5)` guarantees strictly (not just weakly) increasing `xp`. NumPy's `interp` with a non-strictly-increasing `xp` produces incorrect results without raising an error. The `1e-5` is negligible relative to typical MSE values (0.1-1.0).

**Bucket fill at early steps**

At step 0, some buckets may have 0 real observations. The ramp prior starts `_loss_history_count` at `np.ones(...)` (not zeros), so `sum / count` never produces NaN. Buckets with only prior data get the ramp value -- a sensible monotone starting curve.

---

## 11. Checkpoint Resume — Bugs Fixed

Three bugs caused the adaptive schedule to malfunction silently when resuming from a checkpoint. None of them crashed training; they simply reset or corrupted the schedule in ways that were hard to notice from the loss curve alone.

---

### 11.1 Bug 1: Warmup counter resets → immediate post-resume refit

**File:** `gaussian_diffusion.py` — `_loss_history_update`

**The problem:**

The refit trigger condition is:

```python
warmup = self._loss_history_update_stride * 3   # e.g. 500 * 3 = 1500
if training_step >= warmup and training_step % self._loss_history_update_stride == 0:
    self._refit_schedule(training_step)
```

`training_step` is correctly passed as `self.step + self.resume_step` from `train_util.py`. If you resume from step 130,000, then on the very first training batch `training_step = 130000`. Both conditions are immediately satisfied:
- `130000 >= 1500` ✓
- `130000 % 500 == 0` ✓ (checkpoints are saved every 10,000 steps, which is always a multiple of the 500-step refit stride)

So a **refit fires on the first batch after resume**, using a freshly-initialised `_loss_history` (the ramp prior from `__init__`). This has no real observations and overwrites the loaded adaptive schedule.

**The fix:**

A `_just_resumed` flag is set by `load_adaptive_schedule` after a successful load. The refit trigger checks this flag and skips exactly one boundary, then clears it.

```python
# gaussian_diffusion.py — _loss_history_update (after fix)
warmup = self._loss_history_update_stride * 3
if training_step >= warmup and training_step % self._loss_history_update_stride == 0:
    if getattr(self, '_just_resumed', False):
        # Skip the first post-resume refit to protect the loaded schedule.
        # The flag is cleared here so normal refitting resumes from the next boundary.
        print(f"[AdaptiveNoise | step {training_step}] Skipping first post-resume "
              f"refit to protect the loaded schedule.")
        self._just_resumed = False
    else:
        self._refit_schedule(training_step)
```

**Why skip only one boundary:** The flag is cleared immediately after skipping, so the very next refit boundary (500 steps later) proceeds normally with the fresh accumulated history from the resumed run.

---

### 11.2 Bug 2: Loss history not saved/restored across restarts

**File:** `gaussian_diffusion.py` — `load_adaptive_schedule`

**The problem:**

`_refit_schedule` does save `loss_history_step_N.npy` and `loss_count_step_N.npy` to disk alongside `alpha_cumprod_step_N.npy`. However, the original `load_adaptive_schedule` never loaded them back:

```python
# BEFORE: only loaded alphas_cumprod, ignored history files
def load_adaptive_schedule(self, path, load_history=False):
    if path.endswith('.npz'):
        ...
        if load_history and 'loss_history' in data:   # only worked for .npz
            ...
    else:
        alphas_cumprod = np.load(path)  # .npy: history files silently ignored
```

The `load_history` parameter existed but defaulted to `False`, and its `.npz` branch was never reachable because all saves write `.npy` files. On every resume:
- `_loss_history` reset to the fresh ramp prior
- `_loss_history_count` reset to all-ones
- Combined with Bug 1, the first refit used nothing but the prior, nuking the loaded schedule

**The fix:**

`load_adaptive_schedule` now auto-detects companion `.npy` history files by matching the step number in the filename:

```python
# AFTER: auto-detect companion loss_history files for .npy paths
def load_adaptive_schedule(self, path, load_history=True):  # default now True
    ...
    else:
        alphas_cumprod = np.load(path)
        if load_history:
            dirname     = os.path.dirname(path)
            basename    = os.path.basename(path)         # alpha_cumprod_step_130000.npy
            step_suffix = basename.replace('alpha_cumprod_step_', '')  # 130000.npy
            history_path = os.path.join(dirname, f'loss_history_step_{step_suffix}')
            count_path   = os.path.join(dirname, f'loss_count_step_{step_suffix}')
            if os.path.exists(history_path) and os.path.exists(count_path):
                self._loss_history       = np.load(history_path).copy()
                self._loss_history_count = np.load(count_path).copy()
                # history_loaded = True  (printed to log)
            else:
                # Warn; schedule still loads, just history starts fresh
                print(f"[AdaptiveNoise] No companion loss history files found ...")
    ...
    self._just_resumed = True   # Bug 1 fix: suppress first post-resume refit
```

**What to expect on a full resume:**

```
[AdaptiveNoise] Restored loss history from companion .npy files
  history : .../adaptive_schedule/loss_history_step_130000.npy
  count   : .../adaptive_schedule/loss_count_step_130000.npy
  shape   : (40, 128)
[AdaptiveNoise] Successfully loaded schedule from: alpha_cumprod_step_130000.npy
[AdaptiveNoise] _just_resumed=True: first post-resume refit will be skipped.
```

**What to expect when history files are missing** (e.g. resuming from an old checkpoint saved before this fix):

```
[AdaptiveNoise] No companion loss history files found — schedule will be loaded but history starts fresh.
  Missing: .../loss_history_step_130000.npy
  Missing: .../loss_count_step_130000.npy
[AdaptiveNoise] Successfully loaded schedule from: alpha_cumprod_step_130000.npy
[AdaptiveNoise] _just_resumed=True: first post-resume refit will be skipped.
```

In the missing-history case the schedule is still correctly loaded and protected (Bug 1 fix still applies). The model just needs one fresh accumulation window (500 steps) before the next refit.

---

### 11.3 Bug 3: train.sh resume block was commented out

**File:** `train.sh`

**The problem:**

The entire resume section of the original `train.sh` was commented out with `#`:

```bash
# RESUME_CHECKPOINT="...PLAIN_model140000.pt"
# ADAPTIVE_SCHEDULE="...alpha_cumprod_step_140000.npy"
```

Both variables were empty strings unless exported manually before calling the script. The script silently trained from scratch every run, making it appear resume logic was active when it wasn't.

**The fix:** See Section 11.4.

---

### 11.4 train.sh rewrite

`train.sh` was rewritten to match the style of `sample.sh` — all user-facing parameters live at the top in a clearly marked **User Configuration** block. No command-line arguments are needed; just edit the file before each run.

**Structure:**
```bash
# ==============================================================================
# User Configuration (Edit your parameters here)
# ==============================================================================

CUDA_DEVICE="0"

# --- Resume paths ---
# Set to "" to train from scratch, or fill in paths to resume.
RESUME_CHECKPOINT=""
# RESUME_CHECKPOINT="/path/to/checkpoints/PLAIN_model130000.pt"

ADAPTIVE_SCHEDULE=""
# ADAPTIVE_SCHEDULE="/path/to/adaptive_schedule/alpha_cumprod_step_130000.npy"

# --- Training hyperparameters ---
BATCH_SIZE=64
LR=0.00005
LR_ANNEAL_STEPS=200000
SAVE_INTERVAL=500
LOG_INTERVAL=20

# --- Adaptive noise schedule ---
ADAPTIVE_NOISE=True
TOKEN_MAX_LENGTH=128
PAD_TOK_ID=0
LOSS_UPDATE_GRANU=50
SCHEDULE_UPDATE_STRIDE=500

# ==============================================================================
```

A summary banner is printed at startup so every run logs its full configuration:

```
======================================================================
 Starting TGM-DLM Training / Resume (Adaptive Noise)
======================================================================
 Project root       : /path/to/tgm-dlm_adanoise1
 Resume checkpoint  : (None - training from scratch)
 Adaptive schedule  : (None - initialising default)
 Batch size         : 64
 LR                 : 5e-05  (anneal over 200000 steps)
 Save interval      : 500
 Adaptive noise     : True  (stride=500, granu=50)
======================================================================
```

**Typical resume workflow:**

1. Look up the latest checkpoint step in `checkpoints/` (e.g. `PLAIN_model130000.pt`)
2. Open `train.sh` and set:
   ```bash
   RESUME_CHECKPOINT="/path/to/checkpoints/PLAIN_model130000.pt"
   ADAPTIVE_SCHEDULE="/path/to/adaptive_schedule/alpha_cumprod_step_130000.npy"
   ```
3. Run `bash train.sh` — no arguments needed.

The companion `loss_history_step_130000.npy` and `loss_count_step_130000.npy` are auto-detected from the same directory as `ADAPTIVE_SCHEDULE` (Bug 2 fix), and the first post-resume refit is suppressed (Bug 1 fix).
