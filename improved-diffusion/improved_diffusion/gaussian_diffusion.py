"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
print("IN SMI_ORI")
import enum
import math

import numpy as np
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood, discretized_text_log_likelihood

print("0807checkpoint in Diffusion LM REGEX AUG!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1-np.sqrt(t + 0.0001),
        )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar2(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,
        )
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  #scale * 0.02
        beta_end = scale * 0.02
        first_part = np.linspace(
            beta_start, beta_mid, 10, dtype=np.float64
        )
        second_part = np.linspace(
            beta_mid, beta_end, num_diffusion_timesteps - 10 , dtype=np.float64
        )
        return np.concatenate(
            [first_part, second_part]
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar2(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1-alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps-1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB
    E2E_KL = enum.auto()
    E2E_MSE = enum.auto()
    E2E_Simple_MSE = enum.auto()
    E2E_Simple_KL = enum.auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        model_arch=None,
        training_mode='emb',
        # model_arch='conv-unet',
        # ── Mixed-space diffusion parameters ──────────────────────────────────
        reg_rate=0.0,        # regulariser weight for mean_embed.norm()
        denoise=False,       # enable discrete absorbing substitution
        denoise_rate=0.2,    # fraction of positions substituted per step
        # ── Adaptive per-token noise schedule parameters ──────────────────────
        # These are OPTIONAL. When adaptive_noise=False (default), all behaviour
        # is identical to the original code. When adaptive_noise=True, the
        # noise schedule arrays are upgraded from shape (T,) to (T, S) where
        # S = token_max_length, giving each sequence position its own noise curve.
        adaptive_noise=False,        # master switch — safe to leave False
        adaptive_noising=False,      # alias for adaptive_noise
        token_max_length=None,       # S: molecule SMILES sequence length (e.g. 256)
        pad_tok_id=None,             # padding token id, used to mask loss contributions
        loss_update_granu=50,        # granularity: group T diffusion steps into T//granu buckets
                                     # for loss accumulation (coarser = more samples per bucket)
        schedule_update_stride=500,  # re-fit the schedule every N training steps
        save_dir=None,               # directory for saving schedule .npy snapshots
        schedule_save_dir=None,      # alias for save_dir
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.model_arch = model_arch
        self.reg_rate = reg_rate
        self.denoise = denoise
        self.denoise_rate = denoise_rate

        # ── Store adaptive noise configuration ────────────────────────────────
        # Accept both parameter names for compatibility
        self.adaptive_noise = bool(adaptive_noise or adaptive_noising)
        self.token_max_length = token_max_length
        self.pad_tok_id = pad_tok_id
        self.save_dir = save_dir or schedule_save_dir or "./generation_outputs"

        # Use float64 for accuracy in schedule arithmetic.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        # NOTE: we deliberately do NOT assert len(betas.shape) == 1 here anymore.
        # The original assertion prevented 2-D betas from being passed, but
        # SpacedDiffusion may re-derive betas from a 2-D alphas_cumprod base.
        # We verify positivity instead, which is the meaningful invariant.
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        # ── Standard 1-D noise schedule computation ───────────────────────────
        # All arrays are computed in 1-D first, regardless of adaptive_noise.
        # If adaptive_noise=True, _expand_schedule_to_2d() is called at the end
        # to tile them to shape (T, S). This two-phase approach keeps the
        # SpacedDiffusion base-computation (which needs 1-D alphas_cumprod)
        # working correctly without any changes to respace.py logic.
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)        # (T,)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])  # (T,)
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)   # (T,)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)            # (T,)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)  # (T,)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)   # (T,)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)     # (T,)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)  # (T,)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )  # (T,)
        # log calculation clipped because the posterior variance is 0 at t=0.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )  # (T,)
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )  # (T,)
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )  # (T,)

        self.training_mode = training_mode
        print('training mode is ', training_mode)
        self.mapping_func = None
        self.maxt = -1

        # ── Adaptive noising: upgrade schedule arrays from (T,) to (T, S) ─────
        # This must happen AFTER the 1-D computation above. The 1-D values are
        # used by SpacedDiffusion for its timestep-spacing logic; only after
        # that point do we expand everything to per-token 2-D arrays.
        if self.adaptive_noise:
            # Validate required parameters
            assert token_max_length is not None, (
                "[AdaptiveNoise] token_max_length is required when adaptive_noise=True. "
                "Set it to your molecule SMILES max sequence length (e.g. 256)."
            )
            assert pad_tok_id is not None, (
                "[AdaptiveNoise] pad_tok_id is required when adaptive_noise=True. "
                "This is used to mask out padding positions from the loss history."
            )
            assert self.save_dir is not None, (
                "[AdaptiveNoise] save_dir is required when adaptive_noise=True. "
                "Schedule .npy snapshots will be saved here for inspection."
            )

            self._loss_interp_granu = int(loss_update_granu)
            self._loss_history_update_stride = schedule_update_stride
            self._last_monotone_violations_frac = None
            self._last_schedule_max_change = None

            # Number of coarse timestep buckets in the loss history table.
            # We accumulate per-token MSE losses in these buckets rather than
            # at every individual timestep — this gives more robust statistics.
            num_buckets = self.num_timesteps // self._loss_interp_granu

            # ── Loss history table: shape (num_buckets, S) ────────────────────
            # Initialised with a gently increasing ramp (0 → 0.5) so the first
            # schedule re-fit has a sensible monotone prior rather than zeros.
            # A flat prior would make the monotonicity-enforcement step
            # (running-max trick) trivially satisfied and give a useless refit.
            self._loss_history = (
                np.ones((num_buckets, token_max_length))
                * np.linspace(0, 0.5, num_buckets)[:, None]
            )  # (num_buckets, S)
            # Count how many (non-padding) token observations fell into each bucket.
            # Used to normalise the accumulated loss sums into averages.
            self._loss_history_count = np.ones((num_buckets, token_max_length))  # (num_buckets, S)

            print(f"\n{'='*60}")
            print(f"[AdaptiveNoise] INITIALISED")
            print(f"  T (diffusion steps):   {self.num_timesteps}")
            print(f"  S (token_max_length):  {token_max_length}")
            print(f"  Loss buckets (T//g):   {num_buckets}  (granularity={self._loss_interp_granu})")
            print(f"  Schedule refit every:  {schedule_update_stride} steps")
            print(f"  Warmup (2x stride):    {schedule_update_stride * 2} steps")
            print(f"  Save dir:              {self.save_dir}")
            print(f"{'='*60}\n")

            # Expand all schedule arrays from (T,) → (T, S).
            # Each token position starts with the SAME global schedule.
            # The adaptive update will gradually differentiate them based on
            # per-position empirical difficulty.
            self._expand_schedule_to_2d()

    # =========================================================================
    # ADAPTIVE NOISING — helper methods
    # =========================================================================

    def _expand_schedule_to_2d(self):
        """
        Expand all noise schedule arrays from 1-D (T,) to 2-D (T, S) by
        tiling along the sequence-position axis. This is done once at init
        (and again after each schedule re-fit via update_time_discretized_parameters).

        WHY: In standard diffusion, every token position i in a sequence shares
        the same amount of noise at timestep t. After expansion, position s gets
        its OWN noise level at timestep t. This enables the adaptive schedule
        to allocate more diffusion budget to hard-to-denoise token positions
        (e.g. rare SMILES atom tokens) and less to easy ones (e.g. parentheses).

        All downstream computations (q_sample, q_posterior_mean_variance, etc.)
        already call _extract_into_tensor which handles both 1-D and 2-D arrays
        identically via numpy indexing — arr[timesteps] on a (T,S) array returns
        (B, S), which then broadcasts correctly into (B, S, D).
        """
        S = self.token_max_length
        T = self.num_timesteps

        # Tile each 1-D array of shape (T,) → (T, S)
        self.alphas_cumprod             = np.tile(self.alphas_cumprod[:, None],             (1, S))
        self.alphas_cumprod_prev        = np.tile(self.alphas_cumprod_prev[:, None],        (1, S))
        self.alphas_cumprod_next        = np.tile(self.alphas_cumprod_next[:, None],        (1, S))
        self.sqrt_alphas_cumprod        = np.tile(self.sqrt_alphas_cumprod[:, None],        (1, S))
        self.sqrt_one_minus_alphas_cumprod = np.tile(self.sqrt_one_minus_alphas_cumprod[:, None], (1, S))
        self.log_one_minus_alphas_cumprod  = np.tile(self.log_one_minus_alphas_cumprod[:, None],  (1, S))
        self.sqrt_recip_alphas_cumprod  = np.tile(self.sqrt_recip_alphas_cumprod[:, None],  (1, S))
        self.sqrt_recipm1_alphas_cumprod = np.tile(self.sqrt_recipm1_alphas_cumprod[:, None], (1, S))
        self.posterior_variance         = np.tile(self.posterior_variance[:, None],         (1, S))
        self.posterior_mean_coef1       = np.tile(self.posterior_mean_coef1[:, None],       (1, S))
        self.posterior_mean_coef2       = np.tile(self.posterior_mean_coef2[:, None],       (1, S))
        self.betas                      = np.tile(self.betas[:, None],                      (1, S))

        # posterior_log_variance_clipped needs special treatment because it was
        # constructed with np.append (scalar at index 0), not from the raw betas.
        # We re-derive it from the newly 2-D posterior_variance.
        # Formula: clip by using posterior_variance[1] at t=0 (avoids log(0)).
        self.posterior_log_variance_clipped = np.log(
            np.vstack([self.posterior_variance[1:2, :], self.posterior_variance[1:, :]])
        )  # (T, S)

        print(f"[AdaptiveNoise] _expand_schedule_to_2d: all arrays now shape {self.alphas_cumprod.shape}")
        assert self.alphas_cumprod.shape == (T, S), (
            f"Expected ({T}, {S}), got {self.alphas_cumprod.shape}"
        )

    def update_time_discretized_parameters(self, alphas_cumprod_new):
        """
        Hot-swap the noise schedule in-place using a freshly interpolated
        alphas_cumprod array of shape (T, S).

        This is called by _refit_schedule() after every schedule_update_stride
        training steps. It:
          1. Writes the new per-position schedule (columns 1..S-1 only; column 0
             / the BOS/start-of-SMILES position is kept fixed to the original
             global schedule for stability).
          2. Re-derives per-step alphas from the cumulative products.
          3. Recomputes all downstream posterior terms in 2-D.

        :param alphas_cumprod_new: np.ndarray of shape (T, S) from _refit_schedule.
        """
        # Pin column 0 (BOS/start-of-SMILES) to the original global schedule.
        # Only update positions 1..S-1 with the new adaptive values.
        self.alphas_cumprod[:, 1:] = alphas_cumprod_new[:, 1:]

        # Re-derive per-step alphas from cumulative products.
        alphas = np.zeros_like(self.alphas_cumprod)
        for i in range(len(self.alphas_cumprod)):
            if i == 0:
                alphas[i] = self.alphas_cumprod[i]          # ᾱ_0 = alpha_0
            else:
                alphas[i] = self.alphas_cumprod[i] / self.alphas_cumprod[i - 1]
        betas = 1.0 - alphas
        self.betas = betas  # (T, S)

        # Recompute all downstream terms using the updated 2-D alphas_cumprod.
        self.alphas_cumprod_prev = np.vstack(
            [np.ones((1, self.token_max_length)), self.alphas_cumprod[:-1]]
        )  # (T, S): row 0 is all-ones (no previous step at t=0)

        self.sqrt_alphas_cumprod            = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod  = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod   = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod      = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod    = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # Posterior variance: β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )  # (T, S)
        # Clip log-variance at t=0: posterior_variance[0] == 0, so take [1] instead.
        pv_clipped = np.vstack([self.posterior_variance[1:2, :], self.posterior_variance[1:, :]])
        self.posterior_log_variance_clipped = np.log(np.maximum(pv_clipped, 1e-20))  # (T, S)
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )  # (T, S)
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )  # (T, S)

    def load_adaptive_schedule(self, path, load_history=True):
        """
        Load a saved 2-D noise schedule from a .npy or .npz snapshot file.

        On resume, this method also restores the loss history accumulators so
        that the first post-resume schedule refit uses real accumulated data
        rather than the uninformative ramp prior from __init__.

        For .npy paths (alpha_cumprod_step_N.npy), companion files
        loss_history_step_N.npy and loss_count_step_N.npy are automatically
        detected in the same directory and loaded when load_history=True.

        Also sets self._just_resumed = True so that _loss_history_update will
        skip the first refit trigger — preventing the loaded schedule from being
        immediately overwritten by a refit on a boundary step.

        Args:
            path: str, path to .npy or .npz file containing alphas_cumprod
            load_history: bool, whether to restore loss history accumulators
                          (default True — always restore when files are present)
        """
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"[AdaptiveNoise] Schedule file not found: {path}")

        history_loaded = False

        if path.endswith('.npz'):
            # .npz bundles alphas_cumprod + optional history in one file
            data = np.load(path)
            alphas_cumprod = data['alphas_cumprod']
            if load_history and 'loss_history' in data and 'loss_history_count' in data:
                self._loss_history       = data['loss_history'].copy()
                self._loss_history_count = data['loss_history_count'].copy()
                history_loaded = True
                print(f"[AdaptiveNoise] Restored loss history from .npz (shape: {self._loss_history.shape})")
        else:
            # .npy path — look for companion loss_history_step_N.npy files
            alphas_cumprod = np.load(path)
            if load_history:
                dirname  = os.path.dirname(path)
                basename = os.path.basename(path)  # e.g. alpha_cumprod_step_130000.npy
                step_suffix = basename.replace('alpha_cumprod_step_', '')  # e.g. 130000.npy
                history_path = os.path.join(dirname, f'loss_history_step_{step_suffix}')
                count_path   = os.path.join(dirname, f'loss_count_step_{step_suffix}')
                if os.path.exists(history_path) and os.path.exists(count_path):
                    self._loss_history       = np.load(history_path).copy()
                    self._loss_history_count = np.load(count_path).copy()
                    history_loaded = True
                    print(f"[AdaptiveNoise] Restored loss history from companion .npy files")
                    print(f"  history : {history_path}")
                    print(f"  count   : {count_path}")
                    print(f"  shape   : {self._loss_history.shape}")
                else:
                    print(f"[AdaptiveNoise] No companion loss history files found — "
                          f"schedule will be loaded but history starts fresh.")
                    if not os.path.exists(history_path):
                        print(f"  Missing: {history_path}")
                    if not os.path.exists(count_path):
                        print(f"  Missing: {count_path}")

        if alphas_cumprod.ndim == 1:
            alphas_cumprod = np.tile(alphas_cumprod[:, None], (1, self.token_max_length or 256))

        # Handle SpacedDiffusion or differing timesteps
        if alphas_cumprod.shape[0] != self.num_timesteps:
            print(f"[AdaptiveNoise] Schedule loaded with {alphas_cumprod.shape[0]} steps, "
                  f"re-indexing to current num_timesteps={self.num_timesteps}...")
            if hasattr(self, 'timestep_map') and len(self.timestep_map) == self.num_timesteps:
                alphas_cumprod = alphas_cumprod[list(self.timestep_map)]
            else:
                indices = np.linspace(0, alphas_cumprod.shape[0] - 1, self.num_timesteps).astype(int)
                alphas_cumprod = alphas_cumprod[indices]

        self.update_time_discretized_parameters(alphas_cumprod)
        print(f"[AdaptiveNoise] Successfully loaded schedule from: {path} (shape: {self.alphas_cumprod.shape})")

        self._just_resumed = True
        print(f"[AdaptiveNoise] _just_resumed=True: first post-resume refit will be skipped.")

    def _load_time_schedule(self, path):
        """Alias for load_adaptive_schedule matching SeqDiffuSeq naming."""
        return self.load_adaptive_schedule(path)

    def _loss_history_update(self, ts, per_token_mse, pad_mask, training_step):
        """
        Accumulate per-token MSE into the loss history table, and periodically
        trigger a schedule re-fit.

        Args:
            ts:            (B,) integer timesteps for this batch
            per_token_mse: (B, S) float tensor — MSE per token, avg over embed dim D
            pad_mask:      (B, S) bool tensor  — True where token is NOT padding
            training_step: int global training step counter
        """
        import torch.distributed as dist_mod

        is_dist = dist_mod.is_available() and dist_mod.is_initialized()
        world_size = dist_mod.get_world_size() if is_dist else 1
        is_rank_0 = (not is_dist) or (dist_mod.get_rank() == 0)

        def _all_gather_numpy(tensor):
            """Gather a tensor from all ranks, return concatenated numpy array."""
            if world_size > 1:
                buf = [th.zeros_like(tensor) for _ in range(world_size)]
                dist_mod.all_gather(buf, tensor.contiguous().detach())
                return th.cat(buf, dim=0).cpu().numpy()
            return tensor.contiguous().detach().cpu().numpy()

        all_losses = _all_gather_numpy(per_token_mse)    # (B*W, S) float
        all_masks  = _all_gather_numpy(pad_mask.float()) # (B*W, S) float (1=real, 0=pad)
        ts_2d      = ts.unsqueeze(1).expand(-1, per_token_mse.shape[1])  # (B, S)
        all_ts     = _all_gather_numpy(ts_2d)[:, 0].astype(int)          # (B*W,)

        bucketed_ts = all_ts // self._loss_interp_granu   # (B*W,) in range [0, num_buckets-1]

        for t_bucket, loss_row, mask_row in zip(bucketed_ts, all_losses, all_masks):
            self._loss_history[t_bucket]       += loss_row          # sum of MSE values
            self._loss_history_count[t_bucket] += mask_row          # count of non-pad tokens

        # ── Per-step diagnostic logging every 20 steps ────────────────────────
        if training_step % 20 == 0 and is_rank_0:
            avg_loss = self._loss_history / np.maximum(self._loss_history_count, 1e-8)

            hard_idx = np.unravel_index(avg_loss.argmax(), avg_loss.shape)
            easy_idx = np.unravel_index(avg_loss.argmin(), avg_loss.shape)

            bucket_fill_mean = self._loss_history_count.mean()
            bucket_fill_min  = self._loss_history_count.min()
            bucket_fill_max  = self._loss_history_count.max()

            mid_t = self.num_timesteps // 2
            if len(self.alphas_cumprod.shape) == 2:
                schedule_spread_mid = self.alphas_cumprod[mid_t].std()
                schedule_spread_T   = self.alphas_cumprod[-1].std()
                schedule_spread_0   = self.alphas_cumprod[0].std()
                alpha_mid_mean = self.alphas_cumprod[mid_t].mean()
                alpha_T_mean   = self.alphas_cumprod[-1].mean()
            else:
                schedule_spread_mid = 0.0
                schedule_spread_T   = 0.0
                schedule_spread_0   = 0.0
                alpha_mid_mean = float(self.alphas_cumprod[mid_t])
                alpha_T_mean   = float(self.alphas_cumprod[-1])

            warmup = self._loss_history_update_stride * 2
            steps_to_refit = (
                self._loss_history_update_stride
                - (training_step % self._loss_history_update_stride)
                if training_step >= warmup
                else warmup - training_step
            )

            print(f"\n{'─'*60}")
            print(f"[AdaptiveNoise | step {training_step}] Loss History Diagnostics")
            print(f"  Loss history table shape:  {avg_loss.shape}  (buckets × S)")
            print(f"  Mean loss (all pos/time):  {avg_loss.mean():.6f}")
            print(f"  Hardest bucket+pos:        bucket={hard_idx[0]}, pos={hard_idx[1]}, "
                  f"loss={avg_loss[hard_idx]:.6f}")
            print(f"  Easiest bucket+pos:        bucket={easy_idx[0]}, pos={easy_idx[1]}, "
                  f"loss={avg_loss[easy_idx]:.6f}")
            print(f"  Per-token avg loss [0-9]:  "
                  f"{avg_loss.mean(axis=0)[:10].round(5).tolist()}")
            print(f"  Bucket fill (mean/min/max): {bucket_fill_mean:.1f} / "
                  f"{bucket_fill_min:.0f} / {bucket_fill_max:.0f}")
            print(f"  ─── Current Schedule State ───────────────────────")
            print(f"  alpha_bar at t=0      (mean/std):  {self.alphas_cumprod[0].mean():.4f} / "
                  f"{schedule_spread_0:.6f}")
            print(f"  alpha_bar at t=T//2   (mean/std):  {alpha_mid_mean:.4f} / "
                  f"{schedule_spread_mid:.6f}   <- grows as positions diverge")
            print(f"  alpha_bar at t=T-1   (mean/std):   {alpha_T_mean:.6f} / "
                  f"{schedule_spread_T:.6f}")
            print(f"  Steps to next refit:       {steps_to_refit}  "
                  f"(warmup={'DONE' if training_step >= warmup else 'in progress'})")
            print(f"{'─'*60}\n")

        # ── Check whether it is time to re-fit the schedule ───────────────────
        warmup = self._loss_history_update_stride * 2
        if training_step >= warmup and training_step % self._loss_history_update_stride == 0:
            if getattr(self, '_just_resumed', False):
                print(f"[AdaptiveNoise | step {training_step}] Skipping first post-resume "
                      f"refit to protect the loaded schedule. Normal refitting resumes next boundary.")
                self._just_resumed = False
            else:
                self._refit_schedule(training_step)

    def _refit_schedule(self, training_step):
        """
        Re-fit the per-token noise schedule using the accumulated loss history.

        Args:
            training_step: int, current global step (used for logging + filenames)
        """
        import torch.distributed as dist_mod

        print(f"\n{'='*60}")
        print(f"[AdaptiveNoise | step {training_step}] *** REFITTING SCHEDULE ***")

        # Step 1: Compute average loss per (bucket, position)
        loss_dist = self._loss_history / np.maximum(self._loss_history_count, 1e-8)

        print(f"  Raw loss_dist: mean={loss_dist.mean():.5f}, "
              f"min={loss_dist.min():.5f}, max={loss_dist.max():.5f}")

        monotone_violations = 0

        # Step 2: Enforce monotonicity with running-max trick.
        for i in range(1, loss_dist.shape[0]):
            mask = loss_dist[i, :] < loss_dist[i - 1, :] + 1e-5
            monotone_violations += int(mask.sum())
            loss_dist[i, :] = np.maximum(loss_dist[i, :], loss_dist[i - 1, :] + 1e-5)

        print(f"  Monotonicity violations corrected: {monotone_violations}")

        # Step 3: Extrapolate endpoints for stable np.interp at boundaries.
        loss_dist = np.vstack([
            loss_dist[:1, :] - (loss_dist[1:2, :] - loss_dist[:1, :]) / 2,
            loss_dist,
            loss_dist[-1:, :] + (loss_dist[-1:, :] - loss_dist[-2:-1, :]) / 2,
        ])

        # Step 4: Per-position interpolation.
        interp_alpha_cumprod = []
        for s in range(loss_dist.shape[1]):
            loss_val = np.linspace(
                np.min(loss_dist[:, s]) - 1e-5,
                np.max(loss_dist[:, s]) + 1e-5,
                self.num_timesteps
            )
            alpha_coarse = np.mean(
                self.alphas_cumprod[:, s].reshape(-1, self._loss_interp_granu), axis=1
            )
            alpha_coarse = np.concatenate([
                [np.max(self.alphas_cumprod[:, s])],
                alpha_coarse,
                [np.min(self.alphas_cumprod[:, s])],
            ])

            interp_alpha_cumprod.append(
                np.interp(loss_val, loss_dist[:, s], alpha_coarse)
            )

        new_alphas_cumprod = np.stack(interp_alpha_cumprod).T  # (T, S)

        # Step 5: Compute schedule change magnitude for diagnostics
        old_alphas = self.alphas_cumprod.copy()
        max_change = np.abs(new_alphas_cumprod - old_alphas).max()
        mean_change = np.abs(new_alphas_cumprod - old_alphas).mean()

        num_buckets = self.num_timesteps // self._loss_interp_granu
        self._last_monotone_violations_frac = float(monotone_violations) / float(num_buckets * self.token_max_length)
        self._last_schedule_max_change = float(max_change)

        print(f"  New schedule: shape={new_alphas_cumprod.shape}, "
              f"range=[{new_alphas_cumprod.min():.5f}, {new_alphas_cumprod.max():.5f}]")
        print(f"  Max alpha_bar change from previous schedule: {max_change:.6f}")
        print(f"  Mean alpha_bar change from previous schedule: {mean_change:.6f}")
        mid_t = self.num_timesteps // 2
        print(f"  alpha_bar at t=T//2, std across positions: "
              f"{new_alphas_cumprod[mid_t].std():.6f}  "
              f"(how much positions have diverged)")
        print(f"  alpha_bar at t=T//2, min/max: "
              f"{new_alphas_cumprod[mid_t].min():.5f} / {new_alphas_cumprod[mid_t].max():.5f}")

        # Step 6: Save snapshot for offline analysis
        is_dist = dist_mod.is_available() and dist_mod.is_initialized()
        is_rank_0 = (not is_dist) or (dist_mod.get_rank() == 0)
        if is_rank_0:
            import os
            os.makedirs(self.save_dir, exist_ok=True)
            np.save(
                os.path.join(self.save_dir, f"alpha_cumprod_step_{training_step}.npy"),
                new_alphas_cumprod
            )
            np.save(
                os.path.join(self.save_dir, f"loss_history_step_{training_step}.npy"),
                self._loss_history
            )
            np.save(
                os.path.join(self.save_dir, f"loss_count_step_{training_step}.npy"),
                self._loss_history_count
            )
            print(f"  Saved .npy snapshots to: {self.save_dir}/")

        # Step 7: Hot-swap the schedule
        self.update_time_discretized_parameters(new_alphas_cumprod)

        # Step 8: Reset loss history accumulators for the next window.
        num_buckets = self.num_timesteps // self._loss_interp_granu
        self._loss_history = (
            np.ones((num_buckets, self.token_max_length))
            * np.linspace(0, 0.5, num_buckets)[:, None]
        )
        self._loss_history_count = np.ones((num_buckets, self.token_max_length))

        print(f"  Loss history reset for next window.")
        print(f"{'='*60}\n")

    def _get_fixed_large_variance(self):
        """
        Returns the (model_variance, model_log_variance) arrays for
        ModelVarType.FIXED_LARGE, handling both 1-D (standard) and 2-D
        (adaptive_noise) schedule arrays.
        """
        if self.adaptive_noise and len(self.betas.shape) == 2:
            mv = np.vstack([self.posterior_variance[1:2, :], self.betas[1:, :]])  # (T, S)
        else:
            mv = np.append(self.posterior_variance[1], self.betas[1:])  # (T,)
        return mv, np.log(mv)

    # =========================================================================
    # END ADAPTIVE NOISING helpers
    # =========================================================================

    def training_losses(self, model, *args, **kwargs):
        if self.training_mode == 'e2e':
            return self.training_losses_e2e(model, *args, **kwargs)
        elif self.training_mode == 'e2e-simple':
            return self.training_losses_e2e_simple(model, *args, **kwargs)
        else:
            return self.training_losses_emb(model, *args, **kwargs)

    def calc_bpd_loop(self, model, *args, **kwargs):
        if self.training_mode == 'e2e':
            return self.calc_bpd_loop_e2e(model, *args, **kwargs)
        else:
            return self.calc_bpd_loop_emb(model, *args, **kwargs)

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None, mean_embed=None):
        """
        Diffuse the data for a given number of diffusion steps.

        Sample from q(x_t | x_0). If mean_embed is given, the forward process is
        centered on this learned soft absorbing state instead of the origin,
        following DiffuSeq-v2's mixed-space diffusion.

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :param mean_embed: if not None, the learned absorbing-state vector (shape [C]);
                           the forward process is shifted to be centered on it.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape

        if mean_embed is None:
            # Original zero-mean Gaussian diffusion — bit-identical to pre-change
            # behaviour when learned_mean_embed=False.
            x_t = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
            )
        else:
            # Mean-embed-centered forward process: shift x_start relative to the
            # absorbing state, diffuse, then shift back.
            x_t = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * (x_start - mean_embed[None, None])
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
                + mean_embed[None, None]
            )

        if self.denoise and mean_embed is not None:
            # Discrete absorbing substitution: randomly replace some positions
            # with mean_embed (the soft absorbing state). mask_rate scales with
            # the noise level so that more positions are replaced near t≈T.
            # Clamped to [0,1] to prevent Bernoulli from raising on p>1.
            mask_rate = _extract_into_tensor(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape[:2]
            ) * self.denoise_rate
            mask_rate = mask_rate.clamp(0.0, 1.0)
            random_mask = mask_rate.bernoulli()[..., None].expand(x_start.shape)
            mean_embed_expand = mean_embed[None, None].expand(x_start.shape)
            x_t = th.where(random_mask == 0, x_t, mean_embed_expand)

        return x_t

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance2(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if self.model_arch == 'conv-unet':
            B, C = x.shape[:2]
        else:
            B, C = x.size(0), x.size(-1)
        assert t.shape == (B,)

        # DEBUG:
        if 'debug_x_t' in model_kwargs:
            flag=True
            debug_x_t = model_kwargs.pop('debug_x_t')
            debug_t_batch = model_kwargs.pop('debug_t_batch')
            debug_direct_pred_eps = model_kwargs.pop('debug_direct_pred_eps')
            debug_x_start_cycle_pred = model_kwargs.pop('debug_x_start_cycle_pred')
        else:
            flag=False
        print(model_kwargs)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        # DEBUG path:
        def is_very_close(a, b):
            return (((a - b) ** 2).mean())
        direct_pred_eps = model(x, self._scale_timesteps(t), **model_kwargs)
        print(is_very_close(direct_pred_eps, model_output), 'debug 01')
        if flag:
            print(model_kwargs)
            print(is_very_close(debug_direct_pred_eps, model_output), 'debug 001')
            print(is_very_close(debug_x_t, x), 'debug 005')
            print(is_very_close(debug_t_batch.float(), t.float()), 'debug 006')
        x_start_cycle_pred = self._predict_xstart_from_eps(x_t=x, t=t, eps=direct_pred_eps)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            if self.model_arch == 'conv-unet':
                assert model_output.shape == (B, C * 2, *x.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
            else:
                assert model_output.shape == (B, x.size(1), C * 2)
                model_output, model_var_values = th.split(model_output, C, dim=-1)

            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                print('process_xstart 1')
                x = denoised_fn(x)
            if clip_denoised:
                print('process_xstart 2')
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                print('should go here')
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
                print(is_very_close(x_start_cycle_pred, pred_xstart), 'debug 02')
                if flag:
                    print(is_very_close(debug_x_start_cycle_pred, model_output), 'debug 002')
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        print(is_very_close(x_start_cycle_pred, pred_xstart), 'debug 03')
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, desc=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if self.model_arch == 'conv-unet' or self.model_arch == '1d-unet':
            B, C = x.shape[:2]
        else:
            B, C = x.size(0), x.size(-1)
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), desc[0], desc[1], **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            if self.model_arch == 'conv-unet':
                assert model_output.shape == (B, C * 2, *x.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
            elif self.model_arch == '1d-unet':
                assert model_output.shape == (B, C * 2, *x.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
            else:
                assert model_output.shape == (B, x.size(1), C * 2)
                model_output, model_var_values = th.split(model_output, C, dim=-1)

            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            # Variance selection for FIXED_LARGE / FIXED_SMALL
            # when adaptive_noise=True, self.betas and self.posterior_variance are 2-D arrays
            # of shape (T, S). _get_fixed_large_variance() handles both 1-D and 2-D.
            if self.model_var_type == ModelVarType.FIXED_LARGE:
                model_variance_arr, model_log_variance_arr = self._get_fixed_large_variance()
            else:  # FIXED_SMALL
                model_variance_arr     = self.posterior_variance
                model_log_variance_arr = self.posterior_log_variance_clipped
            model_variance     = _extract_into_tensor(model_variance_arr,     t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance_arr, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x, t)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_sample(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None,
            top_p=None, desc=None
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            desc=desc
        )
        if top_p is not None and top_p > 0:
            noise = th.randn_like(x)
            replace_mask = th.abs(noise) > top_p
            while replace_mask.any():
                noise[replace_mask] = th.randn_like(noise[replace_mask])
                replace_mask = th.abs(noise) > top_p
            assert (th.abs(noise) <= top_p).all()
        else:
            noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

        # Apply discrete absorbing substitution at inference matching training
        mean_embed = getattr(model, 'mean_embed', None)
        if self.denoise and mean_embed is not None:
            mask_rate = nonzero_mask * th.exp(0.5 * out["log_variance"]) * self.denoise_rate
            mask_rate = mask_rate[:, :, 0].clamp(0.0, 1.0)
            random_mask = mask_rate.bernoulli()[..., None].expand(x.shape)
            mean_embed_expand = mean_embed[None, None].expand(x.shape)
            sample = th.where(random_mask == 0, sample, mean_embed_expand)

        return {"sample": sample, "pred_xstart": out["pred_xstart"],
                'greedy_mean':out["mean"], 'out':out}

    def p_debug_loop(self,
                    model,
                    shape,
                    noise=None,
                    clip_denoised=True,
                    denoised_fn=None,
                    model_kwargs=None,
                    device=None,
                    progress=False,):
        final = None
        for sample in self.p_debug_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_debug_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            custom_t_start=100, 
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(custom_t_start))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        desc=None,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            top_p=top_p,
            desc=desc,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        desc=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise.to(device)
        else:
            img = th.randn(*shape, device=device)
            # Shift the initial noise toward the learned absorbing state so
            # the reverse chain starts from the correct distribution.
            if getattr(model, 'mean_embed', None) is not None:
                img = img + model.mean_embed[None, None].to(device)
        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        if desc is not None:
            print('Text Guiding Generation ......')
            desc = (desc[0].to(img.device),desc[1].to(img.device))
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    top_p=top_p,
                    desc=desc
                )
                yield out
                img = out["sample"]

    def p_sample_loop_langevin_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        langevin_func=None,
        top_p=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    top_p=top_p,
                )
                if langevin_func is not None:
                    out['t'] = t
                    out['img'] = img 
                    out = langevin_func(out)
                yield out
                img = out["sample"]

    def p_sample_loop_progressive_infill(
        self,
        model,
        shape,
        partial_enc,
        partial_mask,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        greedy=False
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            t_batch = th.tensor([self.num_timesteps - 1] * shape[0], device=device)
            partial_enc_with_noise = self.q_sample(partial_enc, t_batch)
            img = th.randn(*shape, device=device)
            img[~partial_mask] = partial_enc_with_noise[~partial_mask]
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                if i > 0:
                    partial_enc_with_noise = self.q_sample(partial_enc, t-1)
                else:
                    partial_enc_with_noise = partial_enc
                if greedy:
                    img = out["greedy_mean"]
                    img[~partial_mask] = partial_enc[~partial_mask]
                    out["sample"] = img
                else:
                    img = out["sample"]
                    img[~partial_mask] = partial_enc[~partial_mask]
                    out["sample"] = img
                yield out

    def p_sample_loop_progressive_merge(
        self,
        model,
        shape,
        partial_enc,
        partial_mask,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        greedy=False
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            t_batch = th.tensor([self.num_timesteps - 1] * shape[0], device=device)
            partial_enc_with_noise = self.q_sample(partial_enc, t_batch)
            img = th.randn(*shape, device=device)
            img[~partial_mask] = partial_enc_with_noise[~partial_mask]
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                if i > 0:
                    partial_enc_with_noise = self.q_sample(partial_enc, t-1)
                else:
                    partial_enc_with_noise = partial_enc
                if greedy:
                    img = out["greedy_mean"]
                    img[~partial_mask] = partial_enc[~partial_mask]
                    out["sample"] = img
                else:
                    img = out["sample"]
                    img[~partial_mask] = partial_enc[~partial_mask]
                    out["sample"] = img
                yield out

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
        langevin_fn=None,
        desc=None
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            desc=desc
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        if langevin_fn:
            sample=langevin_fn(sample, mean_pred, sigma, self.alphas_cumprod_prev[t[0]], t, x)
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        top_p=-1.0,
        langevin_fn=None,
        desc=None
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            langevin_fn=langevin_fn,
            desc=desc
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        langevin_fn=None,
        desc=None
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
        if desc is not None:
            print('Text Guiding Generation ......')
            desc = (desc[0].to(img.device),desc[1].to(img.device))
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                    langevin_fn=langevin_fn,
                    desc=desc
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None,
            noise=None, denoised_fn=None,
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        if model_kwargs is not None and 'input_ids' in model_kwargs:
            input_ids = model_kwargs.pop('input_ids')
            mapping_func = model_kwargs.pop('mapping_func', self.mapping_func)
        else:
            input_ids = None
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs,
            denoised_fn=denoised_fn,
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        if input_ids is not None:
            assert mapping_func is not None 
            if mapping_func is not None and th.any(t == 0):
                decoder_nll = mapping_func(out["mean"], input_ids) / out["mean"].size(-1)
            else:
                decoder_nll = th.zeros_like(x_start)
            model_kwargs['input_ids'] = input_ids
            model_kwargs['mapping_func'] = mapping_func
        else:
            decoder_nll = -discretized_gaussian_log_likelihood(
                x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
            )
            assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def _vb_terms_bpd_e2e(
            self, model, x_start, x_t, t, input_ids, get_logits, x_start_mean, x_start_log_var, clip_denoised=True,
            model_kwargs=None, noise=None, denoised_fn=None,
    ):
        """
        Get a term for the variational lower-bound.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        assert input_ids is not None
        mapping_func = model_kwargs.pop('mapping_func', self.mapping_func)

        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs,
            denoised_fn=denoised_fn,
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = self.token_discrete_loss(x_start, get_logits, input_ids)

        decoder_nll = decoder_nll / out["mean"].size(-1)
        decoder_nll = decoder_nll / np.log(2.0)

        mask_1 = (t == 0)
        if mask_1.any():
            kl_T = normal_kl(
                x_start_mean, x_start_log_var, out["mean"], out["log_variance"]
            )
            kl_T = mean_flat(kl_T) / np.log(2.0)
            kl = th.where(mask_1, kl_T, kl)

        out_mean, out_variance, \
        out_log_variance_clipped = self.q_mean_variance(x_start,
                                                        th.LongTensor([self.num_timesteps - 1]).to(x_start.device))
        kl_T = normal_kl(
            out_mean, out_log_variance_clipped, 0, 0
        )
        kl_T = mean_flat(kl_T) / np.log(2.0)

        output = kl + decoder_nll + kl_T 
        return {"output": output, "pred_xstart": out["pred_xstart"], 'kl': kl, 'decoder_nll':decoder_nll, 'kl_T':kl_T}

    def get_x_start(self, x_start_mean, std):
        """
        Using the interpolating policy OR using the convolution policy...
        :param x_start_mean:
        :return:
        """
        noise = th.randn_like(x_start_mean)
        assert noise.shape == x_start_mean.shape
        return (
             x_start_mean + std * noise
        )

    def token_discrete_loss(self, x_t, get_logits, input_ids):
        if self.model_arch == 'conv-unet' or self.model_arch == '1d-unet':
            reshaped_x_t = x_t.view(x_t.size(0), x_t.size(1), -1).permute(0, 2, 1)
        else:
            reshaped_x_t = x_t
        
        logits = get_logits(reshaped_x_t)

        loss_fct = th.nn.CrossEntropyLoss(reduction='none')
        decoder_nll = loss_fct(logits.view(-1, logits.size(-1)), input_ids.view(-1)).view(input_ids.shape)
        decoder_nll = decoder_nll.mean(dim=-1)
        return decoder_nll

    def x0_helper(self, model_output, x, t):
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            pred_prev = model_output

        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = model_output
            else:
                pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        else:
            raise NotImplementedError(self.model_mean_type)
        return {'pred_xprev':pred_prev, 'pred_xstart':pred_xstart}

    def training_losses_e2e(self, model, micro, t, model_kwargs=None, noise=None,
                            training_step=0):
        """
        Compute training losses for a single timestep (end-to-end mode).

        :param model: the model to evaluate loss on.
        :param micro: tuple of (input_ids, desc_state, desc_mask, corrupt_ids).
        :param t: a batch of timestep indices, shape (B,).
        :param model_kwargs: if not None, a dict of extra keyword arguments.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :param training_step: int, global training step counter. Used for:
            - adaptive noise schedule accumulation (passed to _loss_history_update)
            - diagnostic logging every 20 steps (only when adaptive_noise=True)
        :return: a dict with the key "loss" containing a tensor of shape [N].
        """
        input_ids = micro[0]
        desc_state = micro[1]
        desc_mask = micro[2]
        corrupt_ids = micro[3]
        assert(corrupt_ids.shape==input_ids.shape)

        #########################################
        # Mix corrupt and clean IDs based on timestep:
        # at low t (<400), use corrupted IDs to make the task harder;
        # at high t, use clean IDs. This is TGM-DLM's corruption schedule.
        mix_ids = th.where(t.reshape(-1,1)<400, corrupt_ids, input_ids)
        if t.max()>self.maxt:
            self.maxt = t.max()
            print('Recieving max t:{}'.format(self.maxt))
        ##########################################

        x_start_mean = model.model.module.get_embeds(input_ids)
        mix_start_mean = model.model.module.get_embeds(mix_ids)
        # Fetch the learned absorbing state; None when learned_mean_embed=False.
        mean_embed = model.model.module.mean_embed

        # std is extracted at t=0: this is the initial embedding noise level.
        # With adaptive_noise=True, sqrt_one_minus_alphas_cumprod is (T, S),
        # so _extract_into_tensor returns (B, S, D) — position-dependent noise.
        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                   th.tensor([0]).to(x_start_mean.device),
                                   x_start_mean.shape)
        x_start = self.get_x_start(x_start_mean, std)
        mix_start = self.get_x_start(mix_start_mean, std)

        if noise is None:
            noise = th.randn_like(mix_start)

        # q_sample with mean_embed-centering and discrete substitution
        x_t = self.q_sample(mix_start, t, noise=noise, mean_embed=mean_embed)
        get_logits = model.model.module.get_logits

        terms = {}

        if self.loss_type == LossType.E2E_KL:
            pass

        elif self.loss_type == LossType.E2E_MSE or self.loss_type == LossType.E2E_RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), desc_state, desc_mask)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                pass

            target = {
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise
            }[self.model_mean_type]

            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            model_out_x_start = self.x0_helper(model_output, x_t, t)['pred_xstart']
            t0_mask = (t == 0)
            t0_loss = mean_flat((x_start_mean - model_out_x_start) ** 2)
            terms["mse"] = th.where(t0_mask, t0_loss, terms["mse"])

            out_mean, _, _ = self.q_mean_variance(x_start, th.LongTensor([self.num_timesteps - 1]).to(x_start.device))
            tT_loss = mean_flat(out_mean ** 2)

            decoder_nll = self.token_discrete_loss(x_start, get_logits, input_ids)

            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"] + (decoder_nll + tT_loss)

            # L2 regulariser on the learned absorbing state
            if mean_embed is not None:
                terms["loss"] = terms["loss"] + self.reg_rate * mean_embed.norm(p=2).sum()

            # ─────────────────────────────────────────────────────────────────
            # ADAPTIVE NOISE SCHEDULE: Per-token MSE logging and accumulation
            # ─────────────────────────────────────────────────────────────────
            if self.adaptive_noise:
                with th.no_grad():
                    # Per-token MSE: average over embedding dimension D only.
                    per_token_mse = th.mean((target - model_output) ** 2, dim=-1)  # (B, S)

                    # At t=0, use the anchor loss instead of the diffusion MSE
                    t0_per_token = th.mean((x_start_mean - model_out_x_start) ** 2, dim=-1)  # (B, S)
                    t0_mask_broadcast = t0_mask.view(-1, 1).expand_as(per_token_mse)  # (B, S)
                    per_token_mse = th.where(t0_mask_broadcast, t0_per_token, per_token_mse)

                    # Build a padding mask: True where the token is NOT padding.
                    pad_mask = (input_ids != self.pad_tok_id)  # (B, S) bool

                    # Zero out padding positions in the MSE
                    per_token_mse = per_token_mse * pad_mask.float()

                    # ── Logging every 20 steps ────────────────────────────────
                    if training_step % 20 == 0:
                        import torch.distributed as dist_mod
                        is_dist = dist_mod.is_available() and dist_mod.is_initialized()
                        is_rank_0 = (not is_dist) or (dist_mod.get_rank() == 0)
                        if is_rank_0:
                            seq_len = input_ids.shape[1]
                            non_pad_per_sample = pad_mask.sum(dim=-1)  # (B,)

                            if len(self.alphas_cumprod.shape) == 2:
                                alpha_at_t = np.array([
                                    self.alphas_cumprod[ti.item()].mean()
                                    for ti in t[:8]
                                ])
                            else:
                                alpha_at_t = np.array([
                                    self.alphas_cumprod[ti.item()]
                                    for ti in t[:8]
                                ])

                            mean_per_pos = per_token_mse.mean(dim=0).cpu().numpy()  # (S,)
                            top5_hard = np.argsort(mean_per_pos)[-5:][::-1]
                            top5_easy = np.argsort(mean_per_pos)[:5]

                            print(f"\n{'·'*60}")
                            print(f"[E2E Loss | step {training_step}] Per-token MSE snapshot:")
                            print(f"  Batch:  B={t.shape[0]}, S={seq_len}, D={x_start.shape[-1]}")
                            print(f"  t sampled (first 8):     {t[:8].cpu().tolist()}")
                            print(f"  alpha_bar at sampled t (first 8): {alpha_at_t.round(4).tolist()}")
                            print(f"  Non-pad tokens/sample:   {non_pad_per_sample[:8].cpu().tolist()}")
                            print(f"  Global per-token MSE:    {per_token_mse.mean().item():.6f}")
                            print(f"  Per-pos MSE [pos 0-15]:  "
                                  f"{mean_per_pos[:16].round(5).tolist()}")
                            print(f"  Top-5 HARDEST positions: {top5_hard.tolist()} "
                                  f"(losses: {mean_per_pos[top5_hard].round(5).tolist()})")
                            print(f"  Top-5 EASIEST positions: {top5_easy.tolist()} "
                                  f"(losses: {mean_per_pos[top5_easy].round(5).tolist()})")
                            out_std  = model_output.std(dim=-1).mean().item()
                            targ_std = target.std(dim=-1).mean().item()
                            print(f"  model_output std (mean over B,S): {out_std:.5f}")
                            print(f"  target         std (mean over B,S): {targ_std:.5f}")
                            noise_std_at_t = per_token_mse.sqrt().mean().item()
                            print(f"  Approx noise std (sqrt per-tok MSE): {noise_std_at_t:.5f}")
                            print(f"{'·'*60}\n")

                    # ── Feed into the adaptive noise accumulation table ────────
                    self._loss_history_update(t, per_token_mse, pad_mask, training_step)
            # ─────────────────────────────────────────────────────────────────
            # END ADAPTIVE NOISE SCHEDULE BLOCK
            # ─────────────────────────────────────────────────────────────────

            # ─────────────────────────────────────────────────────────────────
            # Section 0: ALWAYS-ON NON-FINITE LOSS CHECK (every step)
            # ─────────────────────────────────────────────────────────────────
            if not th.isfinite(terms["loss"]).all():
                print(f"#ALERT# step={training_step} NON-FINITE LOSS: "
                      f"loss={terms['loss'].tolist()}")

            # ─────────────────────────────────────────────────────────────────
            # Section 1 & 2: COMPACT MACHINE-PARSEABLE #METRICS# LOGGING (every 20 steps)
            # ─────────────────────────────────────────────────────────────────
            if training_step % 20 == 0:
                import torch.distributed as dist_mod
                is_dist = dist_mod.is_available() and dist_mod.is_initialized()
                is_rank_0 = (not is_dist) or (dist_mod.get_rank() == 0)
                if is_rank_0:
                    payload = {
                        "step": int(training_step),
                        "loss": round(terms["loss"].mean().item(), 6),
                        "mse": round(terms["mse"].mean().item(), 6),
                        "decoder_nll": round(decoder_nll.mean().item(), 6),
                        "tT_loss": round(tT_loss.mean().item(), 6),
                    }

                    # Section B: Adaptive Noising Metrics
                    if self.adaptive_noise:
                        mid_t = self.num_timesteps // 2
                        if len(self.alphas_cumprod.shape) == 2:
                            schedule_std_mid = float(self.alphas_cumprod[mid_t].std())
                            schedule_range_mid = [
                                round(float(self.alphas_cumprod[mid_t].min()), 6),
                                round(float(self.alphas_cumprod[mid_t].max()), 6)
                            ]
                        else:
                            schedule_std_mid = 0.0
                            schedule_range_mid = [
                                round(float(self.alphas_cumprod[mid_t]), 6),
                                round(float(self.alphas_cumprod[mid_t]), 6)
                            ]

                        an_dict = {
                            "mean_mse_global": round(per_token_mse.mean().item(), 6),
                            "schedule_std_mid": round(schedule_std_mid, 6),
                            "schedule_range_mid": schedule_range_mid,
                            "bucket_fill_min": round(float(self._loss_history_count.min()), 2),
                            "monotone_violations_frac": (
                                round(self._last_monotone_violations_frac, 6)
                                if self._last_monotone_violations_frac is not None else None
                            ),
                            "schedule_max_change": (
                                round(self._last_schedule_max_change, 6)
                                if self._last_schedule_max_change is not None else None
                            ),
                            "top5_hard_positions": top5_hard.tolist(),
                        }
                        payload["an"] = an_dict

                    # Section C: Mixed-Space Metrics
                    if mean_embed is not None:
                        reg_term_value = (self.reg_rate * mean_embed.norm(p=2).sum()).item()
                        loss_mean = terms["loss"].mean().item()
                        ms_dict = {
                            "mean_embed_norm": round(mean_embed.norm(p=2).item(), 6),
                            "reg_term_value": round(float(reg_term_value), 6),
                            "reg_term_frac": round(float(reg_term_value) / (loss_mean + 1e-12), 6),
                            "embed_dist_to_tokens": round(
                                (x_start_mean - mean_embed[None, None]).norm(dim=-1).mean().item(), 6
                            ),
                        }
                        if self.denoise:
                            with th.no_grad():
                                mask_rate = _extract_into_tensor(
                                    self.sqrt_one_minus_alphas_cumprod, t, x_start.shape[:2]
                                ) * self.denoise_rate
                                mask_rate = th.clamp(mask_rate, 0.0, 1.0)
                                ms_dict["mask_rate_mean"] = round(mask_rate.mean().item(), 6)

                                terc1 = (t < (self.num_timesteps // 3))
                                terc2 = (t >= (self.num_timesteps // 3)) & (t < (2 * self.num_timesteps // 3))
                                terc3 = (t >= (2 * self.num_timesteps // 3))
                                terc_means = []
                                for terc_m in [terc1, terc2, terc3]:
                                    if terc_m.any():
                                        terc_means.append(round(mask_rate[terc_m].mean().item(), 4))
                                    else:
                                        terc_means.append(None)
                                ms_dict["mask_rate_by_t_bucket"] = terc_means
                        payload["ms"] = ms_dict

                    # Section D: Joint AN + MS Interaction Metrics
                    if self.adaptive_noise and self.denoise and (mean_embed is not None):
                        mid_t = self.num_timesteps // 2
                        if len(self.sqrt_one_minus_alphas_cumprod.shape) == 2:
                            mask_rate_row = np.clip(
                                self.sqrt_one_minus_alphas_cumprod[mid_t] * self.denoise_rate, 0.0, 1.0
                            )
                            mask_rate_std = float(mask_rate_row.std())
                            avg_loss_row = (
                                self._loss_history / np.maximum(self._loss_history_count, 1e-8)
                            ).mean(axis=0)
                            if mask_rate_std > 1e-8 and avg_loss_row.std() > 1e-8:
                                corr = np.corrcoef(mask_rate_row, avg_loss_row)[0, 1]
                                corr_val = round(float(corr), 4) if np.isfinite(corr) else 0.0
                            else:
                                corr_val = 0.0
                        else:
                            mask_rate_std = 0.0
                            corr_val = 0.0

                        payload["joint"] = {
                            "mask_rate_std_at_midT": round(mask_rate_std, 6),
                            "mask_rate_loss_corr": corr_val,
                        }

                    import json
                    print(f"#METRICS# {json.dumps(payload)}")

                    if hasattr(self, 'save_dir') and self.save_dir:
                        import os
                        os.makedirs(self.save_dir, exist_ok=True)
                        metrics_path = os.path.join(self.save_dir, "metrics_log.jsonl")
                        with open(metrics_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(payload) + "\n")

        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def training_losses_e2e_simple(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        assert 'input_ids' in model_kwargs
        x_start = None
        input_ids = model_kwargs.pop('input_ids').to(t.device)
        x_start_mean = model.model.module.get_embeds(input_ids)
        if self.model_arch == 'conv-unet':
            seqlen = int(np.sqrt(input_ids.size(1)))
            x_start_mean = x_start_mean.view(x_start_mean.size(0), seqlen, seqlen, x_start_mean.size(-1)).permute(0, 3,
                                                                                                                  1, 2)
        elif self.model_arch == '1d-unet':
            x_start_mean = x_start_mean.permute(0, 2, 1)
        x_start = x_start_mean
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        get_logits = model.model.module.get_logits

        terms = {}

        if self.loss_type == LossType.E2E_Simple_KL:
            raise NotImplementedError

        elif self.loss_type == LossType.E2E_Simple_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                if self.model_arch == 'conv-unet' or self.model_arch == '1d-unet':
                    B, C = x_t.shape[:2]
                else:
                    B, C = x_t.size(0), x_t.size(-1)

                if self.model_arch == 'conv-unet':
                    assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                    model_output, model_var_values = th.split(model_output, C, dim=1)
                    frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                else:
                    assert model_output.shape == (B, x_t.size(1), C * 2)
                    model_output, model_var_values = th.split(model_output, C, dim=-1)
                    frozen_out = th.cat([model_output.detach(), model_var_values], dim=-1)

                terms["vb"] = self._vb_terms_bpd_e2e(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    input_ids=input_ids,
                    get_logits=get_logits,
                    x_start_mean=x_start_mean, x_start_log_var=x_start_log_var,
                    clip_denoised=False,
                    noise=noise,
                )["output"]
                if self.loss_type == LossType.E2E_RESCALED_MSE:
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape

            ce = th.nn.CrossEntropyLoss(reduction='none')
            model_logits = get_logits(model_output)
            ce_loss = ce(model_logits.view(-1, model_logits.size(-1)), input_ids.view(-1))
            ce_loss = ce_loss.view(input_ids.shape)
            terms["ce"] = mean_flat(ce_loss)

            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["ce"]

        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop_e2e(self, model, x_start, clip_denoised=True, model_kwargs=None, denoised_fn=None):
        device = x_start.device
        batch_size = x_start.shape[0]

        input_ids = model_kwargs.pop('input_ids').to(device)
        x_start_mean = model.get_embeds(input_ids)
        if self.model_arch == 'conv-unet':
            seqlen = int(np.sqrt(input_ids.size(1)))
            x_start_mean = x_start_mean.view(x_start_mean.size(0), seqlen, seqlen, x_start_mean.size(-1)).permute(0, 3,
                                                                                                                  1, 2)
        elif self.model_arch == '1d-unet':
            x_start_mean = x_start_mean.permute(0, 2, 1)
        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                   th.tensor([0]).to(x_start_mean.device),
                                   x_start_mean.shape)
        x_start_log_var = 2 * th.log(std)
        x_start = self.get_x_start(x_start_mean, std)
        get_logits = model.get_logits

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            with th.no_grad():
                out = self._vb_terms_bpd_e2e(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    input_ids=input_ids,
                    get_logits=get_logits,
                    x_start_mean=x_start_mean, x_start_log_var=x_start_log_var,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                    noise=noise,
                    denoised_fn=denoised_fn,
                )
            if t == self.num_timesteps -1:
                assert len(vb) == 0
                vb.append(out["kl_T"])
            vb.append(out["kl"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))
        vb.append(out["decoder_nll"])

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = out["kl_T"]
        total_bpd = vb.sum(dim=1)
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }

    def calc_bpd_loop_emb(self, model, x_start, clip_denoised=True, model_kwargs=None,
                          denoised_fn=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                    noise=noise,
                    denoised_fn=denoised_fn,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a numpy array for a batch of timestep indices and
    broadcast the result to match broadcast_shape.

    This function handles BOTH the standard 1-D case and the adaptive noise 2-D case:

    Case 1 — 1-D arr of shape (T,):
        arr[timesteps] → shape (B,)
        After unsqueezing: (B, 1, 1) → expands to (B, S, D)
        Every token position at every timestep gets the SAME noise coefficient.

    Case 2 — 2-D arr of shape (T, S) [adaptive noise schedule]:
        arr[timesteps] → shape (B, S)  [numpy indexing on axis 0]
        After unsqueezing: (B, S, 1) → expands to (B, S, D)
        Each token position s gets its OWN noise coefficient at timestep t.

    The while-loop unsqueezing generalises correctly to any number of trailing
    dimensions (e.g. (B, S, D) for embeddings, (B, S) for masks, etc.) without
    any special-casing for the 2-D schedule.

    :param arr: numpy array of shape (T,) or (T, S).
    :param timesteps: a 1-D integer tensor of shape (B,) — batch of timestep indices.
    :param broadcast_shape: target shape, e.g. (B, S, D) or (B, S).
    :return: a float tensor of shape broadcast_shape with values from arr.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
