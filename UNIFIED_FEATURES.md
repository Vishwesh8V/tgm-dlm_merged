# TGM-DLM + Adaptive Noising + Mixed-Space Diffusion: Unified Implementation

This repository contains the unified, airtight implementation merging:
1. **Adaptive Noising** (`tgm-dlm_adanoise`): Upgrades the diffusion schedule from 1-D `(T,)` to 2-D `(T, S)`, adaptively tuning per-position noise allocation based on empirical reconstruction loss.
2. **Mixed-Space Diffusion** (`tgm-dlm_mixedspace`): Re-centers the forward diffusion process on a learned absorbing state vector `mean_embed`, adding discrete absorbing token substitutions during training and inference.

---

## 1. Unified Parameter Surface

| Flag | Default | Effect when Disabled (`False` / `0.0`) | Effect when Enabled |
|---|---|---|---|
| `--adaptive_noise` / `--adaptive_noising` | `False` | Standard uniform 1-D noise schedule across all sequence positions | 2-D schedule `(T, S)` refit dynamically every `schedule_update_stride` steps |
| `--token_max_length` | `256` | Tokenizer max length fallback | Sequence length `S` for the 2-D schedule table |
| `--pad_tok_id` | `0` | Default pad token ID | Padding token excluded from adaptive loss tracking |
| `--loss_update_granu` | `50` | Default timestep grouping | Buckets `T` diffusion steps into `T // granu` bins for robust MSE estimation |
| `--schedule_update_stride`| `500` / `10000` | Refit cadence in steps | Step interval at which the schedule is re-interpolated |
| `--save_dir` | Path | Output directory for schedule snapshots | Directory where `alpha_cumprod_step_N.npy` and loss logs are saved |
| `--learned_mean_embed` | `False` | Continuous Gaussian diffusion centered at the origin | Forward diffusion centered on a learned soft absorbing vector `mean_embed` (`nn.Parameter`) |
| `--denoise` | `False` | No discrete token substitution | Discrete absorbing substitution randomly replaces positions with `mean_embed` |
| `--denoise_rate` | `0.2` | Scaling factor | Controls the Bernoulli substitution probability `mask_rate` |
| `--reg_rate` | `0.0` | No regularization | L2 penalty on `mean_embed.norm(p=2).sum()` added to total loss |

---

## 2. Orthogonality & Synergy

- **Position-Adaptive Discrete Substitution:**
  When both `--adaptive_noise True` and `--denoise True` are active, `self.sqrt_one_minus_alphas_cumprod` is a `(T, S)` tensor. `_extract_into_tensor` returns a tensor with per-position values `(B, S)`, so the discrete substitution rate `mask_rate` naturally inherits token-level difficulty without additional overhead.
- **Full Backward Compatibility:**
  - Setting all flags to defaults reduces exactly to baseline `tgm-dlm`.
  - Setting only Adaptive Noise flags runs solo AN.
  - Setting only Mixed-Space flags runs solo MS.
  - Setting both unlocks joint adaptive continuous-discrete diffusion.

---

## 3. Running Training & Sampling

### Training
```bash
bash train.sh
```
Configure your experiment hyperparameters in the **User Configuration** block of [`train.sh`](file:///C:/Users/vishw/Desktop/Diffusion%20Models/Generation/Models/tgm-dlm_merged/train.sh).

### Sampling & Inference
```bash
bash sample.sh
```
In [`sample.sh`](file:///C:/Users/vishw/Desktop/Diffusion%20Models/Generation/Models/tgm-dlm_merged/sample.sh), you can specify model checkpoints, random seed sweeps, and adaptive schedule paths.
