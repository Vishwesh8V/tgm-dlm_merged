import subprocess, sys, os, json
import torch as th
import numpy as np

ROOT_MERGED = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ROOT_ADANOISE = os.path.abspath(os.path.join(ROOT_MERGED, '..', 'tgm-dlm_adanoise'))
ROOT_MIXEDSPACE = os.path.abspath(os.path.join(ROOT_MERGED, '..', 'tgm-dlm_mixedspace'))

def run_isolated(code, root_dir):
    script = f"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(r'{root_dir}', 'improved-diffusion'))
sys.path.insert(0, os.path.join(r'{root_dir}', 'improved-diffusion', 'scripts'))
sys.path.insert(0, os.path.join(r'{root_dir}', 'transformers', 'src'))

import torch.distributed as dist
if not dist.is_initialized():
    store_file = os.path.join(tempfile.gettempdir(), f'dist_store_{{os.getpid()}}.tmp')
    if os.path.exists(store_file):
        try: os.remove(store_file)
        except Exception: pass
    store = dist.FileStore(store_file, 1)
    dist.init_process_group(backend='gloo', store=store, rank=0, world_size=1)

{code}
"""
    tmp_path = os.path.join(ROOT_MERGED, 'tests', f'_tmp_sub_{os.getpid()}.py')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(script)
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    try:
        res = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True, check=True, env=env, encoding='utf-8', errors='replace')
        return res.stdout
    except subprocess.CalledProcessError as e:
        print("SUBPROCESS ERROR:\n", e.stderr)
        raise
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# Shared Parameterized Model Code snippet to be injected into subprocesses
MODEL_DEF_CODE = """
import torch as th
import torch.nn as nn
import numpy as np

B, S, D = 4, 256, 32
vocab_size = 100

class ParameterizedModule(nn.Module):
    def __init__(self, learned_mean_embed=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, D)
        self.proj = nn.Linear(D, D)
        self.time_embed = nn.Linear(1, D)
        self.out_head = nn.Linear(D, D)
        self.lm_head = nn.Linear(D, vocab_size)
        if learned_mean_embed:
            self.mean_embed = nn.Parameter(th.randn(D))
        else:
            self.mean_embed = None

    def get_embeds(self, input_ids):
        return self.embed(input_ids)

    def get_logits(self, x):
        return self.lm_head(x)

    def forward(self, x, t, desc_state, desc_mask):
        t_emb = self.time_embed(t.float().view(-1, 1, 1)).expand_as(x)
        h = th.relu(self.proj(x) + t_emb)
        return self.out_head(h)

class ParameterizedModel(nn.Module):
    def __init__(self, learned_mean_embed=False):
        super().__init__()
        class Inner(nn.Module):
            def __init__(self, lme):
                super().__init__()
                self.module = ParameterizedModule(lme)
        self.model = Inner(learned_mean_embed)
        self.inner = self.model.module

    def forward(self, x, t, desc_state, desc_mask):
        return self.inner(x, t, desc_state, desc_mask)
"""


# ==============================================================================
# Ablation (a): Vanilla TGM-DLM Parity (Schedule, q_sample, and training_losses_e2e)
# ==============================================================================
def test_ablation_a_vanilla():
    print("=" * 80)
    print("[Ablation (a)] Vanilla TGM-DLM Parity Test (Merged defaults vs Upstream base)")
    print("=" * 80)
    code = f"""
{MODEL_DEF_CODE}
from improved_diffusion import gaussian_diffusion as gd
from improved_diffusion.respace import SpacedDiffusion

diff = SpacedDiffusion(
    use_timesteps=[i for i in range(2000)],
    betas=gd.get_named_beta_schedule('sqrt', 2000),
    model_mean_type=gd.ModelMeanType.START_X,
    model_var_type=gd.ModelVarType.FIXED_LARGE,
    loss_type=gd.LossType.E2E_MSE,
    rescale_timesteps=True,
    model_arch='transformer',
    training_mode='e2e',
)

# Seeded inputs & model
th.manual_seed(2024)
np.random.seed(2024)
model = ParameterizedModel(learned_mean_embed=False)

input_ids = th.randint(1, vocab_size, (B, S))
corrupt_ids = th.randint(1, vocab_size, (B, S))
desc_state = th.randn(B, 30, 768)
desc_mask = th.ones(B, 30).long()
micro = (input_ids, desc_state, desc_mask, corrupt_ids)

# Fixed diffusion noise and timesteps
th.manual_seed(42)
x_start = th.randn(B, S, D)
fixed_noise = th.randn(B, S, D)
t_sample = th.tensor([0, 100, 500, 1999]).long()

# 1. Test q_sample
qs = diff.q_sample(x_start, t_sample, noise=fixed_noise)

# 2. Test training_losses_e2e at t=0
th.manual_seed(777)
t0 = th.zeros(B).long()
noise0 = th.randn(B, S, D)
terms0 = diff.training_losses_e2e(model, micro, t0, noise=noise0, training_step=0)

# 3. Test training_losses_e2e at general t
th.manual_seed(888)
t_mid = th.tensor([150, 350, 800, 1400]).long()
noise_mid = th.randn(B, S, D)
terms_mid = diff.training_losses_e2e(model, micro, t_mid, noise=noise_mid, training_step=1)

out = {{
    'alphas_cumprod': diff.alphas_cumprod.tolist(),
    'posterior_variance': diff.posterior_variance.tolist(),
    'qs': qs.tolist(),
    'loss_t0': terms0['loss'].tolist(),
    'mse_t0': terms0['mse'].tolist(),
    'loss_mid': terms_mid['loss'].tolist(),
    'mse_mid': terms_mid['mse'].tolist(),
}}
import json
print('__JSON_START__' + json.dumps(out))
"""
    out_orig = run_isolated(code, ROOT_ADANOISE)
    out_merged = run_isolated(code, ROOT_MERGED)

    data_orig = json.loads(out_orig.split('__JSON_START__')[1])
    data_merged = json.loads(out_merged.split('__JSON_START__')[1])

    alpha_orig = np.array(data_orig['alphas_cumprod'], dtype=np.float64)
    alpha_merged = np.array(data_merged['alphas_cumprod'], dtype=np.float64)
    qs_orig = np.array(data_orig['qs'], dtype=np.float32)
    qs_merged = np.array(data_merged['qs'], dtype=np.float32)
    loss0_orig = np.array(data_orig['loss_t0'], dtype=np.float32)
    loss0_merged = np.array(data_merged['loss_t0'], dtype=np.float32)
    loss_mid_orig = np.array(data_orig['loss_mid'], dtype=np.float32)
    loss_mid_merged = np.array(data_merged['loss_mid'], dtype=np.float32)

    diff_alpha = np.max(np.abs(alpha_orig - alpha_merged))
    diff_qs = np.max(np.abs(qs_orig - qs_merged))
    diff_loss0 = np.max(np.abs(loss0_orig - loss0_merged))
    diff_loss_mid = np.max(np.abs(loss_mid_orig - loss_mid_merged))

    print(f"  [Schedule & q_sample]")
    print(f"    Raw max abs diff (alphas_cumprod):           {diff_alpha:.17e}")
    print(f"    Raw max abs diff (q_sample output):          {diff_qs:.17e}")
    print(f"  [training_losses_e2e on Parameterized Model]")
    print(f"    Raw max abs diff (training_losses_e2e @ t=0):  {diff_loss0:.17e}")
    print(f"    Raw max abs diff (training_losses_e2e @ t_mid):{diff_loss_mid:.17e}")

    assert diff_alpha == 0.0, f"alpha mismatch: {diff_alpha}"
    assert diff_qs == 0.0, f"qs mismatch: {diff_qs}"
    assert diff_loss0 == 0.0, f"loss_t0 mismatch: {diff_loss0}"
    assert diff_loss_mid == 0.0, f"loss_mid mismatch: {diff_loss_mid}"
    print("  -> STATUS: PASS (Exact 0.0 bitwise match across all tensors)")


# ==============================================================================
# Ablation (b): Solo Adaptive Noising (AN-only) Parity
# ==============================================================================
def test_ablation_b_an_only():
    print("\n" + "=" * 80)
    print("[Ablation (b)] Solo Adaptive Noising (AN-only) Parity Test")
    print("=" * 80)
    code = f"""
{MODEL_DEF_CODE}
from improved_diffusion import gaussian_diffusion as gd
from improved_diffusion.respace import SpacedDiffusion

diff = SpacedDiffusion(
    use_timesteps=[i for i in range(2000)],
    betas=gd.get_named_beta_schedule('sqrt', 2000),
    model_mean_type=gd.ModelMeanType.START_X,
    model_var_type=gd.ModelVarType.FIXED_LARGE,
    loss_type=gd.LossType.E2E_MSE,
    rescale_timesteps=True,
    model_arch='transformer',
    training_mode='e2e',
    adaptive_noise=True,
    token_max_length=S,
    pad_tok_id=0,
    loss_update_granu=50,
    schedule_update_stride=500,
    save_dir=os.path.join(r'{ROOT_MERGED}', 'tests', '_tmp_save_an'),
)

th.manual_seed(2024)
np.random.seed(2024)
model = ParameterizedModel(learned_mean_embed=False)

input_ids = th.randint(1, vocab_size, (B, S))
corrupt_ids = th.randint(1, vocab_size, (B, S))
desc_state = th.randn(B, 30, 768)
desc_mask = th.ones(B, 30).long()
micro = (input_ids, desc_state, desc_mask, corrupt_ids)

# 1. Test 2-D q_sample
th.manual_seed(42)
x_start = th.randn(B, S, D)
fixed_noise = th.randn(B, S, D)
t_sample = th.tensor([0, 150, 800, 1999]).long()
qs = diff.q_sample(x_start, t_sample, noise=fixed_noise)

# 2. Test training_losses_e2e at t=0
th.manual_seed(777)
t0 = th.zeros(B).long()
noise0 = th.randn(B, S, D)
terms0 = diff.training_losses_e2e(model, micro, t0, noise=noise0, training_step=0)

# 3. Test training_losses_e2e at general t
th.manual_seed(888)
t_mid = th.tensor([150, 350, 800, 1400]).long()
noise_mid = th.randn(B, S, D)
terms_mid = diff.training_losses_e2e(model, micro, t_mid, noise=noise_mid, training_step=1)

out = {{
    'alphas_cumprod_shape': list(diff.alphas_cumprod.shape),
    'alphas_cumprod': diff.alphas_cumprod.tolist(),
    'posterior_variance': diff.posterior_variance.tolist(),
    'qs': qs.tolist(),
    'loss_t0': terms0['loss'].tolist(),
    'mse_t0': terms0['mse'].tolist(),
    'loss_mid': terms_mid['loss'].tolist(),
    'mse_mid': terms_mid['mse'].tolist(),
    'loss_history': diff._loss_history.tolist(),
    'loss_history_count': diff._loss_history_count.tolist(),
}}
import json
print('__JSON_START__' + json.dumps(out))
"""
    out_orig = run_isolated(code, ROOT_ADANOISE)
    out_merged = run_isolated(code, ROOT_MERGED)

    data_orig = json.loads(out_orig.split('__JSON_START__')[1])
    data_merged = json.loads(out_merged.split('__JSON_START__')[1])

    alpha_orig = np.array(data_orig['alphas_cumprod'], dtype=np.float64)
    alpha_merged = np.array(data_merged['alphas_cumprod'], dtype=np.float64)
    qs_orig = np.array(data_orig['qs'], dtype=np.float32)
    qs_merged = np.array(data_merged['qs'], dtype=np.float32)
    loss0_orig = np.array(data_orig['loss_t0'], dtype=np.float32)
    loss0_merged = np.array(data_merged['loss_t0'], dtype=np.float32)
    loss_mid_orig = np.array(data_orig['loss_mid'], dtype=np.float32)
    loss_mid_merged = np.array(data_merged['loss_mid'], dtype=np.float32)
    hist_orig = np.array(data_orig['loss_history'], dtype=np.float64)
    hist_merged = np.array(data_merged['loss_history'], dtype=np.float64)

    diff_alpha = np.max(np.abs(alpha_orig - alpha_merged))
    diff_qs = np.max(np.abs(qs_orig - qs_merged))
    diff_loss0 = np.max(np.abs(loss0_orig - loss0_merged))
    diff_loss_mid = np.max(np.abs(loss_mid_orig - loss_mid_merged))
    diff_hist = np.max(np.abs(hist_orig - hist_merged))

    print(f"  [2-D Schedule & q_sample]")
    print(f"    2-D Schedule shape:                          {alpha_merged.shape}")
    print(f"    Raw max abs diff (2-D alphas):               {diff_alpha:.17e}")
    print(f"    Raw max abs diff (2-D q_sample):             {diff_qs:.17e}")
    print(f"  [training_losses_e2e on Parameterized Model]")
    print(f"    Raw max abs diff (training_losses_e2e @ t=0):  {diff_loss0:.17e}")
    print(f"    Raw max abs diff (training_losses_e2e @ t_mid):{diff_loss_mid:.17e}")
    print(f"    Raw max abs diff (Loss history accumulation):{diff_hist:.17e}")

    assert diff_alpha == 0.0, f"2D alpha mismatch: {diff_alpha}"
    assert diff_qs == 0.0, f"2D qs mismatch: {diff_qs}"
    assert diff_loss0 == 0.0, f"loss_t0 mismatch: {diff_loss0}"
    assert diff_loss_mid == 0.0, f"loss_mid mismatch: {diff_loss_mid}"
    assert diff_hist == 0.0, f"loss history mismatch: {diff_hist}"
    print("  -> STATUS: PASS (Exact 0.0 bitwise match across all tensors & accumulators)")

    tmp_dir = os.path.join(ROOT_MERGED, 'tests', '_tmp_save_an')
    if os.path.exists(tmp_dir):
        import shutil
        shutil.rmtree(tmp_dir)


# ==============================================================================
# Ablation (c): Solo Mixed-Space (MS-only) Parity
# ==============================================================================
def test_ablation_c_ms_only():
    print("\n" + "=" * 80)
    print("[Ablation (c)] Solo Mixed-Space (MS-only) Parity Test")
    print("=" * 80)
    code = f"""
{MODEL_DEF_CODE}
from improved_diffusion import gaussian_diffusion as gd
from improved_diffusion.respace import SpacedDiffusion

diff = SpacedDiffusion(
    use_timesteps=[i for i in range(2000)],
    betas=gd.get_named_beta_schedule('sqrt', 2000),
    model_mean_type=gd.ModelMeanType.START_X,
    model_var_type=gd.ModelVarType.FIXED_LARGE,
    loss_type=gd.LossType.E2E_MSE,
    rescale_timesteps=True,
    model_arch='transformer',
    training_mode='e2e',
    reg_rate=0.01,
    denoise=True,
    denoise_rate=0.2,
)

th.manual_seed(2024)
np.random.seed(2024)
model = ParameterizedModel(learned_mean_embed=True)

input_ids = th.randint(1, vocab_size, (B, S))
corrupt_ids = th.randint(1, vocab_size, (B, S))
desc_state = th.randn(B, 30, 768)
desc_mask = th.ones(B, 30).long()
micro = (input_ids, desc_state, desc_mask, corrupt_ids)

# 1. Test continuous and discrete q_sample
th.manual_seed(999)
x_start = th.randn(B, S, D)
fixed_noise = th.randn(B, S, D)
t_sample = th.tensor([0, 100, 500, 1999]).long()

diff.denoise = False
qs_cont = diff.q_sample(x_start, t_sample, noise=fixed_noise, mean_embed=model.inner.mean_embed)

diff.denoise = True
th.manual_seed(12345)
qs_disc = diff.q_sample(x_start, t_sample, noise=fixed_noise, mean_embed=model.inner.mean_embed)

# 2. Test training_losses_e2e at t=0 (with discrete substitution fixed seed)
th.manual_seed(777)
t0 = th.zeros(B).long()
noise0 = th.randn(B, S, D)
terms0 = diff.training_losses_e2e(model, micro, t0, noise=noise0)

# 3. Test training_losses_e2e at general t
th.manual_seed(888)
t_mid = th.tensor([150, 350, 800, 1400]).long()
noise_mid = th.randn(B, S, D)
terms_mid = diff.training_losses_e2e(model, micro, t_mid, noise=noise_mid)

out = {{
    'qs_cont': qs_cont.tolist(),
    'qs_disc': qs_disc.tolist(),
    'loss_t0': terms0['loss'].tolist(),
    'mse_t0': terms0['mse'].tolist(),
    'loss_mid': terms_mid['loss'].tolist(),
    'mse_mid': terms_mid['mse'].tolist(),
}}
import json
print('__JSON_START__' + json.dumps(out))
"""
    out_orig = run_isolated(code, ROOT_MIXEDSPACE)
    out_merged = run_isolated(code, ROOT_MERGED)

    data_orig = json.loads(out_orig.split('__JSON_START__')[1])
    data_merged = json.loads(out_merged.split('__JSON_START__')[1])

    qs_cont_orig = np.array(data_orig['qs_cont'], dtype=np.float32)
    qs_cont_merged = np.array(data_merged['qs_cont'], dtype=np.float32)
    qs_disc_orig = np.array(data_orig['qs_disc'], dtype=np.float32)
    qs_disc_merged = np.array(data_merged['qs_disc'], dtype=np.float32)
    loss0_orig = np.array(data_orig['loss_t0'], dtype=np.float32)
    loss0_merged = np.array(data_merged['loss_t0'], dtype=np.float32)
    loss_mid_orig = np.array(data_orig['loss_mid'], dtype=np.float32)
    loss_mid_merged = np.array(data_merged['loss_mid'], dtype=np.float32)

    diff_qs_cont = np.max(np.abs(qs_cont_orig - qs_cont_merged))
    diff_qs_disc = np.max(np.abs(qs_disc_orig - qs_disc_merged))
    diff_loss0 = np.max(np.abs(loss0_orig - loss0_merged))
    diff_loss_mid = np.max(np.abs(loss_mid_orig - loss_mid_merged))

    print(f"  [Mixed Space q_sample]")
    print(f"    Raw max abs diff (continuous mean-centered): {diff_qs_cont:.17e}")
    print(f"    Raw max abs diff (discrete substitution):    {diff_qs_disc:.17e}")
    print(f"  [training_losses_e2e on Parameterized Model (with reg_rate & absorption)]")
    print(f"    Raw max abs diff (training_losses_e2e @ t=0):  {diff_loss0:.17e}")
    print(f"    Raw max abs diff (training_losses_e2e @ t_mid):{diff_loss_mid:.17e}")

    assert diff_qs_cont == 0.0, f"MS continuous mismatch: {diff_qs_cont}"
    assert diff_qs_disc == 0.0, f"MS discrete mismatch: {diff_qs_disc}"
    assert diff_loss0 == 0.0, f"loss_t0 mismatch: {diff_loss0}"
    assert diff_loss_mid == 0.0, f"loss_mid mismatch: {diff_loss_mid}"
    print("  -> STATUS: PASS (Exact 0.0 bitwise match across all tensors)")


if __name__ == '__main__':
    test_ablation_a_vanilla()
    test_ablation_b_an_only()
    test_ablation_c_ms_only()
    print("\n" + "=" * 80)
    print("*** ALL 3 PARITY ABLATIONS (SCHEDULE, Q_SAMPLE & TRAINING_LOSSES_E2E) PASSED WITH 0.0 DIFF! ***")
    print("=" * 80)
