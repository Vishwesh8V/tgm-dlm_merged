import sys, os
import torch as th
import torch.nn as nn
import numpy as np

ROOT_MERGED = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT_MERGED, 'improved-diffusion'))
sys.path.insert(0, os.path.join(ROOT_MERGED, 'improved-diffusion', 'scripts'))
sys.path.insert(0, os.path.join(ROOT_MERGED, 'transformers', 'src'))

from improved_diffusion import gaussian_diffusion as gd
from improved_diffusion.respace import SpacedDiffusion

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


def test_smoke_e2e_configurations():
    print("=" * 80)
    print("SMOKE TEST: Control Flow, Dynamic Shapes, and Finite Loss Sanity Check")
    print("NOTE: This is a single-repo smoke test. For cross-repo parity, see test_three_way_parity.py.")
    print("=" * 80)

    configs = [
        ('Vanilla (Baseline Defaults)', False, False, False),
        ('Adaptive Noise Solo', True, False, False),
        ('Mixed Space Solo', False, True, True),
        ('Joint AN + MS (Unified)', True, True, True),
    ]

    for name, use_an, use_lme, use_denoise in configs:
        print(f"\nOperational Mode: {name}")
        
        # Controlled seed for reproducible model and data
        th.manual_seed(2024)
        np.random.seed(2024)
        model = ParameterizedModel(learned_mean_embed=use_lme)

        diff = SpacedDiffusion(
            use_timesteps=[i for i in range(2000)],
            betas=gd.get_named_beta_schedule('sqrt', 2000),
            model_mean_type=gd.ModelMeanType.START_X,
            model_var_type=gd.ModelVarType.FIXED_LARGE,
            loss_type=gd.LossType.E2E_MSE,
            rescale_timesteps=True,
            model_arch='transformer',
            training_mode='e2e',
            reg_rate=0.01 if use_lme else 0.0,
            denoise=use_denoise,
            denoise_rate=0.2,
            adaptive_noise=use_an,
            token_max_length=S,
            pad_tok_id=0,
            loss_update_granu=50,
            schedule_update_stride=500,
            save_dir=os.path.join(ROOT_MERGED, 'tests', '_tmp_save_dir'),
        )

        input_ids = th.randint(1, vocab_size, (B, S))
        corrupt_ids = th.randint(1, vocab_size, (B, S))
        desc_state = th.randn(B, 30, 768)
        desc_mask = th.ones(B, 30).long()
        micro = (input_ids, desc_state, desc_mask, corrupt_ids)

        # Step at t=0
        th.manual_seed(777)
        t0 = th.zeros(B).long()
        noise0 = th.randn(B, S, D)
        terms0 = diff.training_losses_e2e(model, micro, t0, noise=noise0, training_step=1)
        loss0 = terms0['loss'].mean().item()
        mse0 = terms0['mse'].mean().item()

        # Step at intermediate t
        th.manual_seed(888)
        t_mid = th.tensor([100, 350, 750, 1500]).long()
        noise_mid = th.randn(B, S, D)
        terms_mid = diff.training_losses_e2e(model, micro, t_mid, noise=noise_mid, training_step=1)
        loss_mid = terms_mid['loss'].mean().item()
        mse_mid = terms_mid['mse'].mean().item()

        print(f"  t=0:   total_loss={loss0:.17e}, mse={mse0:.17e}")
        print(f"  t_mid: total_loss={loss_mid:.17e}, mse={mse_mid:.17e}")
        assert not th.isnan(terms0['loss']).any(), "NaN in t=0 loss!"
        assert not th.isnan(terms_mid['loss']).any(), "NaN in t_mid loss!"
        print("  -> SMOKE CHECK: PASS")

    tmp_dir = os.path.join(ROOT_MERGED, 'tests', '_tmp_save_dir')
    if os.path.exists(tmp_dir):
        import shutil
        shutil.rmtree(tmp_dir)

if __name__ == '__main__':
    test_smoke_e2e_configurations()
