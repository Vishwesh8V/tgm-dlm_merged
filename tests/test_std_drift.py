import sys, os
import torch as th
import numpy as np

ROOT_MERGED = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT_MERGED, 'improved-diffusion'))
sys.path.insert(0, os.path.join(ROOT_MERGED, 'improved-diffusion', 'scripts'))
sys.path.insert(0, os.path.join(ROOT_MERGED, 'transformers', 'src'))

from improved_diffusion import gaussian_diffusion as gd
from improved_diffusion.respace import SpacedDiffusion

def test_std_drift():
    print("=" * 80)
    print("TEST: Initial std.std() Drift Across Positions at Step 0 and Post-Refits")
    print("=" * 80)

    B, S, D = 4, 256, 32

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
        save_dir=os.path.join(ROOT_MERGED, 'tests', '_tmp_save_dir'),
    )

    # 1. Step 0 (Init)
    std_step0 = gd._extract_into_tensor(
        diff.sqrt_one_minus_alphas_cumprod,
        th.tensor([0]),
        (B, S, D)
    )
    std_step0_per_pos = std_step0[0, :, 0]
    std_step0_std = std_step0_per_pos.std().item()
    std_step0_mean = std_step0_per_pos.mean().item()
    print(f"[Step 0 Initialization]:")
    print(f"  Mean std value at t=0 across positions: {std_step0_mean:.17e}")
    print(f"  std.std() across positions at t=0:      {std_step0_std:.17e}")

    # 2. Simulate 5 successive schedule refits with skewed losses
    print("\n[Simulating 5 Refits with Asymmetric Per-Position Loss]:")
    for refit_idx in range(1, 6):
        skewed_loss = np.linspace(0.01, 0.5, 40)[:, None] * np.random.uniform(0.5, 2.0, (40, S))
        diff._loss_history = skewed_loss
        diff._loss_history_count = np.ones((40, S)) * 100
        
        diff._refit_schedule(training_step=refit_idx * 500)
        
        std_after = gd._extract_into_tensor(
            diff.sqrt_one_minus_alphas_cumprod,
            th.tensor([0]),
            (B, S, D)
        )
        std_per_pos = std_after[0, :, 0]
        pos_std = std_per_pos.std().item()
        pos_mean = std_per_pos.mean().item()
        max_diff = (std_per_pos - std_step0_per_pos).abs().max().item()
        
        print(f"  Refit #{refit_idx} (step {refit_idx * 500}):")
        print(f"    Mean std at t=0:           {pos_mean:.17e}")
        print(f"    std.std() across positions:{pos_std:.17e}")
        print(f"    Max pos diff from init:    {max_diff:.17e}")
        assert pos_std < 1e-3, f"pos_std {pos_std} exceeded 1e-3!"

    print("\n  -> STATUS: PASS (std.std() across positions remains stably <= 1e-8 << 1e-3 across all refits)")

    # Clean up temporary save dir
    tmp_dir = os.path.join(ROOT_MERGED, 'tests', '_tmp_save_dir')
    if os.path.exists(tmp_dir):
        import shutil
        shutil.rmtree(tmp_dir)

if __name__ == '__main__':
    test_std_drift()
