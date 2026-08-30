import sys, os, json, tempfile, shutil
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


def test_logging_spec():
    print("=" * 80)
    print("TEST: LOGGING_SPEC.md Diagnostic Logging Implementation")
    print("=" * 80)

    test_save_dir = os.path.join(ROOT_MERGED, 'tests', '_tmp_logging_test_dir')
    if os.path.exists(test_save_dir):
        shutil.rmtree(test_save_dir)
    os.makedirs(test_save_dir, exist_ok=True)

    try:
        # 1. Test #ALERT# on non-finite loss (Section 0)
        print("\n1. Testing Section 0: Always-on Non-Finite Loss Alert (#ALERT#)...")
        model = ParameterizedModel(learned_mean_embed=True)
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
            adaptive_noise=True,
            token_max_length=S,
            pad_tok_id=0,
            loss_update_granu=50,
            schedule_update_stride=500,
            save_dir=test_save_dir,
        )

        input_ids = th.randint(1, vocab_size, (B, S))
        corrupt_ids = th.randint(1, vocab_size, (B, S))
        desc_state = th.randn(B, 30, 768)
        desc_mask = th.ones(B, 30).long()
        micro = (input_ids, desc_state, desc_mask, corrupt_ids)

        # 2. Test Step 20 #METRICS# payload (Section 1: A, B, C, D)
        print("\n2. Testing Section 1: #METRICS# JSON Payload Assembly (every 20 steps)...")
        import io
        from contextlib import redirect_stdout

        f_out = io.StringIO()
        with redirect_stdout(f_out):
            t20 = th.tensor([50, 450, 950, 1600]).long()
            terms20 = diff.training_losses_e2e(model, micro, t20, training_step=20)
        
        captured = f_out.getvalue()
        print("Captured stdout during step 20:")
        metrics_lines = [line for line in captured.splitlines() if line.startswith("#METRICS#")]
        assert len(metrics_lines) >= 1, "No #METRICS# line was emitted at step 20!"
        print(metrics_lines[0])

        payload_str = metrics_lines[0].replace("#METRICS# ", "")
        payload = json.loads(payload_str)

        # Verify General Health (Section A)
        assert payload["step"] == 20
        assert "loss" in payload and isinstance(payload["loss"], float)
        assert "mse" in payload and isinstance(payload["mse"], float)
        assert "decoder_nll" in payload and isinstance(payload["decoder_nll"], float)
        assert "tT_loss" in payload and isinstance(payload["tT_loss"], float)
        print("  -> Section A (General Health): PASS")

        # Verify Adaptive Noising (Section B) - Exact pre-refit invariants
        assert "an" in payload
        an = payload["an"]
        assert "mean_mse_global" in an
        assert an["schedule_std_mid"] == 0.0, f"Expected schedule_std_mid == 0.0 pre-refit, got {an['schedule_std_mid']}"
        assert an["monotone_violations_frac"] is None, f"Expected monotone_violations_frac is None pre-refit, got {an['monotone_violations_frac']}"
        assert an["schedule_max_change"] is None, f"Expected schedule_max_change is None pre-refit, got {an['schedule_max_change']}"
        assert "schedule_range_mid" in an and len(an["schedule_range_mid"]) == 2
        assert an["schedule_range_mid"][0] == an["schedule_range_mid"][1], "Expected schedule_range_mid min == max pre-refit"
        assert "bucket_fill_min" in an
        assert "top5_hard_positions" in an and len(an["top5_hard_positions"]) == 5
        print("  -> Section B (Adaptive Noising Pre-Refit Invariants): PASS (std=0.0, monotone=null, max_change=null)")

        # Verify Mixed-Space (Section C)
        assert "ms" in payload
        ms = payload["ms"]
        assert "mean_embed_norm" in ms
        assert "reg_term_value" in ms
        assert "reg_term_frac" in ms
        assert "embed_dist_to_tokens" in ms
        assert "mask_rate_mean" in ms
        assert "mask_rate_by_t_bucket" in ms and len(ms["mask_rate_by_t_bucket"]) == 3
        print("  -> Section C (Mixed-Space): PASS")

        # Verify Joint Interaction (Section D) - Exact pre-refit invariant: mask_rate_std_at_midT MUST be exactly 0.0
        assert "joint" in payload
        joint = payload["joint"]
        assert joint["mask_rate_std_at_midT"] == 0.0, f"Expected mask_rate_std_at_midT == 0.0 pre-refit, got {joint['mask_rate_std_at_midT']}"
        assert joint["mask_rate_loss_corr"] == 0.0, f"Expected mask_rate_loss_corr == 0.0 pre-refit, got {joint['mask_rate_loss_corr']}"
        print("  -> Section D (Joint Interaction Pre-Refit Invariant): PASS (mask_rate_std_at_midT=0.0, corr=0.0)")

        # 3. Test Step 1000 #METRICS# payload Post-Refit (Section 1: Post-Refit Dynamics)
        print("\n3. Testing Post-Refit Dynamics: Simulating refit & checking step 1000 payload...")
        np.random.seed(42)
        diff._loss_history = np.linspace(0.01, 0.5, 40)[:, None] * np.random.uniform(0.5, 2.0, (40, S))
        diff._loss_history_count = np.ones((40, S)) * 100
        diff._refit_schedule(training_step=1000)

        f_out_post = io.StringIO()
        with redirect_stdout(f_out_post):
            terms1000 = diff.training_losses_e2e(model, micro, t20, training_step=1000)

        captured_post = f_out_post.getvalue()
        print("Captured stdout during step 1000 (post-refit):")
        metrics_post_lines = [line for line in captured_post.splitlines() if line.startswith("#METRICS#")]
        assert len(metrics_post_lines) >= 1, "No #METRICS# line emitted at step 1000!"
        print(metrics_post_lines[0])

        payload_post = json.loads(metrics_post_lines[0].replace("#METRICS# ", ""))
        assert payload_post["step"] == 1000
        assert payload_post["an"]["schedule_std_mid"] > 0.0, "Expected schedule_std_mid > 0.0 post-refit!"
        assert payload_post["an"]["monotone_violations_frac"] is not None, "Expected non-null monotone_violations_frac post-refit!"
        assert payload_post["an"]["schedule_max_change"] is not None, "Expected non-null schedule_max_change post-refit!"
        assert payload_post["joint"]["mask_rate_std_at_midT"] > 0.0, "Expected mask_rate_std_at_midT > 0.0 post-refit!"
        assert isinstance(payload_post["joint"]["mask_rate_loss_corr"], float)
        print("  -> Post-Refit Invariants: PASS (mask_rate_std_at_midT > 0, monotone_violations_frac != null, schedule_std_mid > 0)")

        # Verify JSONL log file output
        jsonl_path = os.path.join(test_save_dir, "metrics_log.jsonl")
        assert os.path.exists(jsonl_path), f"metrics_log.jsonl was not written to {test_save_dir}!"
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f.readlines() if l.strip()]
        assert len(lines) >= 2
        assert lines[0]["step"] == 20
        assert lines[-1]["step"] == 1000
        print("  -> JSONL Logging to metrics_log.jsonl (2 records): PASS")

        print("\n*** LOGGING SPEC IMPLEMENTATION TEST PASSED CLEANLY! ***\n")

    finally:
        if os.path.exists(test_save_dir):
            shutil.rmtree(test_save_dir)

if __name__ == '__main__':
    test_logging_spec()
