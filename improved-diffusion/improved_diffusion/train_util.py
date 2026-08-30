import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
import wandb
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        checkpoint_path='',
        gradient_clipping=-1.,
        eval_data=None,
        eval_interval=-1,
        resume_warmup_steps=300,
    ):
        print("IN AUG trainutil")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print("initialing Trainer for",rank,'/',world_size)
        self.rank = rank
        self.world_size = world_size
        self.diffusion = diffusion
        self.data = data
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr*world_size
        print("ori lr:",lr,"new lr:",self.lr)
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.gradient_clipping = gradient_clipping
        self.resume_warmup_steps = resume_warmup_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()
        print('checkpoint_path:{}'.format(checkpoint_path))
        self.checkpoint_path = checkpoint_path # DEBUG **
        
        self.model = model.to(rank)
       
        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        
        

        if th.cuda.is_available(): # DEBUG **
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[self.rank],
                # device_ids=[dist_util.dev()],
                # output_device=dist_util.dev(),
                # broadcast_buffers=False,
                # bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            assert False
            # if dist.get_world_size() > 1:
            #     logger.warn(
            #         "Distributed training requires CUDA. "
            #         "Gradients will not be synchronized properly!"
            #     )
            # self.use_ddp = False
            # self.ddp_model = self.model
        self.model_params = list(self.ddp_model.parameters())
        self.master_params = self.model_params
        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            # self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint and os.path.exists(resume_checkpoint):
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                print(f"[Resume] Loading model from checkpoint: {resume_checkpoint} (resume_step={self.resume_step})...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        print('START LOOP FLAG')
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps//self.world_size
        ):
            batch = next(self.data)
            cond = None
            # if self.step<3:
            #     print("RANK:",self.rank,"STEP:",self.step,"BATCH:",batch)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                # dist.barrier()
                pass
                # print('loggggg')
                #logger.dumpkvs()
            if self.eval_data is not None and self.step % self.eval_interval == 0:
                # batch_eval, cond_eval = next(self.eval_data)
                # self.forward_only(batch, cond)
                print('eval on validation set')
                pass# logger.dumpkvs()
            if self.step % self.save_interval == 0 and self.step!=0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_only(self, batch, cond):
        with th.no_grad():
            zero_grad(self.model_params)
            for i in range(0, batch.shape[0], self.microbatch):
                micro = batch[i: i + self.microbatch].to(dist_util.dev())
                micro_cond = {
                    k: v[i: i + self.microbatch].to(dist_util.dev())
                    for k, v in cond.items()
                }
                last_batch = (i + self.microbatch) >= batch.shape[0]
                t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
                # print(micro_cond.keys())
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    micro,
                    t,
                    model_kwargs=micro_cond,
                )

                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

                log_loss_dict(
                    self.diffusion, t, {f"eval_{k}": v * weights for k, v in losses.items()}
                )


    def forward_backward(self, batch, cond):
        # zero_grad(self.model_params)
        self.opt.zero_grad()
        for i in range(0, batch[0].shape[0], self.microbatch):
            micro = (batch[0].to(self.rank),batch[1].to(self.rank),batch[2].to(self.rank),batch[3].to(self.rank))
            last_batch = True
            t, weights = self.schedule_sampler.sample(micro[0].shape[0], self.rank)

            # ── Pass training_step into training_losses_e2e ───────────────────
            # WHY: training_losses_e2e uses self.step to:
            #   1. Log per-token MSE diagnostics every 20 steps
            #   2. Feed the loss history table for adaptive noise accumulation
            #   3. Trigger schedule re-fits at the configured stride
            # The step counter is threaded through functools.partial so the
            # diffusion object doesn't need a direct reference to TrainLoop.
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=None,
                training_step=self.step + self.resume_step,  # global step accounting for resume
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()

            # ── Original per-log_interval print ───────────────────────────────
            if self.step % self.log_interval == 0 and self.rank==0:
                print("rank0: ",self.step,loss.item())
                wandb.log({'loss':loss.item()})

            # ── Comprehensive diagnostic banner every 20 steps ────────────────
            # This is separate from the adaptive noise logs in gaussian_diffusion.py.
            # Here we log training-loop-level quantities: loss, LR, gradient norms,
            # timestep distribution, and adaptive noise status. All guarded by
            # self.rank == 0 to avoid duplicate output in multi-GPU training.
            if self.step % 20 == 0 and self.rank == 0:
                current_lr = self.opt.param_groups[0]['lr']

                # Timestep distribution summary for this micro-batch.
                # A balanced distribution across [0, T] is important for learning.
                t_np = t.cpu().numpy()

                print(f"\n{'='*60}")
                print(f"[TrainLoop | step {self.step}]")
                print(f"  Loss (weighted mean):   {loss.item():.6f}")
                if "mse" in losses:
                    print(f"  MSE component:          {losses['mse'].mean().item():.6f}")
                print(f"  Learning rate:          {current_lr:.3e}")
                print(f"  t sampled: min={t_np.min()}, max={t_np.max()}, "
                      f"mean={t_np.mean():.1f}, std={t_np.std():.1f}")
                print(f"  t histogram (5 bins):   "
                      f"{np.histogram(t_np, bins=5)[0].tolist()}")
                # Log adaptive noise status if enabled
                if hasattr(self.diffusion, 'adaptive_noise') and self.diffusion.adaptive_noise:
                    warmup  = self.diffusion._loss_history_update_stride * 3
                    stride  = self.diffusion._loss_history_update_stride
                    in_warmup = self.step < warmup
                    steps_to_next = stride - (self.step % stride) if not in_warmup else warmup - self.step
                    print(f"  Adaptive noise:         ON")
                    print(f"  Schedule status:        "
                          f"{'WARMUP (' + str(steps_to_next) + ' steps remaining)' if in_warmup else 'ACTIVE (next refit in ' + str(steps_to_next) + ' steps)'}")
                    if len(self.diffusion.alphas_cumprod.shape) == 2:
                        mid_t = self.diffusion.num_timesteps // 2
                        spread = self.diffusion.alphas_cumprod[mid_t].std()
                        print(f"  alpha_bar spread at t=T//2:     {spread:.6f}  "
                              f"(0=identical schedules, >0=adapted)")
                else:
                    print(f"  Adaptive noise:         OFF (standard 1-D schedule)")
                print(f"{'='*60}\n")

            if self.use_fp16:
                # loss_scale = 2 ** self.lg_loss_scale
                # (loss * loss_scale).backward()
                pass
            else:
                loss.backward()

            # ── Emission of Trainer Metrics after backward ────────────────────
            if self.step % 20 == 0 and self.rank == 0:
                total_grad_norm = 0.0
                for p in self.model_params:
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                total_grad_norm = total_grad_norm ** 0.5

                current_lr = self.opt.param_groups[0]['lr']

                trainer_payload = {
                    "step": int(self.step + self.resume_step),
                    "grad_norm": round(float(total_grad_norm), 6),
                    "lr": current_lr,
                }

                # Check for mean_embed gradient norm
                mean_embed_grad_norm = None
                for name, param in self.model.named_parameters():
                    if "mean_embed" in name and param.grad is not None:
                        mean_embed_grad_norm = param.grad.detach().norm(2).item()
                        break
                if mean_embed_grad_norm is not None:
                    trainer_payload["mean_embed_grad_norm"] = round(float(mean_embed_grad_norm), 6)

                import json
                print(f"#METRICS# {json.dumps(trainer_payload)}")

                save_dir = getattr(self.diffusion, 'save_dir', None) or getattr(self, 'checkpoint_path', None)
                if save_dir:
                    import os
                    os.makedirs(save_dir, exist_ok=True)
                    metrics_path = os.path.join(save_dir, "metrics_log.jsonl")
                    with open(metrics_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(trainer_payload) + "\n")


    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def grad_clip(self):
        # print('doing gradient clipping')
        max_grad_norm=self.gradient_clipping #3.0
        if hasattr(self.opt, "clip_grad_norm"):
            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
            self.opt.clip_grad_norm(max_grad_norm)
        # else:
        #     assert False
        # elif hasattr(self.model, "clip_grad_norm_"):
        #     # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
        #     self.model.clip_grad_norm_(args.max_grad_norm)
        else:
            # Revert to normal clipping otherwise, handling Apex or full precision
            th.nn.utils.clip_grad_norm_(
                self.model.parameters(), #amp.master_params(self.opt) if self.use_apex else
                max_grad_norm,
            )

    def optimize_normal(self):
        if self.gradient_clipping > 0:
            self.grad_clip()
        # self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        # logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return

        # The LR this step should be at according to the global linear decay.
        # Uses (self.step + self.resume_step) so the decay continues from the
        # correct position in the schedule, whether resuming or training fresh.
        global_step = self.step + self.resume_step
        frac_done = global_step / self.lr_anneal_steps
        lr_target = self.lr * (1 - frac_done)

        # ── Post-resume LR warm-up ─────────────────────────────────────────────
        # When resuming, Adam's moment estimates (m, v) are reset to zero because
        # the optimiser state is not saved/loaded. This causes an effective
        # "step-size spike" for the first ~few hundred steps — the adaptive
        # scale v starts at zero, making lr / (sqrt(v) + eps) temporarily large.
        #
        # Fix: ramp from WARMUP_START_FRAC * lr_target → lr_target linearly over
        # resume_warmup_steps steps. By the time we reach the full lr_target,
        # v has had enough updates to be a reliable scale estimate again.
        #
        # Only activates when self.resume_step > 0 (i.e. an actual resume).
        # Has zero effect on fresh training runs.
        WARMUP_START_FRAC = 0.1  # start warm-up at 10 % of the annealed LR
        if self.resume_step > 0 and self.step < self.resume_warmup_steps:
            warmup_progress = self.step / max(self.resume_warmup_steps, 1)  # 0 → 1
            scale = WARMUP_START_FRAC + (1.0 - WARMUP_START_FRAC) * warmup_progress
            lr = lr_target * scale
            if self.step % 20 == 0 and self.rank == 0:
                print(f"  [LR warm-up] step {self.step}/{self.resume_warmup_steps} "
                      f"— lr {lr:.3e} (target {lr_target:.3e}, scale {scale:.3f})")
        else:
            lr = lr_target

        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                # logger.log(f"saving model {rate}...")
                print(f"saving model {rate}...")
                if not rate:
                    filename = f"PLAIN_model{((self.step+self.resume_step)*self.world_size):06d}.pt"
                else:
                    filename = f"PLAIN_ema_{rate}_{((self.step+self.resume_step)*self.world_size):06d}.pt"
                # print('writing to', bf.join(get_blob_logdir(), filename))
                # print('writing to', bf.join(self.checkpoint_path, filename))
                # with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                #     th.save(state_dict, f)
                with bf.BlobFile(bf.join(self.checkpoint_path, filename), "wb") as f: # DEBUG **
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # if dist.get_rank() == 0: # DEBUG **
        #     with bf.BlobFile(
        #         bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
        #         "wb",
        #     ) as f:
        #         th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                list(self.model.parameters()), master_params # DEBUG **
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        cleaned_state_dict = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
        params = []
        for name, p in self.model.named_parameters():
            if name in cleaned_state_dict:
                tensor = cleaned_state_dict[name]
                params.append(tensor.to(p.device) if hasattr(tensor, "to") else tensor)
            else:
                params.append(p.data.clone())
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse step integer from filenames like PLAIN_model120000.pt, PLAIN_ema_0.9999_120000.pt, 140k.pt, etc.
    """
    import re
    match_k = re.search(r'(\d+)[kK]\.pt$', filename)
    if match_k:
        return int(match_k.group(1)) * 1000
    match = re.search(r'(\d+)\.pt$', filename)
    if match:
        return int(match.group(1))
    return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    # If the main checkpoint itself is an EMA checkpoint for this rate
    if f"ema_{rate}" in os.path.basename(main_checkpoint) and bf.exists(main_checkpoint):
        return main_checkpoint

    dir_path = bf.dirname(main_checkpoint)
    candidates = [
        f"PLAIN_ema_{rate}_{(step):06d}.pt",
        f"ema_{rate}_{(step):06d}.pt",
        f"PLAIN_ema_{rate}_{step}.pt",
        f"ema_{rate}_{step}.pt",
    ]
    if step and step % 1000 == 0:
        candidates.extend([
            f"PLAIN_ema_{rate}_{step // 1000}k.pt",
            f"ema_{rate}_{step // 1000}k.pt",
            f"PLAIN_ema_{rate}_{step // 1000}K.pt",
            f"ema_{rate}_{step // 1000}K.pt",
        ])
    for filename in candidates:
        path = bf.join(dir_path, filename)
        if bf.exists(path):
            return path
    return None


def log_loss_dict(diffusion, ts, losses):
    return
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

