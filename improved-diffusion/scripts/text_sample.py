import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
IMPROVED_DIFFUSION_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
TRANSFORMERS_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, "transformers/src"))

for p in [TRANSFORMERS_SRC, IMPROVED_DIFFUSION_DIR]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from mytokenizers import SimpleSmilesTokenizer
import argparse
import json
from rdkit import Chem
import numpy as np
import torch as th
import torch.distributed as dist

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
from improved_diffusion.rounding import rounding_func, load_models, load_tokenizer
from improved_diffusion import gaussian_diffusion as gd
from improved_diffusion.respace import SpacedDiffusion, space_timesteps
import numpy as np
from improved_diffusion import dist_util, logger
from improved_diffusion.transformer_model2 import TransformerNetModel2
from improved_diffusion.test_util import get_weights, denoised_fn_round

from improved_diffusion import dist_util, logger
from functools import partial
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    add_dict_to_argparser,
    args_to_dict,
)
from mydatasets import get_dataloader,ChEBIdataset

def main():
    args = create_argparser().parse_args()
    seed = int(getattr(args, 'seed', 121))
    set_seed(seed)
    logger.log(f"Using random seed: {seed}")

    # dist_util.setup_dist()
    logger.configure()
    args.sigma_small = True

    # args.diffusion_steps = 200 #500  # DEBUG

    if args.experiment == 'random1': args.experiment = 'random'
    logger.log("creating model and diffusion...")
    from mytokenizers import regexTokenizer
    # Vocab resolution is opt-in via --vocab_path only. It must NOT be
    # inferred from --data_dir: a checkpoint's training data_dir and its
    # vocab file are not reliably the same path (confirmed by a regression --
    # inferring it from data_dir silently swapped in the wrong vocab for a
    # checkpoint that was working fine before). Default is the exact
    # original hardcoded behavior; only override when explicitly asked to,
    # e.g. for a checkpoint trained on a different vocab (retro).
    vocab_path = getattr(args, 'vocab_path', '') or '../../datasets/SMILES/generate_vocab.txt'
    tokenizer = regexTokenizer(path=vocab_path, max_len=getattr(args, 'token_max_length', 256))
    model = TransformerNetModel2(
        in_channels=32,  # 3, DEBUG**
        # deep_channels = 10,
        model_channels=128,
        dropout=0.1,
        use_checkpoint=False,
        config_name='bert-base-uncased',
        training_mode='e2e',
        vocab_size=len(tokenizer),
        experiment_mode='lm',
        logits_mode=1,
        hidden_size = 1024,
        num_attention_heads=16,
        num_hidden_layers = 12,
        learned_mean_embed=getattr(args, 'learned_mean_embed', False),
    )
    # Determine timesteps to sample over
    if args.timestep_respacing and args.timestep_respacing.isdigit():
        step_stride = int(args.timestep_respacing)
    else:
        step_stride = 10
    use_timesteps = [i for i in range(0, 2000, step_stride)]

    # Same robust pattern as dpm_solver_diffusion/token_adaptive_diffusion
    # below: derive adaptive_noising from whether a real schedule path was
    # given, rather than requiring a SEPARATE --adaptive_noising flag to be
    # kept in sync with it. The old decoupled-flag version left this object
    # built with adaptive_noising=False (1-D alphas_cumprod) whenever
    # --adaptive_schedule_path was passed without ALSO explicitly passing
    # --adaptive_noising true -- and load_adaptive_schedule() below runs
    # unconditionally whenever a path is given, so it would then crash
    # trying to write 2-D-style into a still-1-D array. This also used to
    # crash needlessly even when `diffusion` was never going to be used for
    # sampling at all (e.g. --step_matrix_path or --use_dpm_solver runs).
    use_adaptive_noise_plain = bool(
        getattr(args, 'adaptive_schedule_path', '')
        and os.path.exists(args.adaptive_schedule_path)
    )

    diffusion = SpacedDiffusion(
        use_timesteps=use_timesteps,
        betas=gd.get_named_beta_schedule('sqrt', 2000),
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=gd.ModelVarType.FIXED_LARGE,
        loss_type=gd.LossType.E2E_MSE,
        rescale_timesteps=True,
        model_arch='transformer',
        training_mode='e2e',
        reg_rate=getattr(args, 'reg_rate', 0.0),
        denoise=getattr(args, 'denoise', False),
        denoise_rate=getattr(args, 'denoise_rate', 0.2),
        adaptive_noising=use_adaptive_noise_plain,
        token_max_length=tokenizer.max_len if use_adaptive_noise_plain else None,
        pad_tok_id=tokenizer.toktoid['[PAD]'] if use_adaptive_noise_plain else None,
        save_dir=getattr(args, 'out_dir', './generation_outputs'),
    )

    if use_adaptive_noise_plain:
        logger.log(f"Loading adaptive schedule from {args.adaptive_schedule_path}...")
        diffusion.load_adaptive_schedule(args.adaptive_schedule_path)
    elif getattr(args, 'adaptive_schedule_path', ''):
        logger.log(f"WARNING: --adaptive_schedule_path={args.adaptive_schedule_path} "
                   "does not exist -- falling back to the plain (non-adaptive) noise schedule.")

    # DPM-Solver++ picks its own log-SNR-spaced steps from the *full* noise
    # schedule, so it should not ride on top of `diffusion`'s uniform-stride
    # SpacedDiffusion subsampling (`--timestep_respacing`). Build a plain,
    # un-respaced GaussianDiffusion for it instead; `--dpm_solver_steps` plays
    # the role `--timestep_respacing` plays for the other two samplers.
    #
    # Same adaptive-noising requirement as `diffusion` above and
    # `token_adaptive_diffusion` below: if this checkpoint was trained
    # against a per-position adaptive noise schedule, sampling from it with
    # the plain 1-D schedule silently diverges from training. adaptive_noising
    # must be passed at construction time (load_adaptive_schedule() only
    # overwrites an existing (T, S) array in place, it does not expand a
    # 1-D one), so this checks --adaptive_schedule_path up front rather than
    # loading it after the fact.
    dpm_solver_diffusion = None
    if getattr(args, 'use_dpm_solver', False):
        use_adaptive_noise_dpm = bool(
            getattr(args, 'adaptive_schedule_path', '')
            and os.path.exists(args.adaptive_schedule_path)
        )
        dpm_solver_diffusion = gd.GaussianDiffusion(
            betas=gd.get_named_beta_schedule('sqrt', 2000),
            model_mean_type=gd.ModelMeanType.START_X,
            model_var_type=gd.ModelVarType.FIXED_LARGE,
            loss_type=gd.LossType.E2E_MSE,
            rescale_timesteps=True,
            model_arch='transformer',
            training_mode='e2e',
            reg_rate=getattr(args, 'reg_rate', 0.0),
            denoise=getattr(args, 'denoise', False),
            denoise_rate=getattr(args, 'denoise_rate', 0.2),
            adaptive_noising=use_adaptive_noise_dpm,
            token_max_length=tokenizer.max_len if use_adaptive_noise_dpm else None,
            pad_tok_id=tokenizer.toktoid['[PAD]'] if use_adaptive_noise_dpm else None,
        )
        if use_adaptive_noise_dpm:
            logger.log(
                f"### Loading adaptive noise schedule from {args.adaptive_schedule_path} "
                "for DPM-Solver++..."
            )
            dpm_solver_diffusion.load_adaptive_schedule(args.adaptive_schedule_path)
        elif getattr(args, 'adaptive_schedule_path', ''):
            logger.log(
                f"### WARNING: --adaptive_schedule_path={args.adaptive_schedule_path} "
                "does not exist -- DPM-Solver++ is falling back to the plain "
                "(non-adaptive) noise schedule."
            )

    # ============================================================
    # Optional token-wise adaptive step schedule (per-position reduced-step
    # inference). Built offline by reduced_step_profile.py (Stage 1) +
    # trajectory_analysis.py (Stages 2-4); see ADAPTIVE_STEP_INFERENCE.md.
    # Mutually exclusive with --use_dpm_solver: both are "spend K model
    # calls instead of T" strategies, this one just lets different sequence
    # positions spend those K calls at different timesteps.
    # ============================================================
    step_matrix = None
    if getattr(args, 'step_matrix_path', ''):
        if getattr(args, 'use_dpm_solver', False):
            raise ValueError(
                "--step_matrix_path and --use_dpm_solver are mutually "
                "exclusive samplers; pick one."
            )
        from improved_diffusion.dpm_solver import load_step_matrix
        logger.log(f"### Loading token-wise step matrix from {args.step_matrix_path}")
        step_matrix = load_step_matrix(
            args.step_matrix_path,
            K=args.token_adaptive_steps,
            seq_len=tokenizer.max_len,
        )
        logger.log(f"### Step matrix loaded: shape={step_matrix.shape}")

    # Token-adaptive DPM-Solver, like `dpm_solver_diffusion` above, needs the
    # full un-respaced base schedule (step_matrix indexes raw timesteps
    # 0..diffusion_steps-1), not SpacedDiffusion's uniform-stride subsample.
    #
    # If an adaptive-noising schedule is going to be loaded below, this must
    # be constructed with adaptive_noising=True (+ token_max_length/pad_tok_id)
    # up front so its schedule arrays are already (T, S) -- load_adaptive_
    # schedule()/update_time_discretized_parameters() only overwrite an
    # existing 2-D array in place, they do not expand a 1-D one.
    token_adaptive_diffusion = None
    if step_matrix is not None:
        use_adaptive_noise = bool(
            getattr(args, 'adaptive_schedule_path', '')
            and os.path.exists(args.adaptive_schedule_path)
        )
        token_adaptive_diffusion = gd.GaussianDiffusion(
            betas=gd.get_named_beta_schedule('sqrt', 2000),
            model_mean_type=gd.ModelMeanType.START_X,
            model_var_type=gd.ModelVarType.FIXED_LARGE,
            loss_type=gd.LossType.E2E_MSE,
            rescale_timesteps=True,
            model_arch='transformer',
            training_mode='e2e',
            reg_rate=getattr(args, 'reg_rate', 0.0),
            denoise=getattr(args, 'denoise', False),
            denoise_rate=getattr(args, 'denoise_rate', 0.2),
            adaptive_noising=use_adaptive_noise,
            token_max_length=tokenizer.max_len if use_adaptive_noise else None,
            pad_tok_id=tokenizer.toktoid['[PAD]'] if use_adaptive_noise else None,
        )
        if use_adaptive_noise:
            logger.log(
                f"### Loading adaptive noise schedule from {args.adaptive_schedule_path} "
                "for token-adaptive DPM-Solver..."
            )
            token_adaptive_diffusion.load_adaptive_schedule(args.adaptive_schedule_path)

    print(args.model_path)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'the parameter count is {pytorch_total_params}')

    print(diffusion.rescale_timesteps, 'a marker for whether we are in the debug mode')
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    print(args.num_samples)
    print('--'*30)
    print('loading {} set'.format(args.split))
    print('--'*30)

    dataset_dir = getattr(args, 'data_dir', '') if getattr(args, 'data_dir', '') and os.path.exists(getattr(args, 'data_dir', '')) else (
        '/home/ee/phd/eez248435/tgm-dlm/datasets/SMILES/' if os.path.exists('/home/ee/phd/eez248435/tgm-dlm/datasets/SMILES/') else '../../datasets/SMILES/'
    )
    train_dataset = ChEBIdataset(
        dir=dataset_dir,
        smi_tokenizer=tokenizer,
        split=args.split,
        replace_desc=False
    )
    print('DATASETINFO-----------------------------')
    print(len(train_dataset),(train_dataset[0]['desc_state'].shape))
    
    start_idx = getattr(args, 'start_idx', 0)
    end_idx = getattr(args, 'end_idx', -1)
    if end_idx == -1 or end_idx > len(train_dataset):
        end_idx = min(start_idx + args.num_samples, len(train_dataset)) if args.num_samples > 0 else len(train_dataset)
    
    desc = [(train_dataset[i]['desc_state'],train_dataset[i]['desc_mask'],train_dataset[i]['smiles']) for i in range(start_idx, end_idx)]
    num_to_sample = len(desc)
    answer = [i[2] for i in desc]
    print(f"Sampling slice [{start_idx} : {end_idx}] (total {num_to_sample} samples)...")
    
    model3 = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model3.requires_grad = False

    allsample = []
    num_done = 0
    while num_done < num_to_sample:
        idend = min(num_done+args.batch_size, num_to_sample)
        print('acquiring  {} : {}'.format(num_done,idend))
        desc_state = th.concat([i[0] for i in desc[num_done:idend]],dim=0)
        desc_mask = th.concat([i[1] for i in desc[num_done:idend]],dim=0)
        
        model_kwargs = {}
        sample_shape = (idend-num_done, tokenizer.max_len, model.in_channels)
        print(sample_shape)
        if getattr(args, 'use_dpm_solver', False):
            print(f'sampling with DPM-Solver++ (steps={args.dpm_solver_steps}, order={args.dpm_solver_order})')
            sample = dpm_solver_diffusion.dpm_solver_sample_loop(
                model,
                sample_shape,
                clip_denoised=args.clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
                progress=True,
                desc=(desc_state, desc_mask),
                steps=args.dpm_solver_steps,
                order=args.dpm_solver_order,
                method=args.dpm_solver_method,
            )
        elif step_matrix is not None:
            print(f'sampling with token-adaptive DPM-Solver++ (K={step_matrix.shape[0]}, order={args.dpm_solver_order})')
            sample = token_adaptive_diffusion.token_adaptive_dpm_solver_sample_loop(
                model,
                sample_shape,
                step_matrix=step_matrix,
                clip_denoised=args.clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
                progress=True,
                desc=(desc_state, desc_mask),
                order=args.dpm_solver_order,
            )
        else:
            print('use_ddim:{}',args.use_ddim)
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                model,
                sample_shape,
                clip_denoised=args.clip_denoised,
                denoised_fn = None,
                model_kwargs=model_kwargs,
                top_p =args.top_p,
                progress = True,
                desc = (desc_state,desc_mask)
            )
        allsample.append(sample)
        num_done = idend
    sample = th.concat(allsample,dim=0)
    print('decoding for e2e', )
    print(sample.shape)
    x_t = th.tensor(sample).cuda()
    reshaped_x_t = x_t
    logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
    cands = th.topk(logits, k=1, dim=-1)
    sample = cands.indices
    sample = sample.squeeze(-1)
    print(sample)
    # NOTE: previously re-imported and re-instantiated `regexTokenizer()`
    # here with no `path=`, which silently shadowed the correctly-configured
    # `tokenizer` from earlier in this function (built with --vocab_path)
    # and decoded through the wrong vocab file whenever it differed from
    # the hardcoded default -- this was invisible as long as every run used
    # the default (forward/ChEBI) checkpoint, since both vocabs happened to
    # agree. Reuse the already-correct `tokenizer` instead of rebuilding it.
    c = tokenizer.decode(sample)
    if os.path.dirname(args.outputdir):
        os.makedirs(os.path.dirname(args.outputdir), exist_ok=True)
    with open(args.outputdir,'w') as f:
        for i,x in enumerate(c):
            if i==0:
                print(x)
            f.write(x.replace('[PAD]','')+'   ||   '+answer[i]+'\n')


    with open(args.outputdir) as f:
        allsmiles = [k.strip().split('||')[0].strip().replace('[EOS]','').replace('[SOS]','') for k in f.readlines()]
    temp_bad_mols_path = os.path.join(os.path.dirname(args.outputdir) if os.path.dirname(args.outputdir) else '.', 'tempbadmols.txt')
    f = open(temp_bad_mols_path, 'w')
    for cnt,s in enumerate(allsmiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            f.write(str(cnt)+'\t'+s+'\n')
    f.close()

def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=50,#10000,
        batch_size=64,
        use_ddim=False,
        mbr_sample=1,
        seed=121,
        model_path="",
        model_arch='conv-unet',
        verbose='yes',
        out_dir="diffusion_lm/improved_diffusion/out_gen"
    )
    text_defaults = dict(modality='text',
                         dataset_name='wikitext',
                         dataset_config_name='wikitext-2-raw-v1',
                         model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None',
                         experiment='gpt2_pre_compress', model_arch='trans-unet',
                         preprocessing_num_workers=1,
                         emb_scale_factor=1.0, clamp='clamp',split = 'test',
                         model_path='../../checkpoints/PLAIN_ema_0.9999_200000.pt',
                         use_ddim=False,
                         batch_size =64,num_samples=3300,top_p =1.0,out_dir='generation_outputs',
                         outputdir='../../textguidtry_256_final.txt',
                         learned_mean_embed=False,
                         denoise=False,
                         denoise_rate=0.2,
                         reg_rate=0.0,
                         adaptive_noising=False,
                         adaptive_schedule_path='',
                         timestep_respacing='10',
                         data_dir='',
                         start_idx=0,
                         end_idx=-1,
                         token_max_length=256,
                         use_dpm_solver=False,
                         dpm_solver_steps=10,
                         dpm_solver_order=2,
                         dpm_solver_method='multistep',
                         step_matrix_path='',
                         token_adaptive_steps=10,
                         vocab_path='',
                         )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()