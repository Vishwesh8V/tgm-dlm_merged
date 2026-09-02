#!/usr/bin/env python3
###
#RUN WITH:
# python pcdes_zero_shot.py \
#   --model_path /home/ee/phd/eez248435/tgm-dlm_merged/checkpoints/PLAIN_ema_0.9999_130000.pt \
#   --adaptive_schedule_path /home/ee/phd/eez248435/tgm-dlm_merged/checkpoints/adaptive_schedule/alpha_cumprod_step_120000.npy \
#   --batch_size 10
###
import os
import sys
import argparse
import random
import csv

import numpy as np
import torch as th

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = "/home/ee/phd/eez248435/tgm-dlm_merged"

SCRIPTS_DIR = os.path.join(
    PROJECT_ROOT,
    "improved-diffusion",
    "scripts"
)

IMPROVED_DIFFUSION_DIR = os.path.join(
    PROJECT_ROOT,
    "improved-diffusion"
)

TRANSFORMERS_SRC = os.path.join(
    PROJECT_ROOT,
    "transformers",
    "src"
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "SMILES"
)

PCDES_DESC = os.path.join(
    DATA_DIR,
    "align_des_filt3.txt"
)

PCDES_SMILES = os.path.join(
    DATA_DIR,
    "align_smiles.txt"
)

SCIBERT_DIR = os.path.join(
    PROJECT_ROOT,
    "scibert"
)

DEFAULT_ADAPTIVE_SCHEDULE = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "adaptive_schedule",
    "alpha_cumprod_step_120000.npy"
)

# ============================================================
# PYTHON PATH
# ============================================================

sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, IMPROVED_DIFFUSION_DIR)
sys.path.insert(0, TRANSFORMERS_SRC)

# ============================================================
# IMPORTS FROM YOUR REPO
# ============================================================

from transformers import (
    AutoTokenizer,
    AutoModel,
    set_seed,
)

from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from rdkit.Chem import (
    MACCSkeys,
    RDKFingerprint,
    AllChem,
)

import improved_diffusion.dist_util as dist_util
import improved_diffusion.gaussian_diffusion as gd

from improved_diffusion.respace import SpacedDiffusion

from improved_diffusion.transformer_model2 import (
    TransformerNetModel2
)

from mytokenizers import regexTokenizer


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

SEED = 121

PCDES_TEST_START = 12900

# FIRST SANITY RUN ONLY
NUM_SAMPLES = 100

PCDES_TEST_END = (
    PCDES_TEST_START + NUM_SAMPLES
)

# Same configuration as your sample.sh
LEARNED_MEAN_EMBED = True
DENOISE = True
DENOISE_RATE = 0.2
REG_RATE = 0.1

ADAPTIVE_NOISING = True

TIMESTEP_RESPACING = 1

TOKEN_MAX_LENGTH = 256

BATCH_SIZE = 10

# ============================================================
# DATA LOADING
# ============================================================

def load_pcdes():

    print("=" * 90)
    print("LOADING PCDes")
    print("=" * 90)

    # --------------------------------------------------------
    # Descriptions
    # --------------------------------------------------------

    with open(
        PCDES_DESC,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as f:

        descriptions = [
            line.rstrip("\n")
            for line in f
        ]

    # --------------------------------------------------------
    # SMILES
    # --------------------------------------------------------

    with open(
        PCDES_SMILES,
        "r",
        encoding="utf-8"
    ) as f:

        smiles = [
            line.strip()
            for line in f
        ]

    print(
        f"Description lines : {len(descriptions)}"
    )

    print(
        f"SMILES             : {len(smiles)}"
    )

    # --------------------------------------------------------
    # PCDes has 15,000 molecule-level SMILES.
    #
    # The test portion is:
    #
    #   12000:15000
    #
    # For this sanity run:
    #
    #   12000:12050
    # --------------------------------------------------------

    if len(smiles) != 15000:

        raise RuntimeError(
            f"Expected 15000 SMILES, "
            f"got {len(smiles)}"
        )

    if len(descriptions) < 15000:

        raise RuntimeError(
            "Not enough aligned descriptions."
        )

    descriptions = descriptions[
        PCDES_TEST_START:
        PCDES_TEST_END
    ]

    smiles = smiles[
        PCDES_TEST_START:
        PCDES_TEST_END
    ]

    if len(descriptions) != NUM_SAMPLES:

        raise RuntimeError(
            f"Expected {NUM_SAMPLES} descriptions, "
            f"got {len(descriptions)}"
        )

    if len(smiles) != NUM_SAMPLES:

        raise RuntimeError(
            f"Expected {NUM_SAMPLES} SMILES, "
            f"got {len(smiles)}"
        )

    print(
        f"\nPCDes test indices:"
        f" {PCDES_TEST_START}:{PCDES_TEST_END}"
    )

    print(
        f"Number of samples: {len(smiles)}"
    )

    print("\nExample:")
    print("-" * 90)

    print("DESCRIPTION:")
    print(descriptions[0])

    print("\nGROUND TRUTH SMILES:")
    print(smiles[0])

    print("-" * 90)

    return descriptions, smiles


# ============================================================
# SciBERT DESCRIPTION STATES
# ============================================================

def create_desc_states(
    descriptions,
    device
):

    print()
    print("=" * 90)
    print("CREATING SciBERT DESCRIPTION STATES")
    print("=" * 90)

    print(
        f"SciBERT: {SCIBERT_DIR}"
    )

    print(
        "max_length = 216"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        SCIBERT_DIR
    )

    model = AutoModel.from_pretrained(
        SCIBERT_DIR
    )

    model.to(device)

    model.eval()

    states = []
    masks = []

    with th.no_grad():

        for i, description in enumerate(
            descriptions
        ):

            if i % 10 == 0:

                print(
                    f"Encoding "
                    f"{i + 1}/{len(descriptions)}"
                )

            tok = tokenizer(
                description,
                max_length=216,
                truncation=True,
                padding="max_length"
            )

            input_ids = th.tensor(
                tok["input_ids"],
                dtype=th.long
            ).unsqueeze(0)

            attention_mask = th.tensor(
                tok["attention_mask"],
                dtype=th.long
            ).unsqueeze(0)

            output = model(
                input_ids.to(device)
            )

            # Same representation used by process_text.py
            state = (
                output
                .last_hidden_state
                .cpu()
            )

            states.append(state)

            masks.append(
                attention_mask
            )

    print(
        "\nSciBERT states created."
    )

    print(
        f"State shape: "
        f"{tuple(states[0].shape)}"
    )

    print(
        f"Mask shape: "
        f"{tuple(masks[0].shape)}"
    )

    return states, masks


# ============================================================
# BUILD MODEL + DIFFUSION
# ============================================================

def build_model(
    model_path,
    adaptive_schedule_path,
    device
):

    print()
    print("=" * 90)
    print("CREATING TGM-DLM MODEL")
    print("=" * 90)

    print(
        "Configuration:"
    )

    print(
        f"  learned_mean_embed = "
        f"{LEARNED_MEAN_EMBED}"
    )

    print(
        f"  denoise            = "
        f"{DENOISE}"
    )

    print(
        f"  denoise_rate       = "
        f"{DENOISE_RATE}"
    )

    print(
        f"  reg_rate           = "
        f"{REG_RATE}"
    )

    print(
        f"  adaptive_noising   = "
        f"{ADAPTIVE_NOISING}"
    )

    print(
        f"  timestep_respacing = "
        f"{TIMESTEP_RESPACING}"
    )

    print(
        f"  adaptive schedule  = "
        f"{adaptive_schedule_path}"
    )

    # --------------------------------------------------------
    # SMILES TOKENIZER
    # --------------------------------------------------------

    vocab_path = os.path.join(
        DATA_DIR,
        "generate_vocab.txt"
    )

    print(
        f"\nUsing SMILES vocabulary:"
        f"\n{vocab_path}"
    )

    tokenizer = regexTokenizer(
        path=vocab_path,
        max_len=TOKEN_MAX_LENGTH
    )

    print(
        f"SMILES tokenizer max length: "
        f"{tokenizer.max_len}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print(
        "\nCreating TransformerNetModel2..."
    )

    model = TransformerNetModel2(
        in_channels=32,
        model_channels=128,
        dropout=0.1,
        use_checkpoint=False,
        config_name="bert-base-uncased",
        training_mode="e2e",
        vocab_size=len(tokenizer),
        experiment_mode="lm",
        logits_mode=1,
        hidden_size=1024,
        num_attention_heads=16,
        num_hidden_layers=12,
        learned_mean_embed=LEARNED_MEAN_EMBED,
    )

    # --------------------------------------------------------
    # DIFFUSION
    # --------------------------------------------------------

    print(
        "\nCreating diffusion..."
    )

    # sample.sh uses timestep_respacing=1
    step_stride = int(
        TIMESTEP_RESPACING
    )

    use_timesteps = list(
        range(
            0,
            2000,
            step_stride
        )
    )

    diffusion = SpacedDiffusion(
        use_timesteps=use_timesteps,

        betas=gd.get_named_beta_schedule(
            "sqrt",
            2000
        ),

        model_mean_type=gd.ModelMeanType.START_X,

        model_var_type=gd.ModelVarType.FIXED_LARGE,

        loss_type=gd.LossType.E2E_MSE,

        rescale_timesteps=True,

        model_arch="transformer",

        training_mode="e2e",

        reg_rate=REG_RATE,

        denoise=DENOISE,

        denoise_rate=DENOISE_RATE,

        adaptive_noising=ADAPTIVE_NOISING,

        token_max_length=TOKEN_MAX_LENGTH,

        pad_tok_id=tokenizer.toktoid["[PAD]"],

        save_dir=os.path.join(
            PROJECT_ROOT,
            "generation_outputs"
        ),
    )

    # --------------------------------------------------------
    # ADAPTIVE SCHEDULE
    # --------------------------------------------------------

    if ADAPTIVE_NOISING:

        if not os.path.isfile(
            adaptive_schedule_path
        ):

            raise FileNotFoundError(
                "Adaptive schedule not found:\n"
                + adaptive_schedule_path
            )

        print(
            "\nLoading adaptive schedule:"
        )

        print(
            adaptive_schedule_path
        )

        # Same mechanism used by your sampler.
        diffusion.load_adaptive_schedule(
            adaptive_schedule_path
        )

        print(
            "Adaptive schedule loaded."
        )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    print(
        "\nLoading checkpoint:"
    )

    print(
        model_path
    )

    state_dict = (
        dist_util.load_state_dict(
            model_path,
            map_location="cpu"
        )
    )

    model.load_state_dict(
        state_dict
    )

    model.to(device)

    model.eval()

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\nParameter count: "
        f"{total_params:,}"
    )

    print(
        "Checkpoint loaded successfully."
    )

    return (
        model,
        diffusion,
        tokenizer
    )


# ============================================================
# GENERATION
# ============================================================

def generate_samples(
    model,
    diffusion,
    tokenizer,
    states,
    masks,
    references,
    device,
    batch_size
):

    print()
    print("=" * 90)
    print("GENERATING PCDes ZERO-SHOT SAMPLES")
    print("=" * 90)

    generated = []

    num_samples = len(
        references
    )

    # --------------------------------------------------------
    # Important:
    #
    # We process batches but preserve PCDes ordering.
    # --------------------------------------------------------

    for start in range(
        0,
        num_samples,
        batch_size
    ):

        end = min(
            start + batch_size,
            num_samples
        )

        current_bs = end - start

        print()
        print(
            f"Generating "
            f"{start + 1}-{end} / "
            f"{num_samples}"
        )

        # ----------------------------------------------------
        # [B, 1, 216, 768]
        # -> [B, 216, 768]
        #
        # The original process_text states are saved with
        # a batch dimension.
        # ----------------------------------------------------

        desc_state = th.cat(
            states[start:end],
            dim=0
        ).to(device)

        desc_mask = th.cat(
            masks[start:end],
            dim=0
        ).to(device)

        # ----------------------------------------------------
        # TGM-DLM latent shape
        # ----------------------------------------------------

        sample_shape = (
            current_bs,
            tokenizer.max_len,
            model.in_channels
        )

        print(
            f"Sample shape: "
            f"{sample_shape}"
        )

        # ----------------------------------------------------
        # Sampling
        #
        # Same sampler selection as text_sample.py.
        # ----------------------------------------------------

        sample_fn = (
            diffusion.ddim_sample_loop
            if False
            else diffusion.p_sample_loop
        )

        with th.no_grad():

            sample = sample_fn(
                model,
                sample_shape,

                clip_denoised=False,

                denoised_fn=None,

                model_kwargs={},

                top_p=1.0,

                progress=True,

                desc=(
                    desc_state,
                    desc_mask
                )
            )

        # ----------------------------------------------------
        # Decode
        # ----------------------------------------------------

        if isinstance(
            sample,
            th.Tensor
        ):

            x_t = sample

        else:

            x_t = th.tensor(
                sample
            )

        x_t = x_t.to(device)

        logits = model.get_logits(
            x_t
        )

        candidates = th.topk(
            logits,
            k=1,
            dim=-1
        )

        sample_tokens = (
            candidates.indices
            .squeeze(-1)
        )

        decoded = tokenizer.decode(
            sample_tokens
        )

        if isinstance(
            decoded,
            str
        ):

            decoded = [decoded]

        generated.extend(
            decoded
        )

    return generated


# ============================================================
# SMILES CLEANING
# ============================================================

def clean_smiles(smiles):

    if smiles is None:

        return ""

    smiles = str(
        smiles
    )

    # Remove special tokens if decoder retained them.

    for token in [
        "[PAD]",
        "[SOS]",
        "[EOS]"
    ]:

        smiles = smiles.replace(
            token,
            ""
        )

    return smiles.strip()


# ============================================================
# CANONICAL SMILES
# ============================================================

def canonicalize(smiles):

    try:

        mol = Chem.MolFromSmiles(
            smiles
        )

        if mol is None:

            return None

        return Chem.MolToSmiles(
            mol,
            canonical=True
        )

    except Exception:

        return None


# ============================================================
# TABLE 5 METRICS
# ============================================================

def calculate_metrics(
    generated,
    references
):

    print()
    print("=" * 90)
    print("CALCULATING METRICS")
    print("=" * 90)

    maccs_scores = []
    rdk_scores = []
    morgan_scores = []

    valid_count = 0
    exact_count = 0

    rows = []

    for i, (
        generated_smiles,
        reference_smiles
    ) in enumerate(
        zip(
            generated,
            references
        )
    ):

        generated_smiles = (
            clean_smiles(
                generated_smiles
            )
        )

        reference_smiles = (
            clean_smiles(
                reference_smiles
            )
        )

        gen_mol = Chem.MolFromSmiles(
            generated_smiles
        )

        ref_mol = Chem.MolFromSmiles(
            reference_smiles
        )

        valid = (
            gen_mol is not None
        )
        reference_valid = (
            ref_mol is not None
        )
        if valid and reference_valid:

            valid_count += 1

            # ------------------------------------------------
            # MACCS
            # ------------------------------------------------

            gen_fp = (
                MACCSkeys.GenMACCSKeys(
                    gen_mol
                )
            )

            ref_fp = (
                MACCSkeys.GenMACCSKeys(
                    ref_mol
                )
            )

            maccs = (
                DataStructs.TanimotoSimilarity(
                    gen_fp,
                    ref_fp
                )
            )

            maccs_scores.append(
                maccs
            )

            # ------------------------------------------------
            # RDK
            # ------------------------------------------------

            gen_fp = (
                RDKFingerprint(
                    gen_mol
                )
            )

            ref_fp = (
                RDKFingerprint(
                    ref_mol
                )
            )

            rdk = (
                DataStructs.TanimotoSimilarity(
                    gen_fp,
                    ref_fp
                )
            )

            rdk_scores.append(
                rdk
            )

            # ------------------------------------------------
            # Morgan
            # ------------------------------------------------

            gen_fp = (
                AllChem
                .GetMorganFingerprintAsBitVect(
                    gen_mol,
                    radius=2,
                    nBits=2048
                )
            )

            ref_fp = (
                AllChem
                .GetMorganFingerprintAsBitVect(
                    ref_mol,
                    radius=2,
                    nBits=2048
                )
            )

            morgan = (
                DataStructs.TanimotoSimilarity(
                    gen_fp,
                    ref_fp
                )
            )

            morgan_scores.append(
                morgan
            )

        else:

            maccs = np.nan
            rdk = np.nan
            morgan = np.nan

        # ----------------------------------------------------
        # Exact canonical SMILES match
        # ----------------------------------------------------

        gen_can = canonicalize(
            generated_smiles
        )

        ref_can = canonicalize(
            reference_smiles
        )

        exact = (
            gen_can is not None
            and ref_can is not None
            and gen_can == ref_can
        )

        if exact:

            exact_count += 1

        rows.append({
            "pcdes_index":
                PCDES_TEST_START + i,

            "generated_smiles":
                generated_smiles,

            "reference_smiles":
                reference_smiles,

            "valid":
                int(valid),

            "exact":
                int(exact),

            "maccs":
                maccs,

            "rdk":
                rdk,

            "morgan":
                morgan,
        })

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    metrics = {

        "MACCS":
            (
                float(
                    np.mean(
                        maccs_scores
                    )
                )
                if len(maccs_scores) > 0
                else 0.0
            ),

        "RDK":
            (
                float(
                    np.mean(
                        rdk_scores
                    )
                )
                if len(rdk_scores) > 0
                else 0.0
            ),

        "Morgan":
            (
                float(
                    np.mean(
                        morgan_scores
                    )
                )
                if len(morgan_scores) > 0
                else 0.0
            ),

        "Exact":
            exact_count / len(
                references
            ),

        "Valid":
            valid_count / len(
                references
            ),
    }

    return metrics, rows


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    generated,
    references,
    rows
):

    output_dir = os.path.join(
        PROJECT_ROOT,
        "generation_outputs",
        "pcdes_zero_shot"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # Filenames are based on the actual experiment size.
    generation_file = os.path.join(
        output_dir,
        f"pcdes_zero_shot_{PCDES_TEST_START}_{NUM_SAMPLES}_seed{SEED}.txt"
    )

    csv_file = os.path.join(
        output_dir,
        f"pcdes_zero_shot_{PCDES_TEST_START}_{NUM_SAMPLES}_seed{SEED}.csv"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Save BOTH files BEFORE metric calculation.
    #
    # This function is called immediately after inference and
    # before calculate_metrics(). If metric calculation crashes,
    # the expensive generated molecules are still safely saved.
    # --------------------------------------------------------

    with open(
        generation_file,
        "w",
        encoding="utf-8"
    ) as f:

        for gen, ref in zip(
            generated,
            references
        ):

            f.write(
                clean_smiles(gen)
                + " || "
                + ref
                + "\n"
            )

        f.flush()
        os.fsync(f.fileno())

    # Save a raw CSV immediately. Metric columns are filled with
    # blank values until metric calculation completes.
    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pcdes_index",
                "generated_smiles",
                "reference_smiles",
                "valid",
                "exact",
                "maccs",
                "rdk",
                "morgan",
            ]
        )

        writer.writeheader()

        for i, (gen, ref) in enumerate(
            zip(generated, references)
        ):

            writer.writerow({
                "pcdes_index":
                    PCDES_TEST_START + i,
                "generated_smiles":
                    clean_smiles(gen),
                "reference_smiles":
                    clean_smiles(ref),
                "valid": "",
                "exact": "",
                "maccs": "",
                "rdk": "",
                "morgan": "",
            })

        f.flush()
        os.fsync(f.fileno())

    print()
    print("=" * 90)
    print("GENERATIONS SAVED BEFORE METRIC EVALUATION")
    print("=" * 90)

    print(
        f"Generation file:\n{generation_file}"
    )

    print(
        f"Raw CSV:\n{csv_file}"
    )

    print(
        "These files are now safe even if metric calculation fails."
    )

    return generation_file, csv_file

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pcdes_index",
                "generated_smiles",
                "reference_smiles",
                "valid",
                "exact",
                "maccs",
                "rdk",
                "morgan",
            ]
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print()
    print(
        f"Generation file:\n"
        f"{generation_file}"
    )

    print(
        f"Detailed CSV:\n"
        f"{csv_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        required=True
    )

    parser.add_argument(
        "--adaptive_schedule_path",
        default=DEFAULT_ADAPTIVE_SCHEDULE
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(
        SEED
    )

    random.seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    th.manual_seed(
        SEED
    )

    if th.cuda.is_available():

        th.cuda.manual_seed_all(
            SEED
        )

    print(
        "=" * 90
    )

    print(
        "TGM-DLM PCDes ZERO-SHOT EVALUATION"
    )

    print(
        "=" * 90
    )

    print(
        f"Seed                : {SEED}"
    )

    print(
        f"Samples             : {NUM_SAMPLES}"
    )

    print(
        f"PCDes indices       : "
        f"{PCDES_TEST_START}:{PCDES_TEST_END}"
    )

    print(
        f"Learned mean embed  : "
        f"{LEARNED_MEAN_EMBED}"
    )

    print(
        f"Denoise             : "
        f"{DENOISE}"
    )

    print(
        f"Denoise rate        : "
        f"{DENOISE_RATE}"
    )

    print(
        f"Regularization      : "
        f"{REG_RATE}"
    )

    print(
        f"Adaptive noising    : "
        f"{ADAPTIVE_NOISING}"
    )

    print(
        f"Timestep respacing  : "
        f"{TIMESTEP_RESPACING}"
    )

    print(
        f"Adaptive schedule   : "
        f"{args.adaptive_schedule_path}"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = dist_util.dev()

    print(
        f"Device              : "
        f"{device}"
    )

    # --------------------------------------------------------
    # Load PCDes
    # --------------------------------------------------------

    descriptions, references = (
        load_pcdes()
    )

    # --------------------------------------------------------
    # SciBERT
    # --------------------------------------------------------

    states, masks = (
        create_desc_states(
            descriptions,
            device
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, diffusion, tokenizer = (
        build_model(
            args.model_path,
            args.adaptive_schedule_path,
            device
        )
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    generated = (
        generate_samples(
            model,
            diffusion,
            tokenizer,
            states,
            masks,
            references,
            device,
            args.batch_size
        )
    )

    # Nothing below this point should be able to erase the
    # generated samples: they are saved immediately after this
    # length check and BEFORE metric calculation.
    if len(generated) != NUM_SAMPLES:

        raise RuntimeError(
            f"Expected {NUM_SAMPLES} "
            f"generated molecules, "
            f"got {len(generated)}"
        )

    # --------------------------------------------------------
    # CRITICAL SAFETY SAVE
    #
    # Save the complete inference result BEFORE doing ANY
    # metric calculation. This prevents a metric/evaluation
    # crash from wasting hours of inference.
    # --------------------------------------------------------

    generation_file, csv_file = save_results(
        generated,
        references,
        rows=[]
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics, rows = (
        calculate_metrics(
            generated,
            references
        )
    )

    # --------------------------------------------------------
    # Update the SAME CSV with metric results.
    #
    # No additional output files are created.
    # --------------------------------------------------------

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pcdes_index",
                "generated_smiles",
                "reference_smiles",
                "valid",
                "exact",
                "maccs",
                "rdk",
                "morgan",
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

        f.flush()
        os.fsync(f.fileno())

    print()
    print("=" * 90)
    print("METRICS COMPLETE — CSV UPDATED")
    print("=" * 90)
    print(f"Generation file: {generation_file}")
    print(f"Detailed CSV:    {csv_file}")

    # --------------------------------------------------------
    # Table 5-style output
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "TABLE 5 — ZERO-SHOT GENERALIZATION ON PCDes"
    )
    print("=" * 90)

    print(
        f"{'Model':<15}"
        f"{'MACCS ↑':>12}"
        f"{'RDK ↑':>12}"
        f"{'Morgan ↑':>12}"
        f"{'FCD ↓':>12}"
        f"{'Exact ↑':>12}"
        f"{'Valid ↑':>12}"
    )

    print("-" * 90)

    print(
        f"{'tgm-dlm':<15}"
        f"{metrics['MACCS']:>12.3f}"
        f"{metrics['RDK']:>12.3f}"
        f"{metrics['Morgan']:>12.3f}"
        f"{'N/A':>12}"
        f"{metrics['Exact']:>12.3f}"
        f"{metrics['Valid']:>12.3f}"
    )

    print("=" * 90)

    print()
    print(
        f"{NUM_SAMPLES}-sample zero-shot run complete."
    )

    print(
        "FCD is intentionally omitted from this "
        "sanity run; we should use the exact FCD "
        "implementation from your existing evaluation "
        "pipeline before reporting the final 3000-sample "
        "Table 5 result."
    )


if __name__ == "__main__":
    main()