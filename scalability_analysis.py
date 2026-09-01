#!/usr/bin/env python3

"""
One-seed scalability analysis for tgm-dlm.

Reproduces the structure of Table 4 from the UTGDiff paper:

                    MACCS FTS
                       |
          +------------+------------+
          |                         |
   Instruction Length           BertzCT
          |                         |
   1-64                      <100
   65-96                     100-300
   97-128                    300-800
   129-160                   >800
   >160

IMPORTANT:
    The paper gives the instruction-length bins but does not
    explicitly state in the Table 4 text whether "Length"
    means characters, words, or tokenizer tokens.

Therefore this script reports BOTH:

    1. whitespace word length
    2. RoBERTa tokenizer token length

This lets us inspect which definition is appropriate.

BertzCT is calculated on the ground-truth molecule.

Default seed:
    121

Usage:

    python scalability_analysis_1seed.py

or:

    python scalability_analysis_1seed.py \
        --seed_file generation_outputs/sampled_smiles_60k_500samp_seed121.txt

"""


import os
import re
import argparse

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import GraphDescriptors
from rdkit import DataStructs


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_TEST_FILE = (
    "datasets/SMILES/test.txt"
)

DEFAULT_SEED_FILE = (
    "/home/ee/phd/eez248435/tgm-dlm_merged/generation_outputs"
    "/sampled_smiles_130k_fullsamp_3008_seed121.txt"
)

DEFAULT_OUTPUT_DIR = (
    "scalability_outputs_1seed"
)


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data(filepath):

    records = []

    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(f):

            line = line.rstrip("\n")

            if not line.strip():
                continue

            parts = line.split("\t")

            if len(parts) < 3:
                print(
                    f"WARNING: Could not parse line "
                    f"{line_number}: {line[:150]}"
                )
                continue

            cid = parts[0].strip()
            smiles = parts[1].strip()
            description = "\t".join(
                parts[2:]
            ).strip()

            # Skip header
            if cid.lower() == "cid":
                continue

            records.append({
                "test_index": len(records),
                "file_line": line_number,
                "cid": cid,
                "ground_truth": smiles,
                "description": description,
            })

    print(
        f"\nLoaded {len(records)} test examples."
    )

    return records


# ============================================================
# LOAD GENERATION FILE
# ============================================================

def load_generation_file(filepath):

    generated = []

    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(f):

            line = line.strip()

            if not line:
                continue

            # Your generation format:
            #
            # generated_smiles || ground_truth_smiles

            if "||" not in line:

                print(
                    f"WARNING: No || delimiter at "
                    f"generation line {line_number}"
                )

                generated.append("")
                continue

            generated_smiles = (
                line
                .split("||", 1)[0]
                .strip()
            )

            # Remove possible special tokens
            special_tokens = [
                "[PAD]",
                "[SOS]",
                "[EOS]",
                "[X]",
                "[XPara]",
                "[XRing]",
            ]

            for token in special_tokens:
                generated_smiles = (
                    generated_smiles
                    .replace(token, "")
                )

            generated_smiles = (
                generated_smiles.strip()
            )

            generated.append(
                generated_smiles
            )

    print(
        f"Loaded {len(generated)} generated "
        f"molecules."
    )

    return generated


# ============================================================
# INSTRUCTION LENGTH
# ============================================================

def word_length(text):

    """
    Whitespace-separated word count.
    """

    return len(
        text.split()
    )


def load_roberta_tokenizer():

    """
    Try to load the same type of tokenizer used
    by the text model.

    We use AutoTokenizer so this works with the
    HuggingFace model/tokenizer available in the
    environment.

    If unavailable, the script will continue and
    report tokenizer length as N/A.
    """

    try:

        from transformers import (
            AutoTokenizer
        )

        tokenizer = AutoTokenizer.from_pretrained(
            "roberta-base"
        )

        print(
            "\nLoaded RoBERTa tokenizer."
        )

        return tokenizer

    except Exception as e:

        print(
            "\nWARNING: Could not load "
            "RoBERTa tokenizer."
        )

        print(
            f"Reason: {e}"
        )

        return None


def tokenizer_length(
    text,
    tokenizer
):

    if tokenizer is None:
        return np.nan

    try:

        tokens = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False
        )

        return len(
            tokens["input_ids"]
        )

    except Exception:

        return np.nan


# ============================================================
# LENGTH BUCKET
# ============================================================

def length_bucket(length):

    if np.isnan(length):
        return None

    if length <= 64:
        return "1-64"

    elif length <= 96:
        return "65-96"

    elif length <= 128:
        return "97-128"

    elif length <= 160:
        return "129-160"

    else:
        return ">160"


# ============================================================
# BERTZCT
# ============================================================

def calculate_bertzct(smiles):

    try:

        mol = Chem.MolFromSmiles(
            smiles
        )

        if mol is None:
            return np.nan

        return float(
            GraphDescriptors.BertzCT(mol)
        )

    except Exception:

        return np.nan


# ============================================================
# BERTZCT BUCKET
# ============================================================

def bertz_bucket(value):

    if np.isnan(value):
        return None

    if value < 100:
        return "<100"

    elif value < 300:
        return "100-300"

    elif value < 800:
        return "300-800"

    else:
        return ">800"


# ============================================================
# MACCS SIMILARITY
# ============================================================

def calculate_maccs(
    generated,
    ground_truth
):

    try:

        gen_mol = Chem.MolFromSmiles(
            generated
        )

        gt_mol = Chem.MolFromSmiles(
            ground_truth
        )

        if (
            gen_mol is None
            or gt_mol is None
        ):
            return np.nan, False

        gen_fp = (
            MACCSkeys.GenMACCSKeys(
                gen_mol
            )
        )

        gt_fp = (
            MACCSkeys.GenMACCSKeys(
                gt_mol
            )
        )

        score = (
            DataStructs.FingerprintSimilarity(
                gen_fp,
                gt_fp
            )
        )

        return float(score), True

    except Exception:

        return np.nan, False


# ============================================================
# PROCESS ONE SEED
# ============================================================

def process_seed(
    test_records,
    generated,
    tokenizer
):

    n = min(
        len(test_records),
        len(generated)
    )

    if (
        len(test_records)
        != len(generated)
    ):

        print(
            "\nWARNING:"
        )

        print(
            f"Test examples     : "
            f"{len(test_records)}"
        )

        print(
            f"Generated examples: "
            f"{len(generated)}"
        )

        print(
            f"Using first {n} examples."
        )

    results = []

    for i in range(n):

        record = test_records[i]

        gt = record[
            "ground_truth"
        ]

        gen = generated[i]

        # ----------------------------------------------------
        # Instruction length
        # ----------------------------------------------------

        words = word_length(
            record["description"]
        )

        tokens = tokenizer_length(
            record["description"],
            tokenizer
        )

        word_bucket = length_bucket(
            words
        )

        token_bucket = length_bucket(
            tokens
        )

        # ----------------------------------------------------
        # BertzCT
        # ----------------------------------------------------

        bertz = calculate_bertzct(
            gt
        )

        bertz_group = bertz_bucket(
            bertz
        )

        # ----------------------------------------------------
        # MACCS
        # ----------------------------------------------------

        maccs, valid = calculate_maccs(
            gen,
            gt
        )

        results.append({

            "test_index":
                record["test_index"],

            "cid":
                record["cid"],

            "description":
                record["description"],

            "ground_truth":
                gt,

            "generated":
                gen,

            "word_length":
                words,

            "word_bucket":
                word_bucket,

            "token_length":
                tokens,

            "token_bucket":
                token_bucket,

            "bertzct":
                bertz,

            "bertz_bucket":
                bertz_group,

            "maccs":
                maccs,

            "valid":
                valid,
        })

    return pd.DataFrame(
        results
    )


# ============================================================
# BUCKET SUMMARY
# ============================================================

def summarize(
    df,
    bucket_column,
    bucket_order
):

    rows = []

    for bucket in bucket_order:

        subset = df[
            df[bucket_column] == bucket
        ]

        all_count = len(
            subset
        )

        valid = subset[
            subset["maccs"].notna()
        ]

        valid_count = len(
            valid
        )

        if valid_count > 0:

            mean_maccs = (
                valid["maccs"].mean()
            )

            std_maccs = (
                valid["maccs"].std()
                if valid_count > 1
                else 0.0
            )

        else:

            mean_maccs = np.nan
            std_maccs = np.nan

        rows.append({

            "bucket":
                bucket,

            "MACCS_FTS":
                mean_maccs,

            "STD":
                std_maccs,

            "N":
                all_count,

            "N_valid":
                valid_count,

            "validity":
                (
                    valid_count / all_count
                    if all_count > 0
                    else np.nan
                ),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# PRINT DISTRIBUTION
# ============================================================

def print_distribution(df):

    print("\n")
    print("=" * 100)
    print("DATASET COMPLEXITY DISTRIBUTION")
    print("=" * 100)

    # --------------------------------------------------------
    # Word lengths
    # --------------------------------------------------------

    print("\nWORD LENGTH")
    print("-" * 100)

    print(
        df[
            "word_length"
        ].describe()
    )

    print("\nWord-length buckets:")

    print(
        df[
            "word_bucket"
        ]
        .value_counts(
            sort=False
        )
    )

    # --------------------------------------------------------
    # Token lengths
    # --------------------------------------------------------

    print("\nTOKEN LENGTH")
    print("-" * 100)

    if df["token_length"].notna().any():

        print(
            df[
                "token_length"
            ].describe()
        )

        print(
            "\nToken-length buckets:"
        )

        order = [
            "1-64",
            "65-96",
            "97-128",
            "129-160",
            ">160",
        ]

        counts = (
            df["token_bucket"]
            .value_counts(
                sort=False
            )
            .reindex(order)
            .fillna(0)
            .astype(int)
        )

        print(counts)

    else:

        print(
            "Tokenizer lengths unavailable."
        )

    # --------------------------------------------------------
    # BertzCT
    # --------------------------------------------------------

    print("\nBERTZCT")
    print("-" * 100)

    valid_bertz = df[
        df["bertzct"].notna()
    ]["bertzct"]

    if len(valid_bertz) > 0:

        print(
            valid_bertz.describe()
        )

        print(
            "\nBertzCT buckets:"
        )

        order = [
            "<100",
            "100-300",
            "300-800",
            ">800",
        ]

        counts = (
            df["bertz_bucket"]
            .value_counts(
                sort=False
            )
            .reindex(order)
            .fillna(0)
            .astype(int)
        )

        print(counts)

    else:

        print(
            "NO VALID BERTZCT VALUES!"
        )

    print("=" * 100)


# ============================================================
# PRINT TABLE
# ============================================================

def print_table(
    title,
    table,
    model_name="tgm-dlm_adamix"
):

    print("\n")
    print("=" * 100)

    print(title)

    print("=" * 100)

    print(
        f"{'Model':<18}",
        end=""
    )

    for bucket in table["bucket"]:

        print(
            f"{bucket:>12}",
            end=""
        )

    print()

    print("-" * 100)

    print(
        f"{model_name:<18}",
        end=""
    )

    for value in table[
        "MACCS_FTS"
    ]:

        if pd.isna(value):

            text = "N/A"

        else:

            text = f"{value:.3f}"

        print(
            f"{text:>12}",
            end=""
        )

    print()

    print("=" * 100)


# ============================================================
# PRINT DETAILED TABLE
# ============================================================

def print_detailed_table(
    title,
    table
):

    print("\n")
    print(title)

    print("-" * 100)

    print(
        f"{'Bucket':<15}"
        f"{'MACCS':>12}"
        f"{'STD':>12}"
        f"{'N':>10}"
        f"{'Valid':>10}"
        f"{'Validity':>12}"
    )

    print("-" * 100)

    for _, row in table.iterrows():

        if pd.isna(
            row["MACCS_FTS"]
        ):

            score = "N/A"
            std = "N/A"

        else:

            score = (
                f"{row['MACCS_FTS']:.4f}"
            )

            std = (
                f"{row['STD']:.4f}"
            )

        validity = (
            f"{row['validity']:.3f}"
            if not pd.isna(
                row["validity"]
            )
            else "N/A"
        )

        print(
            f"{row['bucket']:<15}"
            f"{score:>12}"
            f"{std:>12}"
            f"{int(row['N']):>10}"
            f"{int(row['N_valid']):>10}"
            f"{validity:>12}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_file",
        default=DEFAULT_TEST_FILE
    )

    parser.add_argument(
        "--seed_file",
        default=DEFAULT_SEED_FILE
    )

    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    print("=" * 100)
    print("tgm-dlm_adamix ONE-SEED SCALABILITY ANALYSIS")
    print("=" * 100)

    print(
        f"\nTest file:\n{args.test_file}"
    )

    print(
        f"\nGeneration file:\n{args.seed_file}"
    )

    # ========================================================
    # LOAD
    # ========================================================

    test_records = load_test_data(
        args.test_file
    )

    generated = load_generation_file(
        args.seed_file
    )

    tokenizer = load_roberta_tokenizer()

    # ========================================================
    # PROCESS
    # ========================================================

    df = process_seed(
        test_records,
        generated,
        tokenizer
    )

    # ========================================================
    # SAVE RAW DATA
    # ========================================================

    raw_path = os.path.join(
        args.output_dir,
        "sample_level_results.csv"
    )

    df.to_csv(
        raw_path,
        index=False
    )

    print(
        f"\nSaved sample-level results:\n"
        f"{raw_path}"
    )

    # ========================================================
    # DISTRIBUTION DIAGNOSTICS
    # ========================================================

    print_distribution(
        df
    )

    # ========================================================
    # LENGTH TABLES
    # ========================================================

    length_order = [
        "1-64",
        "65-96",
        "97-128",
        "129-160",
        ">160",
    ]

    word_table = summarize(
        df,
        "word_bucket",
        length_order
    )

    token_table = summarize(
        df,
        "token_bucket",
        length_order
    )

    # ========================================================
    # BERTZ TABLE
    # ========================================================

    bertz_order = [
        "<100",
        "100-300",
        "300-800",
        ">800",
    ]

    bertz_table = summarize(
        df,
        "bertz_bucket",
        bertz_order
    )

    # ========================================================
    # PRINT PAPER-STYLE TABLES
    # ========================================================

    print_table(
        "TABLE 4-STYLE: INSTRUCTION LENGTH "
        "(WORD COUNT)",
        word_table
    )

    print_table(
        "TABLE 4-STYLE: INSTRUCTION LENGTH "
        "(TOKEN COUNT)",
        token_table
    )

    print_table(
        "TABLE 4-STYLE: MOLECULE COMPLEXITY "
        "(BERTZCT)",
        bertz_table
    )

    # ========================================================
    # PRINT DETAILED TABLES
    # ========================================================

    print_detailed_table(
        "WORD LENGTH — DETAILED",
        word_table
    )

    print_detailed_table(
        "TOKEN LENGTH — DETAILED",
        token_table
    )

    print_detailed_table(
        "BERTZCT — DETAILED",
        bertz_table
    )

    # ========================================================
    # SAVE TABLES
    # ========================================================

    word_path = os.path.join(
        args.output_dir,
        "instruction_length_words.csv"
    )

    token_path = os.path.join(
        args.output_dir,
        "instruction_length_tokens.csv"
    )

    bertz_path = os.path.join(
        args.output_dir,
        "bertzct.csv"
    )

    word_table.to_csv(
        word_path,
        index=False
    )

    token_table.to_csv(
        token_path,
        index=False
    )

    bertz_table.to_csv(
        bertz_path,
        index=False
    )

    # ========================================================
    # SAVE FINAL TEXT TABLE
    # ========================================================

    final_path = os.path.join(
        args.output_dir,
        "table4_tgm_dlm.txt"
    )

    with open(
        final_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "tgm-dlm_adamix Scalability Analysis\n"
        )

        f.write(
            "Metric: MACCS Fingerprint "
            "Tanimoto Similarity\n\n"
        )

        f.write(
            "Instruction Length "
            "(Word Count)\n"
        )

        f.write(
            "Model\t"
            + "\t".join(
                length_order
            )
            + "\n"
        )

        f.write(
            "tgm-dlm_adamix\t"
            + "\t".join(
                "N/A"
                if pd.isna(v)
                else f"{v:.3f}"
                for v in word_table[
                    "MACCS_FTS"
                ]
            )
            + "\n\n"
        )

        f.write(
            "Instruction Length "
            "(Token Count)\n"
        )

        f.write(
            "Model\t"
            + "\t".join(
                length_order
            )
            + "\n"
        )

        f.write(
            "tgm-dlm_adamix\t"
            + "\t".join(
                "N/A"
                if pd.isna(v)
                else f"{v:.3f}"
                for v in token_table[
                    "MACCS_FTS"
                ]
            )
            + "\n\n"
        )

        f.write(
            "BertzCT\n"
        )

        f.write(
            "Model\t"
            + "\t".join(
                bertz_order
            )
            + "\n"
        )

        f.write(
            "tgm-dlm_adamix\t"
            + "\t".join(
                "N/A"
                if pd.isna(v)
                else f"{v:.3f}"
                for v in bertz_table[
                    "MACCS_FTS"
                ]
            )
            + "\n"
        )

    print(
        f"\nFinal table saved to:\n"
        f"{final_path}"
    )

    print("\n")
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()