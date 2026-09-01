#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, RDKFingerprint, AllChem

RDLogger.DisableLog("rdApp.*")


def canonicalize(smiles):
    if not isinstance(smiles, str):
        return None

    smiles = smiles.strip()

    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None

        return Chem.MolToSmiles(
            mol,
            canonical=True
        )

    except Exception:
        return None


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Generated CSV from pcdes_zero_shot script"
    )

    args = parser.parse_args()

    print("=" * 90)
    print("PCDes ZERO-SHOT METRIC EVALUATION")
    print("=" * 90)

    print(
        f"Input: {args.input}"
    )

    df = pd.read_csv(
        args.input
    )

    print(
        f"Samples: {len(df)}"
    )

    maccs_scores = []
    rdk_scores = []
    morgan_scores = []

    valid_count = 0
    exact_count = 0

    invalid_examples = []

    for i, row in df.iterrows():

        generated = str(
            row["generated_smiles"]
        ).strip()

        reference = str(
            row["reference_smiles"]
        ).strip()

        gen_mol = Chem.MolFromSmiles(
            generated
        )

        ref_mol = Chem.MolFromSmiles(
            reference
        )

        # ====================================================
        # VALID
        # ====================================================

        if gen_mol is None:

            invalid_examples.append(
                (
                    i,
                    generated
                )
            )

            continue

        valid_count += 1

        # ====================================================
        # If reference is invalid, don't calculate similarity
        # ====================================================

        if ref_mol is None:
            continue

        # ====================================================
        # MACCS
        # ====================================================

        gen_fp = MACCSkeys.GenMACCSKeys(
            gen_mol
        )

        ref_fp = MACCSkeys.GenMACCSKeys(
            ref_mol
        )

        maccs = DataStructs.TanimotoSimilarity(
            gen_fp,
            ref_fp
        )

        maccs_scores.append(
            maccs
        )

        # ====================================================
        # RDK
        # ====================================================

        gen_fp = RDKFingerprint(
            gen_mol
        )

        ref_fp = RDKFingerprint(
            ref_mol
        )

        rdk = DataStructs.TanimotoSimilarity(
            gen_fp,
            ref_fp
        )

        rdk_scores.append(
            rdk
        )

        # ====================================================
        # MORGAN
        # ====================================================

        gen_fp = AllChem.GetMorganFingerprintAsBitVect(
            gen_mol,
            radius=2,
            nBits=2048
        )

        ref_fp = AllChem.GetMorganFingerprintAsBitVect(
            ref_mol,
            radius=2,
            nBits=2048
        )

        morgan = DataStructs.TanimotoSimilarity(
            gen_fp,
            ref_fp
        )

        morgan_scores.append(
            morgan
        )

        # ====================================================
        # EXACT MATCH
        # ====================================================

        gen_can = canonicalize(
            generated
        )

        ref_can = canonicalize(
            reference
        )

        if (
            gen_can is not None
            and ref_can is not None
            and gen_can == ref_can
        ):

            exact_count += 1

    # ========================================================
    # FINAL METRICS
    # ========================================================

    total = len(df)

    valid = (
        valid_count / total
    )

    exact = (
        exact_count / total
    )

    maccs = (
        np.mean(maccs_scores)
        if maccs_scores
        else 0.0
    )

    rdk = (
        np.mean(rdk_scores)
        if rdk_scores
        else 0.0
    )

    morgan = (
        np.mean(morgan_scores)
        if morgan_scores
        else 0.0
    )

    # ========================================================
    # TABLE 5
    # ========================================================

    print()
    print("=" * 90)
    print("TABLE 5 — ZERO-SHOT GENERALIZATION ON PCDes")
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
        f"{maccs:>12.3f}"
        f"{rdk:>12.3f}"
        f"{morgan:>12.3f}"
        f"{'N/A':>12}"
        f"{exact:>12.3f}"
        f"{valid:>12.3f}"
    )

    print("=" * 90)

    # ========================================================
    # ADDITIONAL INFORMATION
    # ========================================================

    print()
    print(
        f"Total samples       : {total}"
    )

    print(
        f"Valid generations   : "
        f"{valid_count}/{total}"
    )

    print(
        f"Invalid generations : "
        f"{total - valid_count}/{total}"
    )

    print(
        f"Exact matches       : "
        f"{exact_count}/{total}"
    )

    print(
        f"MACCS evaluated     : "
        f"{len(maccs_scores)}"
    )

    print(
        f"RDK evaluated       : "
        f"{len(rdk_scores)}"
    )

    print(
        f"Morgan evaluated    : "
        f"{len(morgan_scores)}"
    )

    # ========================================================
    # INVALID EXAMPLES
    # ========================================================

    if invalid_examples:

        print()
        print(
            "First invalid generations:"
        )

        for idx, smiles in invalid_examples[:10]:

            print(
                f"[{idx}] {smiles}"
            )

    print()
    print(
        "Done."
    )


if __name__ == "__main__":
    main()