"""
Phase 1 dataset converter (see RETRO_ADAPTATION_PLAN.md, sections 3 and 6).

Builds a retrosynthesis dataset directory with the exact file layout ChEBIdataset
already expects (`{split}.txt` as `cid\\tsmiles\\tdesc\\n`, one header line, 3 tab
columns), so the existing training pipeline (gaussian_diffusion.py, train.py,
process_text.py) needs zero changes to its diffusion math -- only --data_dir needs
to point here.

Role mapping (see plan section 3):
    smiles column (diffusion target)      <- reactant SMILES (multi-component, '.'-joined)
    desc column   (cross-attention cond.) <- product SMILES

Supports two raw formats, auto-detected:

1. Mol-Instructions style (what UTGDiff's MOIretro uses) -- a JSON list of dicts:
       {"instruction": ..., "input": <SELFIES product>, "output": <SELFIES reactants>,
        "metadata": {"split": "train"|"valid"|"test"}}
   Requires the `selfies` package (pip install selfies).

2. Generic USPTO-50k-style reaction-SMILES file -- one reaction per line, either
   plain text or CSV, containing a "reactants>>product" (or "reactants>reagents>product")
   SMILES string, plus an optional split column. Pass --format uspto50k and tell us
   the column layout with --rxn_col / --split_col / --delimiter if it's not the
   common Kaggle/USPTO-50k `id,class,reactants>reagents>production,split` layout.

Usage:
    python build_retro_dataset.py --input retrosynthesis.json --format molinstructions \\
        --out_dir ../../datasets/RETRO

    python build_retro_dataset.py --input uspto50k_raw.csv --format uspto50k \\
        --out_dir ../../datasets/RETRO
"""
import argparse
import csv
import json
import os
import sys

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    HAVE_RDKIT = True
except ImportError:
    HAVE_RDKIT = False


def canonicalize(smi):
    """RDKit canonicalization, Kekulized. Returns None if RDKit rejects the SMILES.

    IMPORTANT: this repo's existing ChEBI dataset (datasets/SMILES/*.txt) is Kekulized --
    generate_vocab.txt contains no aromatic-lowercase tokens (no bare 'c','n','o','s','p';
    see mydatasets.py's changeorder() which calls Chem.Kekulize explicitly). Real-world
    retro sources (USPTO-50k, Mol-Instructions) are normally canonicalized WITH aromatic
    lowercase. If we don't Kekulize here too, the first aromatic ring in the data will hit
    a token that's absent from any vocab we build in that same style, and regexTokenizer's
    self.toktoid[...] lookup will KeyError the moment training/process_text.py tokenizes it.
    Kekulizing keeps the representation consistent with how this pipeline has always been
    exercised, at the cost of slightly longer token sequences than aromatic notation would give."""
    if not HAVE_RDKIT:
        return smi
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Chem.KekulizeException:
        return None
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


def load_molinstructions(path):
    """Mol-Instructions retrosynthesis.json -> list of (split, product_smiles, reactant_smiles)."""
    try:
        import selfies as sf
    except ImportError:
        sys.exit("Mol-Instructions format is SELFIES-encoded; install with: pip install selfies")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    out = []
    n_bad = 0
    for info in data:
        try:
            product_smiles = sf.decoder(info['input'], compatible=True)
            reactant_smiles = sf.decoder(info['output'], compatible=True)
        except Exception:
            n_bad += 1
            continue
        split = info.get('metadata', {}).get('split', 'train')
        split = {'valid': 'validation'}.get(split, split)
        out.append((split, product_smiles, reactant_smiles))
    if n_bad:
        print(f"[build_retro_dataset] {n_bad} examples failed SELFIES decoding and were dropped")
    return out


def load_uspto50k(path, rxn_col=None, split_col=None, delimiter=None):
    """Generic reaction-SMILES file -> list of (split, product_smiles, reactant_smiles).
    Auto-detects the common USPTO-50k Kaggle layout
    (id,class,reactants>reagents>production,split) if columns aren't specified."""
    with open(path, 'r', encoding='utf-8') as f:
        sniff = f.read(4096)
        f.seek(0)
        if delimiter is None:
            delimiter = ',' if sniff.count(',') > sniff.count('\t') else '\t'
        reader = csv.reader(f, delimiter=delimiter)
        rows = list(reader)

    if not rows:
        return []

    header = rows[0]
    has_header = any('>' not in c and not c.replace('.', '').isdigit() for c in header) and \
        not any('>' in c for c in header)

    if has_header:
        header_lower = [h.strip().lower() for h in header]
        if rxn_col is None:
            for i, h in enumerate(header_lower):
                if 'reagent' in h or 'reaction' in h or 'smiles' in h or 'production' in h:
                    rxn_col = i
                    break
        if split_col is None and 'split' in header_lower:
            split_col = header_lower.index('split')
        data_rows = rows[1:]
    else:
        data_rows = rows
        if rxn_col is None:
            rxn_col = 0

    if rxn_col is None:
        sys.exit("Could not auto-detect the reaction-SMILES column. Pass --rxn_col explicitly "
                  "(0-indexed) after checking a few lines of your file.")

    out = []
    n_bad = 0
    for row in data_rows:
        if rxn_col >= len(row):
            continue
        rxn = row[rxn_col].strip()
        parts = rxn.split('>')
        if len(parts) == 3:
            reactants, _reagents, product = parts
        elif len(parts) == 2:
            reactants, product = parts
        else:
            n_bad += 1
            continue
        if not reactants or not product:
            n_bad += 1
            continue
        split = row[split_col].strip() if split_col is not None and split_col < len(row) else 'train'
        out.append((split, product, reactants))
    if n_bad:
        print(f"[build_retro_dataset] {n_bad} rows had an unparseable reaction SMILES and were dropped")
    return out


def write_splits(examples, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    by_split = {}
    for split, product, reactants in examples:
        by_split.setdefault(split, []).append((product, reactants))

    split_to_filename = {'train': 'train.txt', 'validation': 'validation.txt', 'test': 'test.txt'}
    unknown = set(by_split) - set(split_to_filename)
    if unknown:
        print(f"[build_retro_dataset] Unrecognized split label(s) {unknown} -- writing them into train.txt")
        by_split.setdefault('train', [])
        for u in unknown:
            by_split['train'].extend(by_split.pop(u))

    counts = {}
    all_bracket_tokens = set()
    for split, fname in split_to_filename.items():
        rows = by_split.get(split, [])
        path = os.path.join(out_dir, fname)
        n_written, n_skipped = 0, 0
        with open(path, 'w', encoding='utf-8') as f:
            f.write('cid\tsmiles\tdesc\n')  # header -- ChEBIdataset.get_ori_data skips line 0
            for cid, (product, reactants) in enumerate(rows):
                can_product = canonicalize(product)
                can_reactants_parts = [canonicalize(p) for p in reactants.split('.')]
                if can_product is None or any(p is None for p in can_reactants_parts):
                    n_skipped += 1
                    continue
                can_reactants = '.'.join(can_reactants_parts)
                f.write(f"{cid}\t{can_reactants}\t{can_product}\n")
                n_written += 1
        counts[fname] = (n_written, n_skipped)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Path to the raw retrosynthesis data file')
    ap.add_argument('--format', choices=['molinstructions', 'uspto50k'], required=True)
    ap.add_argument('--out_dir', default='../../datasets/RETRO')
    ap.add_argument('--rxn_col', type=int, default=None, help='(uspto50k) 0-indexed column with the reaction SMILES')
    ap.add_argument('--split_col', type=int, default=None, help='(uspto50k) 0-indexed column with the split label')
    ap.add_argument('--delimiter', default=None, help='(uspto50k) override delimiter auto-detection')
    args = ap.parse_args()

    if not HAVE_RDKIT:
        print("[build_retro_dataset] WARNING: rdkit not importable -- skipping canonicalization/validation. "
              "ChEBIdataset does not validate SMILES itself, so garbage input will silently reach training.")

    if args.format == 'molinstructions':
        examples = load_molinstructions(args.input)
    else:
        examples = load_uspto50k(args.input, args.rxn_col, args.split_col, args.delimiter)

    if not examples:
        sys.exit("No examples parsed -- check --format and, for uspto50k, --rxn_col/--split_col.")

    counts = write_splits(examples, args.out_dir)
    print(f"\nWrote dataset to {args.out_dir}:")
    for fname, (written, skipped) in counts.items():
        print(f"  {fname}: {written} examples written, {skipped} dropped (failed RDKit sanitization)")
    print("\nNext: run build_retro_vocab.py against this directory to generate generate_vocab.txt, "
          "then process_text.py --dataset_dir <out_dir> for each split.")


if __name__ == '__main__':
    main()