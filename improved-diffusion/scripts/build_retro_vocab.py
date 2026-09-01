"""
Builds generate_vocab.txt for a dataset directory in the format regexTokenizer expects
(mytokenizers.py): one token per line, bracket-atom tokens like '[C@@H]' or '[nH+]' plus
the fixed organic-subset base characters. toktoid ids are assigned as line_number + 3
(ids 0/1/2 are reserved for [PAD]/[SOS]/[EOS]), so this file's row order defines the
vocabulary -- append-only if you ever add more data later, don't reorder existing lines
or you'll invalidate any checkpoint already trained against this vocab.

Scans BOTH the smiles and desc columns of every {split}.txt in the target directory,
because ChEBIdataset.__getitem__ always tokenizes the smiles column with this same
tokenizer (see mydatasets.py) -- a token missing from the vocab is a KeyError at
training time, not a silent skip.

Usage:
    python build_retro_vocab.py --dataset_dir ../../datasets/RETRO
"""
import argparse
import os
import re

BRACKET_RE = re.compile(r'\[[^\]]+\]')

# The fixed organic-subset base charset already used by datasets/SMILES/generate_vocab.txt.
# Kept identical here so a dataset built with build_retro_dataset.py's Kekulized output
# doesn't need anything outside this set for non-bracket atoms.
BASE_TOKENS = list('0123456789') + ['Br', 'Cl'] + list('BCNOSPFI') + \
    list('()=#-+\\/:~@?>*$.') + [r'%[0-9]{2}']  # last one is a formatting note, not emitted


def scan_file(path, tokens):
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) != 3:
                continue
            for smi in (parts[1], parts[2]):
                for tok in BRACKET_RE.findall(smi):
                    tokens.add(tok)
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_dir', required=True)
    ap.add_argument('--splits', nargs='+', default=['train', 'validation', 'test'])
    args = ap.parse_args()

    bracket_tokens = set()
    total = 0
    for split in args.splits:
        total += scan_file(os.path.join(args.dataset_dir, f'{split}.txt'), bracket_tokens)

    if total == 0:
        raise SystemExit(f"No rows found under {args.dataset_dir} for splits {args.splits} -- "
                          "run build_retro_dataset.py first.")

    base = ['5', '8', '6', '4', '3', '7', '2', '9', '1',
            '#', 'Br', 'O', 'N', '/', 'P', '.', ')', 'Cl', 'C', 'I', '(', 'S', 'B', '\\', '=', 'F']
    ordered = base + sorted(bracket_tokens)

    out_path = os.path.join(args.dataset_dir, 'generate_vocab.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        for tok in ordered:
            f.write(tok + '\n')

    print(f"Scanned {total} SMILES strings across {args.splits}")
    print(f"Found {len(bracket_tokens)} distinct bracket-atom tokens not in the base charset")
    print(f"Wrote {len(ordered)} tokens to {out_path}")


if __name__ == '__main__':
    main()