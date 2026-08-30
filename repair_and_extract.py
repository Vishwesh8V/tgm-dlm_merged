"""
repair_and_extract.py

Takes two files:
  1) main_file  — full generation output, ev.py's expected format:
         "[SOS]<generated>[EOS]   ||   <ground_truth>" per line,
         line number == sample index
  2) bad_file   — the failed-molecules file, format:
         "<index>\\t<smiles>" per line (e.g. tempbadmols.txt)

It repairs parenthesis-imbalance issues in bad_file's SMILES, then
writes a NEW file containing ONLY those indices (not the full
main_file), in the same "[SOS]...[EOS]   ||   gt" format ev.py expects
— pulling each line's ground-truth from main_file at the matching
index.

Usage:
    # repaired version of just the failed lines, in ev.py format
    python repair_and_extract.py generation_output.txt tempbadmols.txt -o repaired_subset.txt

    # UNrepaired version of just the failed lines, in ev.py format
    # (for a clean before/after comparison against the file above)
    python repair_and_extract.py generation_output.txt tempbadmols.txt -o unrepaired_subset.txt --no-repair

    # FULL file (all lines, e.g. 3300) with repaired molecules spliced
    # back in at their original positions — everything else untouched
    python repair_and_extract.py generation_output.txt tempbadmols.txt -o full_repaired.txt --full-output

    # ONLY the molecules that repair actually fixed (excludes still-invalid
    # ones), in ev.py format — for isolating the clean wins
    python repair_and_extract.py generation_output.txt tempbadmols.txt -o only_fixed.txt --only-fixed
"""

import argparse
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def repair_parens(smi: str, mode: str = 'both') -> str:
    """
    Single left-to-right pass over the SMILES string.

    mode='both'   (default): drop unmatched ')' AND append missing ')' at
                  the end for whatever's left open. This is the combined
                  minimum-edit fix.
    mode='delete': ONLY drop unmatched ')' (extra closes). Leaves any
                  unclosed '(' as-is — the string may still be invalid
                  if it had missing closes too.
    mode='add':    ONLY append missing ')' for unclosed '(' at the end.
                  Leaves unmatched ')' in place untouched — the string
                  may still be invalid if it had extra closes too.
    """
    if mode not in ('both', 'delete', 'add'):
        raise ValueError(f"Unknown repair mode: {mode!r}")

    out = []
    depth = 0
    for ch in smi:
        if ch == '(':
            depth += 1
            out.append(ch)
        elif ch == ')':
            if depth == 0:
                if mode in ('both', 'delete'):
                    continue  # unmatched close -> drop it
                else:  # mode == 'add': leave the stray ')' in place
                    out.append(ch)
                continue
            depth -= 1
            out.append(ch)
        else:
            out.append(ch)
    if depth > 0 and mode in ('both', 'add'):
        out.append(')' * depth)  # close whatever's still open
    return ''.join(out)


def is_valid(smi: str) -> bool:
    return Chem.MolFromSmiles(smi) is not None


def load_main_file(path):
    """Returns list of (gen_part, gt_part) tuples, one per line, indexed by line number."""
    lines = []
    with open(path) as f:
        for raw_line in f:
            raw_line = raw_line.rstrip('\n')
            if not raw_line.strip():
                continue
            gen_part, gt_part = raw_line.split('||', 1)
            lines.append((gen_part.strip(), gt_part.strip()))
    return lines


def load_bad_file(path):
    """Returns list of (index, smiles) tuples, in file order."""
    entries = []
    with open(path) as f:
        for raw_line in f:
            raw_line = raw_line.rstrip('\n')
            if not raw_line.strip():
                continue
            idx_str, smi = raw_line.split('\t', 1)
            entries.append((int(idx_str), smi.strip()))
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('main_file', help='Full generation output file (gen||gt format, ev.py input)')
    ap.add_argument('bad_file', help='Failed-molecules file (index\\tsmiles format)')
    ap.add_argument('-o', '--output', default=None,
                     help='Output path (default: <bad_file>_repaired_evformat.txt)')
    ap.add_argument('--keep-tags', action='store_true',
                     help='Wrap output SMILES in [SOS]...[EOS] (optional — get_smis() '
                          'strips these tags either way)')
    ap.add_argument('--no-repair', action='store_true',
                     help='Skip the paren-repair step — just reformat bad_file into ev.py '
                          'format as-is. Useful for producing an UNrepaired baseline file '
                          'to diff against the repaired one.')
    ap.add_argument('--repair-mode', choices=['both', 'delete', 'add'], default='both',
                     help="Which paren fix(es) to apply (default: both). "
                          "'delete' = only drop unmatched extra ')' (Case A). "
                          "'add' = only append missing ')' for unclosed '(' (Case B). "
                          "'both' = do both fixes in one pass. Ignored if --no-repair is set.")
    ap.add_argument('--full-output', action='store_true',
                     help='Write the FULL main_file (all lines) instead of just the '
                          'bad_file subset — every line untouched except the indices in '
                          'bad_file, which get the repaired (or, with --no-repair, '
                          'unchanged) SMILES spliced in. Index N in bad_file replaces '
                          'line N+1 (1-indexed) of main_file, i.e. main_lines[N] in a '
                          '0-indexed array — matches this script\'s existing indexing.')
    ap.add_argument('--only-fixed', action='store_true',
                     help='Write ONLY the molecules that repair successfully made valid '
                          '(excludes entries that are still invalid after repair). '
                          'Incompatible with --full-output (which needs every line) and '
                          '--no-repair (nothing to have "fixed" if repair never ran).')
    args = ap.parse_args()

    if args.only_fixed and args.full_output:
        ap.error('--only-fixed cannot be combined with --full-output')
    if args.only_fixed and args.no_repair:
        ap.error('--only-fixed cannot be combined with --no-repair')

    if args.full_output:
        mode_suffix = '' if args.repair_mode == 'both' else f'_{args.repair_mode}only'
        default_suffix = '_full_unrepaired.txt' if args.no_repair else f'_full_repaired{mode_suffix}.txt'
    elif args.only_fixed:
        default_suffix = '_only_fixed_evformat.txt'
    else:
        mode_suffix = '' if args.repair_mode == 'both' else f'_{args.repair_mode}only'
        default_suffix = '_unrepaired_evformat.txt' if args.no_repair else f'_repaired{mode_suffix}_evformat.txt'
    out_path = args.output or args.bad_file.rsplit('.', 1)[0] + default_suffix

    main_lines = load_main_file(args.main_file)
    bad_entries = load_bad_file(args.bad_file)
    bad_map = dict(bad_entries)  # index -> original smiles, for --full-output lookups

    n_total = len(bad_entries)
    n_fixed = 0
    n_still_bad = 0
    n_missing_index = 0

    if args.full_output:
        with open(out_path, 'w') as out_f:
            for idx, (gen_part, gt_part) in enumerate(main_lines):
                if idx in bad_map:
                    smi = bad_map[idx]
                    final_smi = smi if args.no_repair else repair_parens(smi, mode=args.repair_mode)
                    if is_valid(final_smi):
                        n_fixed += 1
                    else:
                        n_still_bad += 1
                    gen_out = f"[SOS]{final_smi}[EOS]" if args.keep_tags else final_smi
                    out_f.write(f"{gen_out}   ||   {gt_part}\n")
                else:
                    out_f.write(f"{gen_part}   ||   {gt_part}\n")

        # any bad_file indices that fell outside main_file's range never got counted above
        n_missing_index = sum(1 for idx in bad_map if idx < 0 or idx >= len(main_lines))
    else:
        with open(out_path, 'w') as out_f:
            for idx, smi in bad_entries:
                if idx >= len(main_lines) or idx < 0:
                    n_missing_index += 1
                    continue

                _, gt_part = main_lines[idx]

                final_smi = smi if args.no_repair else repair_parens(smi, mode=args.repair_mode)
                valid_now = is_valid(final_smi)
                if valid_now:
                    n_fixed += 1
                else:
                    n_still_bad += 1

                if args.only_fixed and not valid_now:
                    continue  # skip entries that are still invalid

                gen_out = f"[SOS]{final_smi}[EOS]" if args.keep_tags else final_smi
                out_f.write(f"{gen_out}   ||   {gt_part}\n")

    mode_label = "reformatted (no repair applied)" if args.no_repair else f"repaired (mode={args.repair_mode})"
    scope_label = f"full file ({len(main_lines)} lines)" if args.full_output else "bad_file subset only"
    print(f"Mode:                              {mode_label}")
    print(f"Scope:                             {scope_label}")
    print(f"Total entries in bad_file:        {n_total}")
    print(f"Valid after this step:            {n_fixed}  ({100*n_fixed/n_total:.1f}%)")
    print(f"Still invalid:                     {n_still_bad}  ({100*n_still_bad/n_total:.1f}%)")
    if n_missing_index:
        print(f"Indices with no match in main_file: {n_missing_index}")
    print(f"Output file written to:            {out_path}")
    if args.only_fixed:
        print(f"(Contains only the {n_fixed} successfully-repaired lines, "
              f"in ev.py format — still-invalid entries excluded)")
    elif not args.full_output:
        print(f"(Contains only the {n_total - n_missing_index} lines from bad_file, "
              f"in ev.py format, NOT the full main_file)")


if __name__ == '__main__':
    main()