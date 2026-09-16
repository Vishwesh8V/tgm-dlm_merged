#!/usr/bin/env python3
"""
Evaluation sweep for run_full_trajectory_pipeline.sh's outputs.

Builds one table per task (generation, retro, forward), each with a row per
(sampler, K) pair -- DPM (order=2, DPM-Solver++(2M)) and DDIM (order=1,
first-order/DDIM-equivalent) interleaved for every K in --steps, in the
exact row order requested: DPM K=2, DDIM K=2, DPM K=3, DDIM K=3, ... -- and
the 8 columns ev.py's own evaluate_file() already computes, so the numbers
here are guaranteed identical to what running ev.py by hand on each file
would give (this script does not reimplement any metric).

Missing files (e.g. a task whose sampling hasn't finished yet) print as
"pending" rather than crashing, so this can be re-run as more results land
without editing anything.

Usage (run from the repo root, same directory as ev.py):
    python evaluate_trajectory_sweep.py
    python evaluate_trajectory_sweep.py --tasks forward   # just one task
    python evaluate_trajectory_sweep.py --steps 2 3 5      # subset of K
"""
import os
import sys
import glob
import csv
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import ev  # noqa: E402  (reuse ev.py's own evaluate_file -- no metric logic duplicated here)

TASKS = ["generation", "retro", "forward"]
STEPS = [2, 3, 4, 5, 10, 100]
# (row label, --dpm_solver_order value)
SAMPLERS = [("DPM", 2), ("DDIM", 1)]

# (short column header, ev.py's dict key) -- order matches what was asked for:
# BLEU4, EXACT, Leven, Validity, MACCS, RDK, Morgan, FCD
COLUMNS = [
    ("BLEU4", "BLEU Score"),
    ("EXACT", "Exact Match Rate"),
    ("Leven", "Levenshtein Dist"),
    ("Validity", "SMILES Validity"),
    ("MACCS", "MACCS Similarity"),
    ("RDK", "RDK Similarity"),
    ("Morgan", "Morgan Similarity"),
    ("FCD", "FCD Metric"),
]


def find_output_file(gen_dir, task, K, order):
    """
    Try every naming convention this pipeline has actually used, oldest
    first (matches what's really on disk right now):
      flat, with seed:    <gen_dir>/<task>_adaptive_K<K>_order<order>_seed*.txt
      flat, no seed:      <gen_dir>/<task>_adaptive_K<K>_order<order>.txt
      per-task subdir:    <gen_dir>/<task>/K<K>_order<order>_seed*.txt
      per-task subdir:    <gen_dir>/<task>/K<K>_order<order>.txt
    Returns the first match (alphabetically, if several seeds exist), or
    None if nothing matches yet.
    """
    patterns = [
        os.path.join(gen_dir, f"{task}_adaptive_K{K}_order{order}_seed*.txt"),
        os.path.join(gen_dir, f"{task}_adaptive_K{K}_order{order}.txt"),
        os.path.join(gen_dir, task, f"K{K}_order{order}_seed*.txt"),
        os.path.join(gen_dir, task, f"K{K}_order{order}.txt"),
    ]
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[0]
    return None


def format_value(v):
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def build_task_rows(gen_dir, task, steps, samplers):
    rows = []
    for K in steps:
        for label, order in samplers:
            filepath = find_output_file(gen_dir, task, K, order)
            row = {"Row": f"{label} K={K}", "File": filepath or ""}
            if filepath is None:
                row["Status"] = "pending (file not found)"
                for short, _ in COLUMNS:
                    row[short] = ""
            else:
                try:
                    metrics = ev.evaluate_file(filepath)
                    row["Status"] = "ok"
                    for short, long_key in COLUMNS:
                        row[short] = format_value(metrics[long_key])
                except Exception as e:
                    row["Status"] = f"error: {e}"
                    for short, _ in COLUMNS:
                        row[short] = ""
            rows.append(row)
    return rows


def print_markdown_table(task, rows):
    headers = ["Row"] + [short for short, _ in COLUMNS] + ["Status"]
    print(f"\n### {task}\n")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        cells = [row["Row"]] + [row[short] for short, _ in COLUMNS] + [row["Status"]]
        print("| " + " | ".join(cells) + " |")


def write_csv(task, rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task}_results.csv")
    headers = ["Row"] + [short for short, _ in COLUMNS] + ["Status", "File"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})
    return out_path


def write_markdown_file(task, rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task}_results.md")
    headers = ["Row"] + [short for short, _ in COLUMNS] + ["Status"]
    with open(out_path, "w") as f:
        f.write(f"### {task}\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            cells = [row["Row"]] + [row[short] for short, _ in COLUMNS] + [row["Status"]]
            f.write("| " + " | ".join(cells) + " |\n")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gen_dir",
        default=os.path.join(SCRIPT_DIR, "generation_outputs", "trajectory_sweep"),
        help="Directory the trajectory-sweep .txt outputs live under.",
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join(SCRIPT_DIR, "evaluation_outputs"),
        help="Where to write per-task .csv/.md result tables.",
    )
    parser.add_argument("--steps", type=int, nargs="+", default=STEPS)
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    args = parser.parse_args()

    for task in args.tasks:
        rows = build_task_rows(args.gen_dir, task, args.steps, SAMPLERS)
        print_markdown_table(task, rows)
        csv_path = write_csv(task, rows, args.out_dir)
        md_path = write_markdown_file(task, rows, args.out_dir)
        print(f"\n -> saved: {csv_path}")
        print(f" -> saved: {md_path}")


if __name__ == "__main__":
    main()