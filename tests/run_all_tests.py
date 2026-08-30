import sys, os, subprocess

ROOT_MERGED = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TESTS_DIR = os.path.join(ROOT_MERGED, 'tests')
LOG_FILE = os.path.join(TESTS_DIR, 'parity_ablation_raw_log.txt')

def main():
    print("=" * 80)
    print("RUNNING COMPLETE VALIDATION PLAN (§7.1, §7.2, §7.3)")
    print(f"Output log: {LOG_FILE}")
    print("=" * 80)

    test_files = [
        ('test_std_drift.py', '§7.1 / §4.5 std.std() Initial & Post-Refit Drift Test'),
        ('test_three_way_parity.py', '§7.2 Three-Way Parity Ablation Test'),
        ('test_e2e_configurations.py', '§7.1 / §7.3 4-Way Operational Modes Smoke Test'),
        ('test_logging_spec.py', 'Diagnostic Logging Spec (LOGGING_SPEC.md) Test'),
    ]

    all_logs = []
    all_passed = True

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    for py_file, description in test_files:
        path = os.path.join(TESTS_DIR, py_file)
        header = f"\n{'=' * 80}\nRUNNING: {description} ({py_file})\n{'=' * 80}\n"
        print(header)
        all_logs.append(header)

        res = subprocess.run([sys.executable, path], capture_output=True, text=True, env=env, encoding='utf-8', errors='replace')
        combined_output = res.stdout + ("\nSTDERR:\n" + res.stderr if res.stderr else "")
        print(combined_output)
        all_logs.append(combined_output)

        if res.returncode != 0:
            all_passed = False
            print(f"FAILED: {py_file} (exit code {res.returncode})")
        else:
            print(f"PASSED: {py_file}")

    summary = f"\n{'=' * 80}\nOVERALL VALIDATION RESULT: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}\n{'=' * 80}\n"
    print(summary)
    all_logs.append(summary)

    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("".join(all_logs))

    print(f"Complete raw log successfully written to: {LOG_FILE}")
    if not all_passed:
        sys.exit(1)

if __name__ == '__main__':
    main()
