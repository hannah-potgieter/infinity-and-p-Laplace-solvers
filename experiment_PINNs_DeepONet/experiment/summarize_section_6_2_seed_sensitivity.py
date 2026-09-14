#!/usr/bin/env python3
"""Summarize best-checkpoint test MSE values from the Table 4 seed runs."""

import argparse
import re
import statistics
from pathlib import Path


EXAMPLES = ("arctan", "aronsson_square", "aronsson_disc")
SEEDS = (1234, 3456, 5678, 6789, 8765)
BEST_PATTERN = re.compile(
    r"Restored best model from epoch\s+\d+\s+"
    r"\(val_loss:\s*[0-9.eE+-]+,\s*test_loss:\s*([0-9.eE+-]+)\)"
)


def read_best_test_mse(log_path):
    matches = BEST_PATTERN.findall(log_path.read_text(encoding="utf-8"))
    if not matches:
        raise ValueError(f"Best-checkpoint test loss not found in {log_path}")
    return float(matches[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_root", nargs="?", default="seed_study",
        help="Directory passed as RUN_ROOT to run_section_6_2_seed_sensitivity.sh",
    )
    args = parser.parse_args()
    run_root = Path(args.run_root)

    print("example,seed_values,mean,sample_sd")
    for example in EXAMPLES:
        values = []
        for seed in SEEDS:
            log_path = (
                run_root / "outputs" / "expr_6_2" / example
                / f"seed{seed}" / "train.log"
            )
            values.append(read_best_test_mse(log_path))
        value_text = ";".join(f"{value:.6e}" for value in values)
        print(
            f"{example},{value_text},{statistics.fmean(values):.6e},"
            f"{statistics.stdev(values):.6e}"
        )


if __name__ == "__main__":
    main()
