#!/usr/bin/env python3
"""Collect the selected unit-disc continuation stages into one CSV file."""

import argparse
import csv
from pathlib import Path

import numpy as np


STAGE_NAMES = (
    "p2to10_disc",
    "p11to15_disc",
    "p16to20_disc",
    "p20to20_disc",
    "p25to50_disc",
    "p60to100_disc",
    "p150to500_disc",
    "p600to1000_disc",
)


def collect(run_root):
    base = Path(run_root) / "expr_6_3_iter_distance_boundary_disc"
    rows_by_p = {}

    for stage_name in STAGE_NAMES:
        npy_dir = base / stage_name / "npy"
        metrics_path = npy_dir / "stage_metrics.csv"
        history_path = npy_dir / "training_history.npz"
        if not metrics_path.exists() or not history_path.exists():
            raise FileNotFoundError(
                f"Missing metrics or history for stage {stage_name}: {npy_dir}"
            )

        history = np.load(history_path)
        epoch_p = np.asarray(history["p"], dtype=float)
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            metrics = list(csv.DictReader(handle))

        for metric in metrics:
            p_value = float(metric["p"])
            indices = np.flatnonzero(np.isclose(epoch_p, p_value))
            if len(indices) == 0:
                raise ValueError(
                    f"No history entries for p={p_value:g} in {history_path}"
                )
            final_index = int(indices[-1])
            rows_by_p[p_value] = {
                "p": f"{p_value:g}",
                "total_loss": f"{float(history['total_loss'][final_index]):.9e}",
                "boundary_loss": f"{float(history['bc_loss'][final_index]):.9e}",
                "pinns_loss": f"{float(history['pde_loss'][final_index]):.9e}",
                "mse_infinity": f"{float(metric['mse_limit']):.9e}",
                "mse_p": f"{float(metric['mse_exact']):.9e}",
            }

    return [rows_by_p[p_value] for p_value in sorted(rows_by_p)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = collect(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
