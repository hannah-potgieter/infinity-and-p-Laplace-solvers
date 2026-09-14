"""Plot all finalized continuation checkpoints for the unit-disc PINN."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np


rcParams["mathtext.fontset"] = "cm"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = (
    REPO_ROOT / "plot" / "data"
    / "experiment_6_3_3_distance_boundary_disc_all_p.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs" / "images" / "p_lap_pinns_finite_p_disc.png"
)


def load_metrics(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    required = ("p", "total_loss", "mse_infinity", "mse_p")
    missing = [name for name in required if name not in data.dtype.names]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {', '.join(missing)}")
    return {name: np.atleast_1d(data[name]) for name in required}


def plot_metrics(metrics, output_path):
    order = np.argsort(metrics["p"])
    p_values = metrics["p"][order]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.semilogy(
        p_values, metrics["total_loss"][order], "o-", color="blue",
        markersize=4, label=r"$\mathrm{MSE(total)}$",
    )
    ax.semilogy(
        p_values, metrics["mse_infinity"][order], "s-", color="orange",
        markersize=4, label=r"$\mathrm{MSE}_{\infty}$",
    )
    ax.semilogy(
        p_values, metrics["mse_p"][order], "^-", color="green",
        markersize=4, label=r"$\mathrm{MSE}_{p}$",
    )

    ax.set_xlabel(r"$p$", fontsize=18)
    ax.set_ylabel("MSE", fontsize=18)
    ax.set_title(r"MSE vs $p$", fontsize=20)
    ax.legend(fontsize=13)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot all iterative finite-p unit-disc PINN checkpoints."
    )
    parser.add_argument(
        "--metrics-csv", type=Path, default=DEFAULT_METRICS,
        help="CSV containing p, total_loss, mse_infinity, and mse_p.",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output PNG path.",
    )
    args = parser.parse_args()
    plot_metrics(load_metrics(args.metrics_csv), args.output)
