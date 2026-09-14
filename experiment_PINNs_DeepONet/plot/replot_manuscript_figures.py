"""Assemble all manuscript images and replot numerical Figures 4--12."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]

STATIC_IMAGES = [
    "dist_visual.png",
    "pinns.png",
    "deeponet.png",
]

NUMERICAL_IMAGES = [
    "loss_inf_lap_arctan.png",
    "appr_inf_lap_arctan.png",
    "exact_inf_lap_arctan.png",
    "loss_inf_lap_aronsson.png",
    "appr_inf_lap_aronsson.png",
    "exact_inf_lap_aronsson.png",
    "loss_inf_lap_aronsson_circle.png",
    "appr_inf_lap_aronsson_circle.png",
    "exact_inf_lap_aronsson_circle.png",
    "p_lap_pinns_test_loss3.png",
    "p_lap_pinns_finite_p_disc.png",
    "p_lap_deeponet_test1.png",
    "p_lap_deeponet_test2.png",
    "p_lap_deeponet_2ddisc.png",
    "p_lap_deeponet_2dellipse1.png",
    "p_lap_deeponet_2dellipse2.png",
    "p_lap_deeponet_2dellipse3.png",
    "p_lap_deeponet_3dsphere.png",
    "p_lap_deeponet_3dcylinder.png",
    "p_lap_deeponet_3dtorus.png",
    "p_lap_deeponet_2dellipseall_theta.png",
    "p_lap_deeponet_2dellipseall_ab.png",
    "p_lap_deeponet_2dellipseall_theta_int2.png",
    "p_lap_deeponet_2dellipseall_theta_int4.png",
    "p_lap_deeponet_2dellipseall_theta_int6.png",
    "p_lap_deeponet_2dellipseall_theta_int8.png",
]

EXPECTED_IMAGES = STATIC_IMAGES + NUMERICAL_IMAGES


def run(command: list[str], env: dict[str, str]):
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Copy the static artwork for Figures 1--3 and regenerate the "
            "numerical image files used by Figures 4--12."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "images",
        help="Destination for the regenerated PNG files.",
    )
    args = parser.parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    static_dir = REPO_ROOT / "outputs" / "images"
    for filename in STATIC_IMAGES:
        source = static_dir / filename
        destination = out_dir / filename
        if not source.is_file():
            raise SystemExit(f"Missing static manuscript artwork: {source}")
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
            print(f"Copied static artwork {source} -> {destination}")

    with tempfile.TemporaryDirectory(prefix="plaplace-matplotlib-") as cache:
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env["MPLCONFIGDIR"] = cache
        env["XDG_CACHE_HOME"] = cache
        python = sys.executable

        commands = [
            [python, "plot/experiment_6_2_plot.py", "--example", "arctan", "--out-dir", str(out_dir)],
            [python, "plot/experiment_6_2_plot.py", "--example", "aronsson_square", "--out-dir", str(out_dir)],
            [python, "plot/experiment_6_2_plot.py", "--example", "aronsson_disc", "--out-dir", str(out_dir)],
            [python, "plot/experiment_6_3_plot.py", "--domain", "square", "--out-dir", str(out_dir)],
            [python, "plot/experiment_6_3_3_distance_boundary_disc_plot.py", "--output", str(out_dir / "p_lap_pinns_finite_p_disc.png")],
            [python, "plot/experiment_6_4_plot.py", "--type", "origin", "--out-dir", str(out_dir)],
            [python, "plot/experiment_6_4_plot.py", "--type", "boundary", "--out-dir", str(out_dir)],
            [python, "plot/experiment_6_4_3_plot.py", "--total-div", "all", "--out-dir", str(out_dir)],
        ]
        for command in commands:
            run(command, env)

    missing = [name for name in EXPECTED_IMAGES if not (out_dir / name).is_file()]
    if missing:
        raise SystemExit("Missing expected manuscript images: " + ", ".join(missing))
    print(
        f"Assembled and verified {len(EXPECTED_IMAGES)} manuscript image files "
        f"({len(STATIC_IMAGES)} static and {len(NUMERICAL_IMAGES)} replotted) in {out_dir}"
    )


if __name__ == "__main__":
    main()
