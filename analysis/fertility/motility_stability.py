#!/usr/bin/env python3
"""Plot sequencing-depth stability of motility/chromatin correlations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis.shared.paths import RESULTS_ROOT, SOURCE_DATA_ROOT
from analysis.shared.plotting import save_figure
import matplotlib.pyplot as plt


PANELS = [
    ("r_prot", "Protamine occupancy", "A"),
    ("r_prot_ss", "Protamine occupancy — sensitivity subset", "B"),
    ("r_nuc", "Nucleosome occupancy", "C"),
    ("r_nuc_ss", "Nucleosome occupancy — sensitivity subset", "D"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=SOURCE_DATA_ROOT / "fertility" / "motility_stability.tsv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=RESULTS_ROOT / "fertility" / "motility_stability",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input, sep="\t")
    depths = sorted(data["depth_bp"].unique())
    labels = [f"{depth / 1e6:g}" for depth in depths]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharey=True)
    for axis, (column, title, panel) in zip(axes.ravel(), PANELS):
        values = [data.loc[data["depth_bp"] == depth, column].to_numpy() for depth in depths]
        boxes = axis.boxplot(values, tick_labels=labels, widths=0.55, patch_artist=True,
                             showfliers=False, medianprops={"color": "#b2182b"})
        for box in boxes["boxes"]:
            box.set(facecolor="#4c78a8", edgecolor="black", linewidth=0.7)
        axis.axhline(0, color="#777777", linewidth=0.6)
        axis.set_title(title, loc="left", fontsize=10)
        axis.set_xlabel("Sequence sampled per individual (Mb)")
        axis.set_ylabel("Pearson r with motility")
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.12, 1.02, panel, transform=axis.transAxes,
                  fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, args.output)
    plt.close(fig)
    print(f"wrote {args.output}.png/.pdf")


if __name__ == "__main__":
    main()
