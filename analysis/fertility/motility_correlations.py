#!/usr/bin/env python3
"""Plot sperm chromatin metrics against CASA motility."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

from analysis.shared.paths import RESULTS_ROOT, SOURCE_DATA_ROOT
from analysis.shared.plotting import save_figure
import matplotlib.pyplot as plt


GROUP_COLORS = {"low": "#0072b2", "mid": "#888888", "high": "#d55e00"}
METRICS = [
    ("prot_frac", "Fraction of bases in protamine footprints"),
    ("nuc_per_kb", "Nucleosome footprints per kb"),
    ("gap_frac", "Fraction of bases accessible"),
]


def scatter_panel(axis, data: pd.DataFrame, metric: str, label: str) -> None:
    for group in ("low", "mid", "high"):
        subset = data.loc[data["group"] == group]
        axis.scatter(
            subset["motility"],
            subset[metric],
            s=58,
            color=GROUP_COLORS[group],
            edgecolor="black",
            linewidth=0.5,
            label=group,
            zorder=3,
        )
    for _, row in data.iterrows():
        axis.annotate(row["sample"], (row["motility"], row[metric]),
                      xytext=(3, 3), textcoords="offset points", fontsize=7)
    correlation, p_value = pearsonr(data["motility"], data[metric])
    axis.text(
        0.04, 0.96,
        f"Pearson r = {correlation:.2f}\nP = {p_value:.2g}",
        transform=axis.transAxes,
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cccccc"},
    )
    axis.set_xlabel("Sperm motility (%)")
    axis.set_ylabel(label)
    axis.grid(alpha=0.20)
    axis.spines[["top", "right"]].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=SOURCE_DATA_ROOT / "fertility" / "motility_metrics.tsv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=RESULTS_ROOT / "fertility" / "motility_correlations",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input, sep="\t").sort_values("motility")
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.5))
    for axis, (metric, label) in zip(axes, METRICS):
        scatter_panel(axis, data, metric, label)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, args.output)
    plt.close(fig)
    print(f"wrote {args.output}.png/.pdf")


if __name__ == "__main__":
    main()
