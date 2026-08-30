#!/usr/bin/env python3
"""Redraw regional nucleosome/protamine occupancy from compact results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.shared.paths import RESULTS_ROOT, SOURCE_DATA_ROOT
from analysis.shared.plotting import save_figure
import matplotlib.pyplot as plt


REGION_COLORS = {
    "gene_body": "#7f7f7f",
    "intergenic": "#7f7f7f",
    "promoter_lacuna_enriched": "#7f7f7f",
    "promoter_lacuna_neutral": "#7f7f7f",
    "telomere": "#66c2a5",
    "centromere": "#3182bd",
    "pericentromere": "#08306b",
    "CDR": "#e41a1c",
    "rDNA": "#ff7f00",
}
REGION_LABELS = {
    "gene_body": "gene body",
    "intergenic": "intergenic",
    "promoter_lacuna_enriched": "lacuna-enriched promoter",
    "promoter_lacuna_neutral": "other promoter",
    "telomere": "telomere",
    "centromere": "centromere",
    "pericentromere": "pericentromere",
    "CDR": "centromere dip region",
    "rDNA": "rDNA",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=SOURCE_DATA_ROOT / "genome_context" / "region_occupancy.tsv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_ROOT / "genome_context" / "region_occupancy",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input, sep="\t")
    required = {
        "region", "reg_total_bp", "nuc_fold", "prot_fold",
        "nuc_fold_lo", "nuc_fold_hi", "prot_fold_lo", "prot_fold_hi",
    }
    if missing := required.difference(data.columns):
        raise ValueError(f"{args.input} lacks columns: {sorted(missing)}")

    x = np.log2(data["nuc_fold"].to_numpy())
    y = np.log2(data["prot_fold"].to_numpy())
    x_error = np.vstack(
        [
            x - np.log2(data["nuc_fold_lo"].to_numpy()),
            np.log2(data["nuc_fold_hi"].to_numpy()) - x,
        ]
    )
    y_error = np.vstack(
        [
            y - np.log2(data["prot_fold_lo"].to_numpy()),
            np.log2(data["prot_fold_hi"].to_numpy()) - y,
        ]
    )
    coverage = np.maximum(data["reg_total_bp"].to_numpy(), 1)
    point_sizes = 45 + 55 * (np.log10(coverage) - np.log10(coverage).min() + 0.25)
    colors = [REGION_COLORS.get(region, "#444444") for region in data["region"]]

    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    for index in range(len(data)):
        ax.errorbar(
            x[index], y[index],
            xerr=x_error[:, index:index + 1],
            yerr=y_error[:, index:index + 1],
            fmt="none", ecolor=colors[index], alpha=0.45, capsize=2,
        )
    ax.scatter(x, y, s=point_sizes, c=colors, edgecolor="black", linewidth=0.5, zorder=3)
    for index, region in enumerate(data["region"]):
        ax.annotate(
            REGION_LABELS.get(region, region),
            (x[index], y[index]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(0, color="#777777", linestyle="--", linewidth=0.7)
    ax.axvline(0, color="#777777", linestyle="--", linewidth=0.7)
    ax.set_xlabel("log2 nucleosome-occupancy fold change")
    ax.set_ylabel("log2 protamine-occupancy fold change")
    ax.set_title("Regional occupancy relative to length-matched background", loc="left")
    ax.grid(alpha=0.16)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, args.output)
    plt.close(fig)
    print(f"wrote {args.output}.png/.pdf")


if __name__ == "__main__":
    main()
