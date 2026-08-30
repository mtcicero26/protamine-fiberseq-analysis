#!/usr/bin/env python3
"""Plot curated GO enrichment for lacuna-enriched promoters."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.shared.paths import RESULTS_ROOT, SOURCE_DATA_ROOT
from analysis.shared.plotting import save_figure
import matplotlib.pyplot as plt


TERM_LABELS = {
    "GO:systemdevelopment": "system development",
    "GO:development": "development",
    "GO:embryodevelopment": "embryo development",
    "GO:chromatinorganization": "chromatin organization",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=SOURCE_DATA_ROOT / "promoter_regulation" / "curated_go_enrichment.tsv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_ROOT / "promoter_regulation" / "curated_go_enrichment",
    )
    parser.add_argument("--q-threshold", type=float, default=0.20)
    args = parser.parse_args()

    data = pd.read_csv(args.input, sep="\t")
    required = {"term", "overlap", "expected", "fold", "q_value"}
    if missing := required.difference(data.columns):
        raise ValueError(f"{args.input} lacks columns: {sorted(missing)}")
    plotted = data.loc[data["q_value"] < args.q_threshold].copy()
    plotted = plotted.sort_values("q_value", ascending=False)
    if plotted.empty:
        raise ValueError(f"No terms pass q < {args.q_threshold}")

    y = np.arange(len(plotted))
    significance = -np.log10(np.clip(plotted["q_value"], 1e-300, None))
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    bars = ax.barh(y, significance, color="#d55e00", alpha=0.90)
    ax.set_yticks(y)
    ax.set_yticklabels([TERM_LABELS.get(term, term) for term in plotted["term"]])
    for bar, (_, row) in zip(bars, plotted.iterrows()):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  {row['fold']:.2f}×; {int(row['overlap'])}/{row['expected']:.1f}",
            va="center",
            fontsize=8,
        )
    ax.set_xlabel("−log10(BH-adjusted P value)")
    ax.set_title("GO enrichment among lacuna-enriched promoters", loc="left")
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, args.output)
    plt.close(fig)
    print(f"wrote {args.output}.png/.pdf")


if __name__ == "__main__":
    main()
