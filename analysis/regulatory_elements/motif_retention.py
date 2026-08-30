#!/usr/bin/env python3
"""Plot promoter and distal motif-associated lacuna retention effects."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.shared.paths import RESULTS_ROOT, SOURCE_DATA_ROOT
from analysis.shared.plotting import save_figure
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


CATEGORY_ORDER = [
    "architectural_universal",
    "dev_gene_associated",
    "germline_late_active",
    "germline_SSC_or_pluripotency",
    "somatic",
    "neg_control",
]
CATEGORY_LABELS = {
    "architectural_universal": "Architectural /\nuniversal",
    "dev_gene_associated": "Developmental-gene\nassociated",
    "germline_late_active": "Germline\nlate-active",
    "germline_SSC_or_pluripotency": "Germline SSC /\npluripotency",
    "somatic": "Somatic",
    "neg_control": "Negative\ncontrol",
}
CATEGORY_COLORS = {
    "architectural_universal": "#4c78a8",
    "dev_gene_associated": "#72b7b2",
    "germline_late_active": "#f58518",
    "germline_SSC_or_pluripotency": "#eeca3b",
    "somatic": "#b279a2",
    "neg_control": "#9d9d9d",
}


def plot_paired_swarm(data: pd.DataFrame, output_stem: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    tick_labels: list[str] = []

    for category_index, category in enumerate(CATEGORY_ORDER):
        counts: list[int] = []
        for context, offset, marker in (
            ("promoter", -0.18, "o"),
            ("distal", 0.18, "D"),
        ):
            subset = data.loc[
                (data["category"] == category) & (data["ctx"] == context)
            ].copy()
            counts.append(len(subset))
            if subset.empty:
                continue
            x = category_index + offset + rng.uniform(-0.09, 0.09, len(subset))
            ax.scatter(
                x,
                subset["signed_effect"],
                marker=marker,
                s=32,
                color=CATEGORY_COLORS[category],
                edgecolor="black",
                linewidth=0.35,
                alpha=0.90,
            )
            mean = subset["signed_effect"].mean()
            ax.hlines(mean, category_index + offset - 0.14, category_index + offset + 0.14,
                      color="black", linewidth=1.5)
            highlighted = subset.loc[subset["tf"] == "CTCF"]
            if not highlighted.empty:
                point = highlighted.iloc[0]
                point_x = x[list(subset.index).index(point.name)]
                ax.scatter(point_x, point["signed_effect"], marker=marker, s=100,
                           facecolor="none", edgecolor="#b2182b", linewidth=1.4)
                if context == "promoter":
                    ax.annotate("CTCF", (point_x, point["signed_effect"]),
                                xytext=(5, 4), textcoords="offset points", fontsize=8)
        tick_labels.append(f"{CATEGORY_LABELS[category]}\n(n={counts[0]}/{counts[1]})")

    ax.axhline(0, color="#666666", linestyle="--", linewidth=0.7)
    ax.set_xticks(np.arange(len(CATEGORY_ORDER)))
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel("Lacuna-retention effect at motif sites")
    ax.set_title("Motif-associated retention by transcription-factor category", loc="left")
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777",
                   markeredgecolor="black", label="promoter"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor="#777777",
                   markeredgecolor="black", label="distal"),
        ],
        frameon=False,
        loc="upper right",
    )
    fig.tight_layout()
    save_figure(fig, output_stem)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=SOURCE_DATA_ROOT / "regulatory_elements" / "motif_retention_effects.tsv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_ROOT / "regulatory_elements" / "motif_retention",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = pd.read_csv(args.input, sep="\t")
    required = {"tf", "signed_effect", "category", "ctx"}
    if missing := required.difference(data.columns):
        raise ValueError(f"{args.input} lacks columns: {sorted(missing)}")
    plot_paired_swarm(data, args.output, args.seed)
    print(f"wrote {args.output}.png/.pdf")


if __name__ == "__main__":
    main()
