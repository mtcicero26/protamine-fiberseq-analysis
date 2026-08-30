#!/usr/bin/env python3
"""Redraw Hia5 titration and technical-replicate assay controls.

The first composite reproduces the Hia5 amount and reaction-time controls from
anonymous per-read m6A percentages and aggregate footprint-size histograms.
The second compares aggregate footprint and accessible-patch distributions
between two technical replicates.  The bundled tables contain no read names,
sample barcodes, genomic coordinates, or sequence records.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from analysis.shared.paths import RESULTS_ROOT, SOURCE_DATA_ROOT
from analysis.shared.plotting import save_figure


CHROMATIN_REFERENCES = (
    ("sperm (high motility, pooled 4 donors)", 22.6, "#8B4B8B"),
    ("GM12878 (somatic)", 17.5, "#1F77B4"),
)


def _read_table(path: Path, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks required columns: {sorted(missing)}")
    return frame


def _condition_order(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame[["condition_value", "condition_label"]]
        .drop_duplicates()
        .sort_values("condition_value")
        .reset_index(drop=True)
    )


def _histogram_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    cohort: str,
    title: str,
) -> None:
    subset = frame.loc[frame["cohort"] == cohort]
    order = _condition_order(subset)
    colors = plt.get_cmap("viridis")(
        np.linspace(0.08, 0.88, len(order))
    )
    for color, condition in zip(colors, order.itertuples(index=False)):
        values = subset.loc[
            subset["condition_value"] == condition.condition_value
        ].sort_values("bin_left_log10")
        edges = np.r_[
            values["bin_left_log10"].to_numpy(),
            values["bin_right_log10"].iloc[-1],
        ]
        counts = values["count"].to_numpy()
        n_total = int(values["n_total"].iloc[0])
        ax.stairs(
            counts,
            edges,
            color=color,
            linewidth=1.25,
            label=f"{condition.condition_label} (n={n_total:,})",
        )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("log10 footprint size (bp) (≥ 100 bp)")
    ax.set_ylabel("count per bin")
    ax.legend(frameon=False, fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)


def _significance_label(value: float) -> str:
    if not np.isfinite(value) or value >= 0.05:
        return "ns"
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    return "*"


def _m6a_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    cohort: str,
    xlabel: str,
) -> None:
    subset = frame.loc[frame["cohort"] == cohort].copy()
    order = _condition_order(subset)
    labels = order["condition_label"].tolist()
    values = [
        subset.loc[
            subset["condition_value"] == condition.condition_value, "m6a_pct"
        ].to_numpy()
        for condition in order.itertuples(index=False)
    ]
    set2 = list(plt.get_cmap("Set2").colors)
    artists = ax.boxplot(
        values,
        positions=np.arange(len(values)),
        widths=0.62,
        patch_artist=True,
        showfliers=False,
        tick_labels=labels,
        medianprops={"color": "#666666", "linewidth": 0.8},
        whiskerprops={"color": "#777777", "linewidth": 0.7},
        capprops={"color": "#777777", "linewidth": 0.7},
        boxprops={"edgecolor": "#777777", "linewidth": 0.7},
    )
    for patch, color in zip(artists["boxes"], set2):
        patch.set_facecolor(color)

    sample = subset.sample(min(800, len(subset)), random_state=0)
    position_lookup = {
        float(condition.condition_value): index
        for index, condition in enumerate(order.itertuples(index=False))
    }
    positions = sample["condition_value"].map(position_lookup).to_numpy(dtype=float)
    jitter = np.random.default_rng(0).uniform(-0.22, 0.22, len(sample))
    ax.scatter(
        positions + jitter,
        sample["m6a_pct"],
        color="#333333",
        s=1.4,
        alpha=0.17,
        linewidths=0,
        rasterized=True,
    )

    comparisons = len(values) - 1
    for index in range(comparisons):
        p_value = float(
            mannwhitneyu(values[index], values[index + 1], alternative="two-sided").pvalue
        )
        adjusted = min(1.0, p_value * comparisons)
        bracket_y = 76.0
        ax.plot(
            [index + 0.08, index + 0.92],
            [bracket_y, bracket_y],
            color="#777777",
            linewidth=0.7,
        )
        ax.text(
            index + 0.5,
            bracket_y + 0.9,
            _significance_label(adjusted),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for label, value, color in CHROMATIN_REFERENCES:
        ax.axhline(
            value,
            color=color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label=f"{label} — p95 = {value:.1f}%",
        )
    ax.set_ylim(0, 82)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("per-fiber m6A bases (% of read length)")
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        fontsize=7.2,
        frameon=True,
        ncol=1,
    )
    ax.spines[["top", "right"]].set_visible(False)


def plot_hia5_titration(
    per_read: pd.DataFrame,
    histograms: pd.DataFrame,
    output_stem: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))
    _histogram_panel(axes[0, 0], histograms, "dose", "Hia5 amount")
    _m6a_panel(axes[0, 1], per_read, "dose", "Hia5 added (µL)")
    _histogram_panel(axes[1, 0], histograms, "time", "Hia5 incubation time")
    _m6a_panel(axes[1, 1], per_read, "time", "Hia5 reaction time (min)")
    for label, ax in zip("ABCD", axes.flat):
        ax.text(
            -0.12,
            1.04,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
        )
    fig.tight_layout(h_pad=2.4, w_pad=2.2)
    save_figure(fig, output_stem, dpi=250)
    plt.close(fig)


def plot_technical_replicates(frame: pd.DataFrame, output_stem: Path) -> None:
    metrics = (
        ("footprint_size", "Footprint sizes", "log10 footprint size (bp) (≥ 100 bp)"),
        (
            "accessible_patch_size",
            "Accessible-patch sizes",
            "log10 accessible-patch size (bp) (≥ 100 bp)",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    colors = ("#355DA8", "#159A9C")
    for panel_label, ax, (metric, title, xlabel) in zip("AB", axes, metrics):
        subset = frame.loc[frame["metric"] == metric]
        order = _condition_order(subset)
        for color, condition in zip(colors, order.itertuples(index=False)):
            values = subset.loc[
                subset["condition_value"] == condition.condition_value
            ].sort_values("bin_left_log10")
            edges = np.r_[
                values["bin_left_log10"].to_numpy(),
                values["bin_right_log10"].iloc[-1],
            ]
            n_total = int(values["n_total"].iloc[0])
            ax.stairs(
                values["count"].to_numpy(),
                edges,
                color=color,
                linewidth=1.25,
                label=f"{condition.condition_label} (n={n_total:,})",
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count per bin")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            -0.10,
            1.03,
            panel_label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
        )
    fig.tight_layout(w_pad=2.3)
    save_figure(fig, output_stem, dpi=250)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source_root = SOURCE_DATA_ROOT / "assay_controls"
    parser.add_argument(
        "--m6a-input",
        type=Path,
        default=source_root / "hia5_m6a_per_read.tsv.gz",
    )
    parser.add_argument(
        "--hia5-histograms",
        type=Path,
        default=source_root / "hia5_footprint_histograms.tsv",
    )
    parser.add_argument(
        "--technical-histograms",
        type=Path,
        default=source_root / "technical_replicate_histograms.tsv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_ROOT / "assay_controls",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_read = _read_table(
        args.m6a_input,
        {"cohort", "condition_value", "condition_label", "m6a_pct"},
    )
    hia5_histograms = _read_table(
        args.hia5_histograms,
        {
            "cohort",
            "condition_value",
            "condition_label",
            "bin_left_log10",
            "bin_right_log10",
            "count",
            "n_total",
        },
    )
    technical_histograms = _read_table(
        args.technical_histograms,
        {
            "condition_value",
            "condition_label",
            "metric",
            "bin_left_log10",
            "bin_right_log10",
            "count",
            "n_total",
        },
    )
    plot_hia5_titration(
        per_read,
        hia5_histograms,
        args.output_dir / "hia5_titration",
    )
    plot_technical_replicates(
        technical_histograms,
        args.output_dir / "technical_replicates",
    )
    print(
        "wrote hia5_titration.png/.pdf and technical_replicates.png/.pdf",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
