#!/usr/bin/env python3
"""Visualize the testis-to-sperm transition embedding.

The bundled table contains deidentified, read-level model outputs: two UMAP
coordinates, the six-state assignment, and chromatin fractions.  Model
training and inference are maintained in the companion FiberFormer repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.shared.paths import RESULTS_ROOT, SOURCE_DATA_ROOT
from analysis.shared.plotting import save_figure


STATE_COLORS = ["#265de6", "#2679b7", "#2c9780", "#55a33e", "#a47d20", "#cc5500"]
CONTINUOUS_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "chromatin_transition", ["#265de6", "#33a040", "#cc5500"]
)


def embedding_limits(data: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min, x_max = np.percentile(data["umap_1"], [0.5, 99.5])
    y_min, y_max = np.percentile(data["umap_2"], [0.5, 99.5])
    x_pad = 0.10 * (x_max - x_min)
    y_pad = 0.10 * (y_max - y_min)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def plot_states(data: pd.DataFrame, output_stem: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data))
    x_limits, y_limits = embedding_limits(data)

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    colors = np.asarray(STATE_COLORS, dtype=object)[data["state"].to_numpy(dtype=int)]
    ax.scatter(
        data["umap_1"].to_numpy()[order],
        data["umap_2"].to_numpy()[order],
        c=colors[order],
        s=1.2,
        alpha=0.55,
        linewidths=0,
        rasterized=True,
    )
    for state in sorted(data["state"].unique()):
        subset = data.loc[data["state"] == state, ["umap_1", "umap_2"]]
        center = subset.median()
        ax.text(
            center["umap_1"],
            center["umap_2"],
            f"S{int(state)}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": STATE_COLORS[int(state)]},
        )
    ax.set(xlim=x_limits, ylim=y_limits, xlabel="UMAP 1", ylabel="UMAP 2")
    ax.set_title("Testis chromatin transition states", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output_stem)
    plt.close(fig)


def plot_continuous(
    data: pd.DataFrame,
    column: str,
    label: str,
    output_stem: Path,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data))
    x_limits, y_limits = embedding_limits(data)

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    points = ax.scatter(
        data["umap_1"].to_numpy()[order],
        data["umap_2"].to_numpy()[order],
        c=data[column].to_numpy()[order],
        cmap=CONTINUOUS_CMAP,
        vmin=0,
        vmax=1,
        s=1.2,
        alpha=0.60,
        linewidths=0,
        rasterized=True,
    )
    ax.set(xlim=x_limits, ylim=y_limits, xlabel="UMAP 1", ylabel="UMAP 2")
    ax.set_title(label, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    colorbar = fig.colorbar(points, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label(label)
    fig.tight_layout()
    save_figure(fig, output_stem)
    plt.close(fig)


def summarize_trajectory(
    data: pd.DataFrame, n_bins: int, min_bin_reads: int
) -> pd.DataFrame:
    """Summarize chromatin fractions in equal-width normalized-time bins."""
    pseudotime = data["pseudotime"].to_numpy(dtype=float)
    minimum = float(np.nanmin(pseudotime))
    maximum = float(np.nanmax(pseudotime))
    normalized = (pseudotime - minimum) / (maximum - minimum)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.searchsorted(edges[1:-1], normalized, side="right")

    rows: list[dict[str, float | int]] = []
    for index in range(n_bins):
        subset = data.loc[bin_index == index]
        if len(subset) < min_bin_reads:
            continue
        row: dict[str, float | int] = {
            "bin": index + 1,
            "pseudotime_start": edges[index],
            "pseudotime_end": edges[index + 1],
            "pseudotime_center": 0.5 * (edges[index] + edges[index + 1]),
            "n_reads": len(subset),
        }
        metrics = {
            "protamine": subset["protamine_fraction"].to_numpy(dtype=float),
            "nucleosome_eviction": 1.0 - subset[
                "nucleosome_fraction_of_non_protamine"
            ].to_numpy(dtype=float),
        }
        for prefix, values in metrics.items():
            row[f"{prefix}_q25"] = float(np.nanquantile(values, 0.25))
            row[f"{prefix}_median"] = float(np.nanmedian(values))
            row[f"{prefix}_q75"] = float(np.nanquantile(values, 0.75))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_trajectory(summary: pd.DataFrame, output_stem: Path) -> None:
    """Plot median and interquartile range along the transition trajectory."""
    fig, axis = plt.subplots(figsize=(8.8, 4.8))
    x = summary["pseudotime_center"].to_numpy(dtype=float)
    for prefix, label, color in (
        ("protamine", "Protamination", "#cc5500"),
        (
            "nucleosome_eviction",
            "Nucleosome eviction from non-protamine sequence",
            "#33a040",
        ),
    ):
        median = summary[f"{prefix}_median"].to_numpy(dtype=float)
        q25 = summary[f"{prefix}_q25"].to_numpy(dtype=float)
        q75 = summary[f"{prefix}_q75"].to_numpy(dtype=float)
        axis.fill_between(x, q25, q75, color=color, alpha=0.22, linewidth=0)
        axis.plot(x, median, color=color, linewidth=2, marker="o", markersize=4,
                  label=label)
    axis.set(
        xlabel="Pseudotime (normalized to observed testis range)",
        ylabel="Fraction (median and interquartile range)",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.set_title("Chromatin transition trajectory", loc="left")
    axis.legend(frameon=False, loc="lower right")
    axis.grid(alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output_stem)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=SOURCE_DATA_ROOT / "testis_transition" / "transition_embedding.tsv.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_ROOT / "testis_transition",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trajectory-bins", type=int, default=20)
    parser.add_argument("--trajectory-min-bin-reads", type=int, default=20)
    args = parser.parse_args()

    data = pd.read_csv(args.input, sep="\t")
    required = {
        "umap_1",
        "umap_2",
        "state",
        "protamine_fraction",
        "nucleosome_fraction_of_non_protamine",
        "pseudotime",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_states(data, args.output_dir / "transition_states", args.seed)
    plot_continuous(
        data,
        "protamine_fraction",
        "Protamine fraction of read",
        args.output_dir / "protamine_fraction",
        args.seed,
    )
    plot_continuous(
        data,
        "nucleosome_fraction_of_non_protamine",
        "Nucleosome fraction of non-protamine sequence",
        args.output_dir / "nucleosome_fraction",
        args.seed,
    )
    trajectory = summarize_trajectory(
        data, args.trajectory_bins, args.trajectory_min_bin_reads
    )
    trajectory.to_csv(
        args.output_dir / "transition_trajectory.tsv", sep="\t", index=False
    )
    plot_trajectory(trajectory, args.output_dir / "transition_trajectory")
    print(f"wrote transition plots to {args.output_dir}")


if __name__ == "__main__":
    main()
