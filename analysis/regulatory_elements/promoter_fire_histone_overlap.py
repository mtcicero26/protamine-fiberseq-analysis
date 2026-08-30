#!/usr/bin/env python3
"""Quantify promoter overlap with FIREs and sperm/testis histone marks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.shared.paths import RESULTS_ROOT, SOURCE_DATA_ROOT
from analysis.shared.plotting import save_figure
import matplotlib.pyplot as plt


MARK_ORDER = [
    "in_K4me3_sperm",
    "in_K27me3_sperm",
    "in_bivalent_curated",
    "in_K27ac_RS",
    "in_K27ac_sperm",
]
MARK_LABELS = {
    "in_K4me3_sperm": "H3K4me3 sperm",
    "in_K27me3_sperm": "H3K27me3 sperm",
    "in_bivalent_curated": "Bivalent",
    "in_K27ac_RS": "H3K27ac round spermatid",
    "in_K27ac_sperm": "H3K27ac sperm",
}
CATEGORY_ORDER = ["FIRE × mark", "FIRE only", "mark only", "neither"]
CATEGORY_COLORS = {
    "FIRE × mark": "#d55e00",
    "FIRE only": "#0072b2",
    "mark only": "#8a5aa5",
    "neither": "#b8b8b8",
}
STANDARD_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}


def load_bed(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
    )


def interval_index(bed: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for chrom, rows in bed.groupby("chrom"):
        rows = rows.sort_values("start")
        index[chrom] = (rows["start"].to_numpy(), rows["end"].to_numpy())
    return index


def overlaps(
    chroms: pd.Series,
    starts: pd.Series,
    ends: pd.Series,
    intervals: dict[str, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    hits = np.zeros(len(chroms), dtype=bool)
    queries = pd.DataFrame({"chrom": chroms, "start": starts, "end": ends})
    for chrom, rows in queries.groupby("chrom"):
        if chrom not in intervals:
            continue
        interval_starts, interval_ends = intervals[chrom]
        for row_index, query in rows.iterrows():
            stop = np.searchsorted(interval_starts, query["end"], side="left")
            hits[row_index] = stop > 0 and np.any(interval_ends[:stop] > query["start"])
    return hits


def compute_summary(
    statistics_path: Path,
    fire_path: Path,
    mark_paths: dict[str, Path],
    promoter_flank: int = 2000,
    q_threshold: float = 0.05,
) -> pd.DataFrame:
    genes = pd.read_csv(statistics_path, sep="\t")
    genes = genes.loc[genes["chrom"].isin(STANDARD_CHROMS)].reset_index(drop=True)
    genes["promoter_start"] = (genes["tss"] - promoter_flank).clip(lower=0)
    genes["promoter_end"] = genes["tss"] + promoter_flank
    genes["volcano_category"] = np.where(
        (genes["q_value"] < q_threshold) & (genes["log2_or"] > 0),
        "enriched",
        "not_sig",
    )

    fire = load_bed(fire_path)
    genes["fire"] = overlaps(
        genes["chrom"],
        genes["promoter_start"],
        genes["promoter_end"],
        interval_index(fire),
    )

    output: list[dict[str, object]] = []
    for mark in MARK_ORDER:
        mark_bed = load_bed(mark_paths[mark])
        genes["mark"] = overlaps(
            genes["chrom"],
            genes["promoter_start"],
            genes["promoter_end"],
            interval_index(mark_bed),
        )
        genes["chromatin_category"] = np.select(
            [
                genes["fire"] & genes["mark"],
                genes["fire"] & ~genes["mark"],
                ~genes["fire"] & genes["mark"],
            ],
            CATEGORY_ORDER[:3],
            default="neither",
        )
        counts = pd.crosstab(genes["volcano_category"], genes["chromatin_category"])
        counts = counts.reindex(index=["enriched", "not_sig"], columns=CATEGORY_ORDER, fill_value=0)
        percentages = counts.div(counts.sum(axis=1), axis=0) * 100
        for volcano_category in counts.index:
            for chromatin_category in counts.columns:
                output.append(
                    {
                        "mark": mark,
                        "vol_cat": volcano_category,
                        "chrom_cat": chromatin_category,
                        "n": int(counts.loc[volcano_category, chromatin_category]),
                        "pct": float(percentages.loc[volcano_category, chromatin_category]),
                    }
                )
    return pd.DataFrame(output)


def plot_summary(data: pd.DataFrame, output_stem: Path) -> None:
    fig, axes = plt.subplots(1, len(MARK_ORDER), figsize=(16, 4.6), sharey=True)
    for axis, mark in zip(axes, MARK_ORDER):
        subset = data.loc[data["mark"] == mark]
        pivot = subset.pivot(index="vol_cat", columns="chrom_cat", values="pct")
        pivot = pivot.reindex(index=["enriched", "not_sig"], columns=CATEGORY_ORDER, fill_value=0)
        bottom = np.zeros(len(pivot))
        for category in CATEGORY_ORDER:
            values = pivot[category].to_numpy()
            axis.bar(
                np.arange(len(pivot)),
                values,
                bottom=bottom,
                width=0.62,
                color=CATEGORY_COLORS[category],
                edgecolor="black",
                linewidth=0.35,
                label=category,
            )
            bottom += values
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["lacuna enriched", "not significant"], rotation=30, ha="right")
        axis.set_title(MARK_LABELS[mark], fontsize=9, loc="left")
        axis.set_ylim(0, 100)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Promoters in chromatin category (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save_figure(fig, output_stem)
    plt.close(fig)


def parse_mark_paths(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        label, path = value.split("=", 1)
        parsed[label] = Path(path)
    missing = set(MARK_ORDER).difference(parsed)
    if missing:
        raise ValueError(f"Missing --mark-bed entries for: {sorted(missing)}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=SOURCE_DATA_ROOT / "regulatory_elements" /
                "promoter_fire_histone_overlap.tsv",
        help="precomputed overlap summary",
    )
    parser.add_argument("--gene-statistics", type=Path)
    parser.add_argument("--fire-bed", type=Path)
    parser.add_argument("--mark-bed", action="append", default=[], metavar="NAME=BED")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_ROOT / "regulatory_elements" /
                "promoter_fire_histone_overlap",
    )
    args = parser.parse_args()

    recompute = any((args.gene_statistics, args.fire_bed, args.mark_bed))
    if recompute:
        if not args.gene_statistics or not args.fire_bed:
            parser.error("recomputation requires --gene-statistics, --fire-bed, and five --mark-bed entries")
        data = compute_summary(
            args.gene_statistics,
            args.fire_bed,
            parse_mark_paths(args.mark_bed),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(args.output.with_name(args.output.name + "_data.tsv"), sep="\t", index=False)
    else:
        data = pd.read_csv(args.input, sep="\t")
    plot_summary(data, args.output)
    print(f"wrote {args.output}.png/.pdf")


if __name__ == "__main__":
    main()
