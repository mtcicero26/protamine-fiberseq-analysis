#!/usr/bin/env python3
"""Compare protamine blocks per read in wild-type and Prm2-knockout mice."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from analysis.shared.paths import RESULTS_ROOT, SOURCE_DATA_ROOT
from analysis.shared.plotting import save_figure
import matplotlib.pyplot as plt


SAMPLE_ORDER = ["WT_FS2", "WT_2", "KO_208", "KO_206"]
SAMPLE_COLORS = ["#7f7f7f", "#bdbdbd", "#cc5500", "#ff8c42"]


def significance_label(p_value: float) -> str:
    if p_value < 1e-4:
        return "****"
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 5e-2:
        return "*"
    return "n.s."


def add_bracket(ax, left: float, right: float, height: float, label: str) -> None:
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    tick = 0.02 * span
    ax.plot([left, left, right, right], [height, height + tick, height + tick, height],
            color="black", linewidth=0.8, clip_on=False)
    ax.text((left + right) / 2, height + tick + 0.01 * span, label,
            ha="center", va="bottom", fontsize=9)


def export_from_cache(
    cache_dir: Path,
    output: Path,
    protamine_min_bp: int,
    max_reads_per_sample: int | None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    genotypes = {"WT_FS2": "WT", "WT_2": "WT", "KO_208": "Prm2 KO", "KO_206": "Prm2 KO"}
    for sample in SAMPLE_ORDER:
        blocks = pd.read_csv(cache_dir / f"{sample}_reads.tsv", sep="\t", usecols=["block_sizes"])
        if max_reads_per_sample is not None:
            blocks = blocks.iloc[:max_reads_per_sample]

        def count(value: object) -> int:
            if not isinstance(value, str) or not value:
                return 0
            return int(np.count_nonzero(np.fromstring(value, sep=",", dtype=int) >= protamine_min_bp))

        values = blocks["block_sizes"].map(count)
        rows.append(pd.DataFrame({
            "sample": sample,
            "genotype": genotypes[sample],
            "sample_read_index": np.arange(len(values)),
            "protamine_blocks": values,
        }))
    data = pd.concat(rows, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, sep="\t", index=False, compression="gzip")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=SOURCE_DATA_ROOT / "fertility" / "mouse_protamine_blocks_per_read.tsv.gz",
    )
    parser.add_argument("--cache-dir", type=Path,
                        help="optional per-read cache; supplying it rebuilds --input")
    parser.add_argument("--protamine-min-bp", type=int, default=200)
    parser.add_argument(
        "--max-reads-per-sample",
        type=int,
        default=3000,
        help="deterministic per-sample cap used with --cache-dir (default: 3000)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=RESULTS_ROOT / "fertility" / "mouse_protamine_blocks",
    )
    args = parser.parse_args()

    if args.max_reads_per_sample is not None and args.max_reads_per_sample < 1:
        parser.error("--max-reads-per-sample must be positive")
    if args.cache_dir:
        data = export_from_cache(
            args.cache_dir,
            args.input,
            args.protamine_min_bp,
            args.max_reads_per_sample,
        )
    else:
        data = pd.read_csv(args.input, sep="\t")
    counts = {
        sample: data.loc[data["sample"] == sample, "protamine_blocks"].to_numpy()
        for sample in SAMPLE_ORDER
    }

    p_wt = mannwhitneyu(counts["WT_FS2"], counts["WT_2"], alternative="two-sided").pvalue
    p_ko = mannwhitneyu(counts["KO_208"], counts["KO_206"], alternative="two-sided").pvalue
    pooled_wt = np.concatenate([counts["WT_FS2"], counts["WT_2"]])
    pooled_ko = np.concatenate([counts["KO_208"], counts["KO_206"]])
    p_between = mannwhitneyu(pooled_wt, pooled_ko, alternative="two-sided").pvalue

    positions = [0.0, 0.7, 1.9, 2.6]
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    boxplot = ax.boxplot(
        [counts[sample] for sample in SAMPLE_ORDER],
        positions=positions,
        widths=0.55,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
    )
    for box, color in zip(boxplot["boxes"], SAMPLE_COLORS):
        box.set(facecolor=color, edgecolor="black", linewidth=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(SAMPLE_ORDER)
    ax.set_ylabel("Protamine blocks per read")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    upper = max(np.percentile(values, 95) for values in counts.values()) * 1.38
    ax.set_ylim(0, upper)
    add_bracket(ax, positions[0], positions[1], upper * 0.67, significance_label(p_wt))
    add_bracket(ax, positions[2], positions[3], upper * 0.67, significance_label(p_ko))
    add_bracket(ax, np.mean(positions[:2]), np.mean(positions[2:]), upper * 0.88,
                significance_label(p_between))
    fig.tight_layout()
    save_figure(fig, args.output)
    plt.close(fig)
    print(f"within WT P={p_wt:.4g}; within KO P={p_ko:.4g}; pooled P={p_between:.4g}")
    print(f"wrote {args.output}.png/.pdf")


if __name__ == "__main__":
    main()
