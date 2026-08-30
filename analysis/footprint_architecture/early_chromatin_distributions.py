#!/usr/bin/env python3
"""Plot the principal distributions describing sperm chromatin architecture.

The bundled-data mode draws four Figure 1 analyses from compact aggregate
tables.  Supplying a FiberHMM BED12 file rebuilds those tables from uniformly
sampled reads before plotting them.  No read names or genomic coordinates are
written to the derived tables.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.shared.paths import RESULTS_ROOT, SOURCE_DATA_ROOT
from analysis.shared.plotting import save_figure


LACUNA_FOOTPRINT_MAX = 250
PATCH_FOOTPRINT_MAX = 300
NUCLEOSOME_MIN = 90
NUCLEOSOME_MAX = 200
EDGE_PROTAMINE_MIN = 250
PROFILE_PROTAMINE_MIN = 1_000
EDGE_UPSTREAM = -100
EDGE_DOWNSTREAM = 5_000
N_EDGE_GROUPS = 40


def packed_blocks(row: object) -> list[int]:
    """Convert one BED12 row to alternating negative gaps/positive blocks."""
    starts = [int(value) for value in str(row.block_starts).rstrip(",").split(",")]
    sizes = [int(value) for value in str(row.block_sizes).rstrip(",").split(",")]
    read_length = int(row.read_end) - int(row.read_start)
    packed: list[int] = []
    previous_end = 0
    for start, size in zip(starts, sizes):
        gap = start - previous_end
        if gap:
            packed.append(-gap)
        if size:
            packed.append(size)
        previous_end = start + size
    trailing_gap = read_length - previous_end
    if trailing_gap:
        packed.append(-trailing_gap)
    return packed


def load_and_sample_bed(
    path: Path,
    n_distribution_reads: int,
    n_edge_reads: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Load BED12 calls, retain the longest alignment/read, and sample reads."""
    columns = ["chrom", "read_start", "read_end", "read_name", "block_starts", "block_sizes"]
    data = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3, 8, 9],
        names=columns,
        dtype={
            "chrom": "string",
            "read_name": "string",
            "block_starts": "string",
            "block_sizes": "string",
        },
    )
    data = data.loc[
        data["block_starts"].notna() & (data["block_starts"] != ".")
    ].copy()
    data["read_length"] = data["read_end"] - data["read_start"]
    data = (
        data.sort_values("read_length", ascending=False)
        .drop_duplicates("read_name", keep="first")
        .reset_index(drop=True)
    )
    n_available = len(data)
    if n_distribution_reads > n_available or n_edge_reads > n_available:
        raise ValueError(
            f"requested samples exceed {n_available:,} eligible reads in {path}"
        )
    distributions = data.sample(n=n_distribution_reads, random_state=seed)
    edges = data.sample(n=n_edge_reads, random_state=seed + 1)
    del data
    gc.collect()
    return distributions, edges, n_available


def _flush_patch(
    size_bp: int,
    nucleosome_count: int,
    weighted_sum: int,
    counter: Counter[tuple[int, int]],
) -> None:
    if size_bp and weighted_sum:
        counter[(size_bp // 100, nucleosome_count)] += 1


def summarize_distributions(
    sample: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate accessibility, lacuna sizes, and nucleosomes per lacuna."""
    accessible_reads = 0
    eligible_reads = 0
    lacuna_sizes: Counter[int] = Counter()
    nucleosomes_by_size: Counter[tuple[int, int]] = Counter()

    for processed, row in enumerate(sample.itertuples(index=False), start=1):
        packed = packed_blocks(row)

        accessible_bp = -sum(value for value in packed if value < 0)
        footprint_bp = sum(value for value in packed if value > 0)
        if accessible_bp + footprint_bp > 0 and footprint_bp > 1:
            eligible_reads += 1
            accessible_reads += int(accessible_bp > 0)

        current_lacuna = 0
        for value in packed:
            if value <= LACUNA_FOOTPRINT_MAX:
                current_lacuna += abs(value)
            elif current_lacuna:
                lacuna_sizes[current_lacuna] += 1
                current_lacuna = 0
        if current_lacuna:
            lacuna_sizes[current_lacuna] += 1

        patch_size = 0
        nucleosome_count = 0
        weighted_sum = 0
        previous_value: int | None = None
        for value in packed:
            if value <= PATCH_FOOTPRINT_MAX:
                weighted_sum += value * abs(value)
                if value != previous_value:
                    patch_size += abs(value)
                    if NUCLEOSOME_MIN < value <= LACUNA_FOOTPRINT_MAX:
                        nucleosome_count += 1
                previous_value = value
            else:
                _flush_patch(
                    patch_size,
                    nucleosome_count,
                    weighted_sum,
                    nucleosomes_by_size,
                )
                patch_size = 0
                nucleosome_count = 0
                weighted_sum = 0
                previous_value = None
        _flush_patch(
            patch_size,
            nucleosome_count,
            weighted_sum,
            nucleosomes_by_size,
        )

        if processed % 100_000 == 0:
            print(f"processed {processed:,} distribution reads", flush=True)

    proportion = accessible_reads / eligible_reads
    bootstrap_rng = np.random.default_rng(42)
    bootstrap = (
        bootstrap_rng.binomial(eligible_reads, proportion, size=100_000)
        / eligible_reads
    )
    lower, upper = np.percentile(bootstrap, [2.5, 97.5])
    accessibility = pd.DataFrame(
        [
            {
                "category": "one_or_more_lacunae",
                "n_reads": accessible_reads,
                "n_reads_total": eligible_reads,
                "pct_reads": 100 * proportion,
                "ci95_low": 100 * lower,
                "ci95_high": 100 * upper,
                "sample_seed": seed,
            },
            {
                "category": "fully_inaccessible",
                "n_reads": eligible_reads - accessible_reads,
                "n_reads_total": eligible_reads,
                "pct_reads": 100 * (1 - proportion),
                "ci95_low": 100 * (1 - upper),
                "ci95_high": 100 * (1 - lower),
                "sample_seed": seed,
            },
        ]
    )
    lacuna = pd.DataFrame(
        [
            {
                "lacuna_size_bp": size,
                "n_lacunae": count,
                "n_reads_sampled": len(sample),
                "sample_seed": seed,
            }
            for size, count in sorted(lacuna_sizes.items())
        ]
    )
    nucleosomes = pd.DataFrame(
        [
            {
                "lacuna_size_bin_100bp": size_bin,
                "nucleosome_footprints": count,
                "n_lacunae": frequency,
                "n_reads_sampled": len(sample),
                "sample_seed": seed,
            }
            for (size_bin, count), frequency in sorted(nucleosomes_by_size.items())
        ]
    )
    return accessibility, lacuna, nucleosomes


def _trim_outer_footprints(packed: list[int]) -> list[int]:
    negative = [index for index, value in enumerate(packed) if value < 0]
    if not negative:
        return []
    return packed[negative[0] : negative[-1] + 1]


def _window_runs(
    packed: list[int],
    window_start: int,
    window_stop: int,
) -> list[tuple[int, int, int]]:
    """Return (relative start, relative stop, packed value) within a window."""
    runs: list[tuple[int, int, int]] = []
    position = 0
    for value in packed:
        run_stop = position + abs(value)
        overlap_start = max(position, window_start)
        overlap_stop = min(run_stop, window_stop)
        if overlap_start < overlap_stop:
            runs.append(
                (overlap_start - window_start, overlap_stop - window_start, value)
            )
        position = run_stop
        if position >= window_stop:
            break
    return runs


def _distance_to_next_long_footprint(
    runs: list[tuple[int, int, int]],
    window_length: int,
) -> int:
    state_runs: list[tuple[bool, int]] = []
    for start, stop, value in runs:
        is_long = PROFILE_PROTAMINE_MIN < value < 1_000_000
        run_length = stop - start
        if state_runs and state_runs[-1][0] == is_long:
            previous_state, previous_length = state_runs[-1]
            state_runs[-1] = (previous_state, previous_length + run_length)
        else:
            state_runs.append((is_long, run_length))
    if len(state_runs) <= 1:
        return EDGE_DOWNSTREAM
    if not state_runs[0][0]:
        return state_runs[0][1]
    return state_runs[1][1]


def summarize_protamine_edges(sample: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Reproduce the notebook's distance-sorted protamine-edge metaprofile."""
    window_length = EDGE_DOWNSTREAM - EDGE_UPSTREAM
    distances: list[int] = []
    nucleosome_windows: list[np.ndarray] = []
    protamine_windows: list[np.ndarray] = []

    for processed, row in enumerate(sample.itertuples(index=False), start=1):
        packed = packed_blocks(row)
        if len(packed) > 2:
            packed = _trim_outer_footprints(packed)
        if not packed:
            continue

        boundaries = np.cumsum(np.abs(packed))
        read_length = int(boundaries[-1])
        edge_positions = [
            int(boundaries[index])
            for index in range(len(packed) - 1)
            if packed[index] >= EDGE_PROTAMINE_MIN
            and packed[index + 1] < EDGE_PROTAMINE_MIN
        ]
        for edge in edge_positions:
            window_start = edge + EDGE_UPSTREAM
            window_stop = edge + EDGE_DOWNSTREAM
            if window_start < 0 or window_stop > read_length:
                continue
            runs = _window_runs(packed, window_start, window_stop)
            distance = _distance_to_next_long_footprint(runs, window_length)
            if not (EDGE_UPSTREAM + 200 < distance < EDGE_DOWNSTREAM - 200):
                continue

            nucleosome = np.zeros(window_length, dtype=np.uint8)
            protamine = np.zeros(window_length, dtype=np.uint8)
            for start, stop, value in runs:
                if NUCLEOSOME_MIN <= value <= NUCLEOSOME_MAX:
                    nucleosome[start:stop] = 1
                if value >= PROFILE_PROTAMINE_MIN:
                    protamine[start:stop] = 1
            distances.append(distance)
            nucleosome_windows.append(nucleosome)
            protamine_windows.append(protamine)

        if processed % 10_000 == 0:
            print(
                f"processed {processed:,} edge reads; retained {len(distances):,} edges",
                flush=True,
            )

    if len(distances) < N_EDGE_GROUPS:
        raise ValueError("too few qualifying protamine edges for the binned profile")
    order = np.argsort(np.asarray(distances), kind="stable")[::-1]
    nucleosome_matrix = np.stack(nucleosome_windows)
    protamine_matrix = np.stack(protamine_windows)
    chunk_size = len(order) // N_EDGE_GROUPS
    positions = np.arange(EDGE_UPSTREAM, EDGE_DOWNSTREAM)

    rows: list[pd.DataFrame] = []
    for group, chunk_start in enumerate(range(0, len(order), chunk_size)):
        selected = order[chunk_start : chunk_start + chunk_size]
        rows.append(
            pd.DataFrame(
                {
                    "distance_bin": group,
                    "position_bp": positions,
                    "nucleosome_enrichment": nucleosome_matrix[selected].mean(axis=0)
                    * 10,
                    "protamine_occupancy": protamine_matrix[selected].mean(axis=0),
                    "n_edges": len(selected),
                    "n_reads_sampled": len(sample),
                    "sample_seed": seed,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def write_source_tables(
    output_dir: Path,
    accessibility: pd.DataFrame,
    lacuna: pd.DataFrame,
    nucleosomes: pd.DataFrame,
    edges: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    accessibility.to_csv(output_dir / "read_accessibility.tsv", sep="\t", index=False)
    lacuna.to_csv(
        output_dir / "lacuna_size_distribution.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    nucleosomes.to_csv(
        output_dir / "nucleosomes_per_lacuna.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    edges.to_csv(
        output_dir / "protamine_edge_nucleosome_profile.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )


def plot_accessibility(data: pd.DataFrame, output_dir: Path) -> None:
    row = data.loc[data["category"] == "one_or_more_lacunae"].iloc[0]
    value = float(row["pct_reads"])
    errors = np.array(
        [[value - float(row["ci95_low"])], [float(row["ci95_high"]) - value]]
    )
    fig, axis = plt.subplots(figsize=(3.5, 4.2))
    axis.bar(
        ["≥1 lacuna"],
        [value],
        yerr=errors,
        color="#9fadb5",
        edgecolor="#38464e",
        linewidth=0.7,
        capsize=3,
    )
    axis.set_ylabel("Reads with one or more lacunae (%)")
    axis.set_ylim(0, 100)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output_dir / "read_accessibility")
    plt.close(fig)


def plot_lacuna_sizes(data: pd.DataFrame, output_dir: Path) -> None:
    subset = data.loc[data["lacuna_size_bp"] > 10**1.5]
    values = np.log10(subset["lacuna_size_bp"].to_numpy())
    weights = subset["n_lacunae"].to_numpy()
    density, edges = np.histogram(values, bins=100, weights=weights, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.fill_between(centers, density, step="mid", color="#83a5bd", alpha=0.82)
    axis.plot(centers, density, color="#456f89", linewidth=0.9)
    ticks_bp = np.array([30, 100, 320, 1_000, 3_200, 10_000])
    axis.set_xticks(np.log10(ticks_bp))
    axis.set_xticklabels(["30", "100", "320", "1,000", "3,200", "10,000"])
    axis.set_xlabel("Distance between protamine footprints (lacuna size, bp)")
    axis.set_ylabel("Density")
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output_dir / "lacuna_size_distribution")
    plt.close(fig)


def plot_nucleosomes_per_lacuna(data: pd.DataFrame, output_dir: Path) -> None:
    bins = list(range(6, 21))
    statistics: list[dict[str, object]] = []
    for size_bin in bins:
        subset = data.loc[data["lacuna_size_bin_100bp"] == size_bin]
        values = (
            subset["nucleosome_footprints"].to_numpy(dtype=float)
            / size_bin
            * 10
        )
        expanded = np.repeat(values, subset["n_lacunae"].to_numpy(dtype=int))
        q1, median, q3 = np.percentile(expanded, [25, 50, 75])
        iqr = q3 - q1
        within = expanded[(expanded >= q1 - 1.5 * iqr) & (expanded <= q3 + 1.5 * iqr)]
        statistics.append(
            {
                "label": str(size_bin * 100),
                "q1": q1,
                "med": median,
                "q3": q3,
                "whislo": within.min(),
                "whishi": within.max(),
                "fliers": [],
            }
        )
    fig, axis = plt.subplots(figsize=(8.2, 4.3))
    artists = axis.bxp(statistics, showfliers=False, patch_artist=True)
    colors = plt.cm.YlOrBr(np.linspace(0.12, 0.88, len(statistics)))
    for box, color in zip(artists["boxes"], colors):
        box.set_facecolor(color)
        box.set_edgecolor("#444444")
        box.set_linewidth(0.7)
    for key in ("whiskers", "caps", "medians"):
        for artist in artists[key]:
            artist.set_color("#444444")
            artist.set_linewidth(0.8)
    axis.set_xlabel("Lacuna size (bp)")
    axis.set_ylabel("Nucleosome footprints per kb within lacuna")
    axis.tick_params(axis="x", rotation=90)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output_dir / "nucleosomes_per_lacuna")
    plt.close(fig)


def plot_protamine_edge_profile(data: pd.DataFrame, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.0, 7.0))
    for group, subset in data.groupby("distance_bin", sort=True):
        x = subset["position_bp"].to_numpy()
        protamine = subset["protamine_occupancy"].to_numpy() + group
        nucleosome = subset["nucleosome_enrichment"].to_numpy() + group
        axis.fill_between(x, group, protamine, color="#cf6b2e", alpha=0.55, linewidth=0)
        axis.fill_between(x, group, nucleosome, color="#37a34a", alpha=0.52, linewidth=0)
        axis.plot(x, nucleosome, color="#16822b", linewidth=0.55)
    axis.set_xlim(0, 4_500)
    axis.set_xlabel("Position relative to edge of protamine footprint (bp)")
    axis.set_ylabel("Nucleosome footprint enrichment\n(binned by lacuna size, normalized)")
    axis.set_yticks([])
    axis.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output_dir / "protamine_edge_nucleosome_profile")
    plt.close(fig)


def plot_all(
    accessibility: pd.DataFrame,
    lacuna: pd.DataFrame,
    nucleosomes: pd.DataFrame,
    edges: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_accessibility(accessibility, output_dir)
    plot_lacuna_sizes(lacuna, output_dir)
    plot_nucleosomes_per_lacuna(nucleosomes, output_dir)
    plot_protamine_edge_profile(edges, output_dir)


def main() -> None:
    source_dir = SOURCE_DATA_ROOT / "footprint_architecture"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fiberhmm-bed", type=Path, help="upstream FiberHMM BED12 calls")
    parser.add_argument(
        "--accessibility-input",
        type=Path,
        default=source_dir / "read_accessibility.tsv",
    )
    parser.add_argument(
        "--lacuna-input",
        type=Path,
        default=source_dir / "lacuna_size_distribution.tsv.gz",
    )
    parser.add_argument(
        "--nucleosome-input",
        type=Path,
        default=source_dir / "nucleosomes_per_lacuna.tsv.gz",
    )
    parser.add_argument(
        "--edge-input",
        type=Path,
        default=source_dir / "protamine_edge_nucleosome_profile.tsv.gz",
    )
    parser.add_argument(
        "--derived-data-dir",
        type=Path,
        default=RESULTS_ROOT / "footprint_architecture" / "derived_data",
        help="destination for aggregate tables when --fiberhmm-bed is supplied",
    )
    parser.add_argument("--n-distribution-reads", type=int, default=1_000_000)
    parser.add_argument("--n-edge-reads", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_ROOT / "footprint_architecture",
    )
    args = parser.parse_args()

    if args.fiberhmm_bed:
        distributions, edge_reads, n_available = load_and_sample_bed(
            args.fiberhmm_bed,
            args.n_distribution_reads,
            args.n_edge_reads,
            args.seed,
        )
        print(f"sampled from {n_available:,} unique eligible reads", flush=True)
        accessibility, lacuna, nucleosomes = summarize_distributions(
            distributions, args.seed
        )
        del distributions
        gc.collect()
        edges = summarize_protamine_edges(edge_reads, args.seed + 1)
        write_source_tables(
            args.derived_data_dir,
            accessibility,
            lacuna,
            nucleosomes,
            edges,
        )
    else:
        accessibility = pd.read_csv(args.accessibility_input, sep="\t")
        lacuna = pd.read_csv(args.lacuna_input, sep="\t")
        nucleosomes = pd.read_csv(args.nucleosome_input, sep="\t")
        edges = pd.read_csv(args.edge_input, sep="\t")

    plot_all(accessibility, lacuna, nucleosomes, edges, args.output_dir)
    print(f"wrote early chromatin distributions to {args.output_dir}")


if __name__ == "__main__":
    main()
