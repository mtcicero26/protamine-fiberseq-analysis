#!/usr/bin/env python3
"""Reconstruct and plot the bundled Fiber-seq and DAF-seq read heatmaps.

The compact input stores only nonzero footprint runs and anonymous row
indices.  Accessible bases are implicit zeros.  A separate row-order table
freezes each displayed selection, including the paired-donor DAF-seq cluster
views.  SHA-256 checks guard both the source matrices and displayed panels.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from analysis.shared.paths import SOURCE_DATA_ROOT, output_directory
from analysis.shared.plotting import save_figure


INPUT_ROOT = SOURCE_DATA_ROOT / "read_heatmaps"
SIZE_VMAX = 1_000
FOOTPRINT_COLORS = ["#265de6", "#33a040", "#33a040", "#CC5500"]
FOOTPRINT_BREAKS = [10, 90, 200, SIZE_VMAX]

COMPOSITE_TITLES = {
    "figure3_tss_heatmaps": "TSS-centered Fiber-seq footprint heatmaps",
    "mouse_prm2_heatmaps": "Mouse sperm footprint heatmaps",
    "motility_heatmaps": "Low-motility human sperm footprint heatmaps",
    "dafseq_chr5_heatmaps": "Chromosome 5 paired-donor DAF-seq footprint heatmaps",
    "dafseq_chr22_heatmaps": "Chromosome 22 paired-donor DAF-seq footprint heatmaps",
    "dafseq_uba1_heatmaps": "UBA1 paired-donor DAF-seq footprint heatmaps",
    "dafseq_mei4_heatmaps": "MEI4 paired-donor DAF-seq footprint heatmaps",
}

MANIFEST_COLUMNS = {
    "panel_id",
    "composite_id",
    "layout_row",
    "layout_col",
    "assay",
    "label",
    "matrix_id",
    "matrix_n_rows",
    "width_bp",
    "x_start_bp",
    "x_end_bp",
    "x_label",
    "n_rows",
    "center_bp",
    "show_cluster_bar",
    "view",
    "selection",
    "sort_rule",
    "matrix_sha256",
    "panel_sha256",
}

INTERVAL_COLUMNS = {
    "matrix_id", "source_row", "start_bp", "end_bp", "footprint_size_bp"
}
ORDER_COLUMNS = {"panel_id", "display_row", "source_row", "cluster_rank"}


def require_columns(frame: pd.DataFrame, expected: set[str], name: str) -> None:
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def matrix_hash(matrix: np.ndarray) -> str:
    payload = np.ascontiguousarray(matrix.astype("<u2", copy=False)).tobytes()
    return hashlib.sha256(payload).hexdigest()


def footprint_colormap() -> mcolors.ListedColormap:
    """Recreate the released 1,000-step FiberHMM footprint colormap."""
    colors = [np.asarray(mcolors.to_rgb(color)) for color in FOOTPRINT_COLORS]
    entries: list[tuple[float, float, float, float]] = [(0, 0, 0, 0)] * 10
    for index in range(len(colors) - 1):
        steps = FOOTPRINT_BREAKS[index + 1] - FOOTPRINT_BREAKS[index]
        start_h, start_l, start_s = colorsys.rgb_to_hls(*colors[index])
        end_h, end_l, end_s = colorsys.rgb_to_hls(*colors[index + 1])
        # The retained plots interpolate hue, saturation, and lightness.
        # Reproduce that operation directly to avoid an extra plotting-only
        # dependency in the public environment.
        segment = np.linspace(
            (start_h, start_s, start_l),
            (end_h, end_s, end_l),
            steps,
            endpoint=True,
        )
        entries.extend(
            (*colorsys.hls_to_rgb(hue, lightness, saturation), 1.0)
            for hue, saturation, lightness in segment
        )
    if len(entries) != SIZE_VMAX:
        raise AssertionError(f"colormap has {len(entries)} entries, expected {SIZE_VMAX}")
    colormap = mcolors.ListedColormap(entries, name="fiberhmm_footprint_size")
    colormap.set_bad("white")
    return colormap


def reconstruct_matrix(
    matrix_id: str,
    matrix_manifest: pd.DataFrame,
    intervals: pd.DataFrame,
) -> np.ndarray:
    row_counts = matrix_manifest["matrix_n_rows"].astype(int).unique()
    widths = matrix_manifest["width_bp"].astype(int).unique()
    hashes = matrix_manifest["matrix_sha256"].astype(str).unique()
    if len(row_counts) != 1 or len(widths) != 1 or len(hashes) != 1:
        raise ValueError(f"{matrix_id}: inconsistent matrix metadata across panels")
    n_rows, width = int(row_counts[0]), int(widths[0])
    matrix = np.zeros((n_rows, width), dtype=np.uint16)
    records = intervals.loc[intervals["matrix_id"] == matrix_id]
    if records.empty:
        raise ValueError(f"{matrix_id}: no footprint intervals")
    for record in records.itertuples(index=False):
        source_row = int(record.source_row)
        start = int(record.start_bp)
        end = int(record.end_bp)
        value = int(record.footprint_size_bp)
        if not (0 <= source_row < n_rows and 0 <= start < end <= width):
            raise ValueError(f"{matrix_id}: interval outside declared matrix bounds")
        if not (0 < value < SIZE_VMAX):
            raise ValueError(f"{matrix_id}: invalid displayed footprint size {value}")
        if np.any(matrix[source_row, start:end]):
            raise ValueError(f"{matrix_id}: overlapping footprint intervals")
        matrix[source_row, start:end] = value
    observed_hash = matrix_hash(matrix)
    if observed_hash != hashes[0]:
        raise ValueError(
            f"{matrix_id}: matrix hash mismatch ({observed_hash} != {hashes[0]})"
        )
    return matrix


def ordered_panel(
    panel: pd.Series,
    matrix: np.ndarray,
    row_orders: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    panel_id = str(panel["panel_id"])
    order = row_orders.loc[row_orders["panel_id"] == panel_id].sort_values(
        "display_row"
    )
    n_rows = int(panel["n_rows"])
    if len(order) != n_rows:
        raise ValueError(f"{panel_id}: expected {n_rows} row-order records, found {len(order)}")
    if not np.array_equal(order["display_row"].to_numpy(), np.arange(n_rows)):
        raise ValueError(f"{panel_id}: displayed rows are not contiguous from zero")
    source_rows = order["source_row"].astype(int).to_numpy()
    if len(np.unique(source_rows)) != len(source_rows):
        raise ValueError(f"{panel_id}: source rows are duplicated")
    if len(source_rows) and (source_rows.min() < 0 or source_rows.max() >= len(matrix)):
        raise ValueError(f"{panel_id}: source row outside matrix bounds")
    displayed = matrix[source_rows]
    observed_hash = matrix_hash(displayed)
    if observed_hash != str(panel["panel_sha256"]):
        raise ValueError(f"{panel_id}: displayed-panel hash mismatch")
    return displayed, order["cluster_rank"].astype(int).to_numpy()


def add_cluster_strip(ax, cluster_ranks: np.ndarray) -> None:
    if len(cluster_ranks) == 0 or np.any(cluster_ranks < 0):
        raise ValueError("clustered panel lacks nonnegative cluster ranks")
    transitions = np.flatnonzero(cluster_ranks[1:] != cluster_ranks[:-1]) + 1
    boundaries = np.concatenate(([0], transitions, [len(cluster_ranks)]))
    for boundary in transitions:
        ax.axhline(boundary, color="white", linewidth=0.4, alpha=0.85)

    divider = make_axes_locatable(ax)
    strip = divider.append_axes("left", size="2.5%", pad="1.2%")
    palette = plt.get_cmap("tab10")
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        rank = int(cluster_ranks[start])
        strip.add_patch(
            Rectangle(
                (0, start),
                1,
                end - start,
                facecolor=palette(rank % 10),
                edgecolor="none",
            )
        )
    strip.set_xlim(0, 1)
    strip.set_ylim(len(cluster_ranks), 0)
    strip.set_xticks([])
    strip.set_yticks([])
    for spine in strip.spines.values():
        spine.set_linewidth(0.5)


def paint_panel(
    ax,
    panel: pd.Series,
    matrix: np.ndarray,
    cluster_ranks: np.ndarray,
    colormap: mcolors.ListedColormap,
) -> None:
    n_rows = len(matrix)
    x_start, x_end = int(panel["x_start_bp"]), int(panel["x_end_bp"])
    display = np.ma.masked_equal(matrix, 0)
    ax.imshow(
        display,
        aspect="auto",
        interpolation="nearest",
        cmap=colormap,
        vmin=0,
        vmax=SIZE_VMAX,
        extent=[x_start, x_end, n_rows, 0],
    )
    if pd.notna(panel["center_bp"]):
        ax.axvline(
            int(panel["center_bp"]),
            color="black",
            linewidth=0.8,
            linestyle="--",
            alpha=0.5,
        )
    ax.set_xlim(x_start, x_end)
    ax.set_ylim(n_rows, 0)
    ax.set_xlabel(str(panel["x_label"]), fontsize=8)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=7)
    clustered = bool(int(panel["show_cluster_bar"]))
    if clustered:
        ax.set_ylabel("")
        short_label = str(panel["label"]).replace(
            "most-protaminated cluster omitted", "most-protaminated omitted"
        )
        title = f"{short_label} (n={n_rows:,})"
    else:
        ax.set_ylabel(f"reads (n={n_rows:,})", fontsize=8)
        title = str(panel["label"])
    ax.set_title(title, fontsize=9, loc="left")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    if clustered:
        add_cluster_strip(ax, cluster_ranks)


def figure_dimensions(n_rows: int, n_cols: int) -> tuple[float, float]:
    if n_cols == 4:
        return 16.0, 7.5
    if n_rows == 2:
        return 11.0, 8.0
    return 11.0, 6.0


def render_composite(
    composite_id: str,
    panels: pd.DataFrame,
    intervals: pd.DataFrame,
    row_orders: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    n_layout_rows = int(panels["layout_row"].max()) + 1
    n_layout_cols = int(panels["layout_col"].max()) + 1
    expected_positions = {
        (row, col) for row in range(n_layout_rows) for col in range(n_layout_cols)
    }
    observed_positions = {
        (int(row.layout_row), int(row.layout_col))
        for row in panels.itertuples(index=False)
    }
    if observed_positions != expected_positions:
        raise ValueError(f"{composite_id}: incomplete or duplicated panel layout")

    matrices = {
        matrix_id: reconstruct_matrix(
            matrix_id,
            panels.loc[panels["matrix_id"] == matrix_id],
            intervals,
        )
        for matrix_id in panels["matrix_id"].unique()
    }
    figure, axes = plt.subplots(
        n_layout_rows,
        n_layout_cols,
        figsize=figure_dimensions(n_layout_rows, n_layout_cols),
        squeeze=False,
    )
    colormap = footprint_colormap()
    for _, panel in panels.sort_values(["layout_row", "layout_col"]).iterrows():
        matrix, cluster_ranks = ordered_panel(
            panel, matrices[str(panel["matrix_id"])], row_orders
        )
        paint_panel(
            axes[int(panel["layout_row"]), int(panel["layout_col"])],
            panel,
            matrix,
            cluster_ranks,
            colormap,
        )
    figure.suptitle(COMPOSITE_TITLES.get(composite_id, composite_id), fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    outputs = save_figure(figure, output_dir / composite_id)
    plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=INPUT_ROOT / "heatmap_manifest.tsv"
    )
    parser.add_argument(
        "--intervals",
        type=Path,
        default=INPUT_ROOT / "footprint_heatmap_intervals.tsv.gz",
    )
    parser.add_argument(
        "--row-order",
        type=Path,
        default=INPUT_ROOT / "heatmap_row_order.tsv.gz",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--only",
        nargs="+",
        help="render only the named composite identifiers",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, sep="\t")
    intervals = pd.read_csv(args.intervals, sep="\t")
    row_orders = pd.read_csv(args.row_order, sep="\t")
    require_columns(manifest, MANIFEST_COLUMNS, "heatmap manifest")
    require_columns(intervals, INTERVAL_COLUMNS, "footprint intervals")
    require_columns(row_orders, ORDER_COLUMNS, "heatmap row order")
    if manifest["panel_id"].duplicated().any():
        raise ValueError("heatmap manifest contains duplicate panel identifiers")
    if row_orders.duplicated(["panel_id", "display_row"]).any():
        raise ValueError("heatmap row order contains duplicate displayed rows")

    available = list(dict.fromkeys(manifest["composite_id"].astype(str)))
    selected = args.only or available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise SystemExit(f"unknown composite identifiers: {', '.join(unknown)}")
    output_dir = args.output_dir or output_directory("read_heatmaps")
    output_dir.mkdir(parents=True, exist_ok=True)
    for composite_id in selected:
        outputs = render_composite(
            composite_id,
            manifest.loc[manifest["composite_id"] == composite_id],
            intervals,
            row_orders,
            output_dir,
        )
        print(f"wrote {', '.join(str(path) for path in outputs)}")


if __name__ == "__main__":
    main()
