#!/usr/bin/env python3
"""Validate source tables and execute every bundled analysis workflow."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SOURCE_ROWS = {
    "assay_controls/hia5_footprint_histograms.tsv": 1_100,
    "assay_controls/hia5_m6a_per_read.tsv.gz": 33_000,
    "assay_controls/technical_replicate_histograms.tsv": 400,
    "assay_validation/fiberseq_autocorrelation.tsv": 5_410,
    "assay_validation/scdafseq_autocorrelation.tsv": 1_001,
    "fertility/motility_metrics.tsv": 13,
    "fertility/motility_stability.tsv": 600,
    "fertility/mouse_protamine_blocks_per_read.tsv.gz": 10_354,
    "footprint_architecture/lacuna_size_distribution.tsv.gz": 21_807,
    "footprint_architecture/nucleosome_phasing.tsv": 6_002,
    "footprint_architecture/nucleosomes_per_lacuna.tsv.gz": 4_057,
    "footprint_architecture/protamine_edge_nucleosome_profile.tsv.gz": 204_000,
    "footprint_architecture/read_accessibility.tsv": 2,
    "genome_context/region_occupancy.tsv": 9,
    "promoter_regulation/curated_go_enrichment.tsv": 14,
    "promoter_regulation/gene_set_enrichment.tsv": 559,
    "promoter_regulation/non_tss_empirical_null.tsv": 21,
    "promoter_regulation/tss_lacuna_gene_statistics.tsv": 42_107,
    "promoter_regulation/tss_lacuna_transcript_statistics.tsv": 127_751,
    "read_heatmaps/footprint_heatmap_intervals.tsv.gz": 57_912,
    "read_heatmaps/heatmap_manifest.tsv": 24,
    "read_heatmaps/heatmap_row_order.tsv.gz": 25_535,
    "regulatory_elements/motif_retention_effects.tsv": 163,
    "regulatory_elements/promoter_fire_histone_overlap.tsv": 40,
    "testis_transition/transition_embedding.tsv.gz": 173_141,
}

PLOT_STEMS = [
    "assay_controls/hia5_titration",
    "assay_controls/technical_replicates",
    "assay_validation/periodicity",
    "assay_validation/periodicity_zoom_200bp",
    "fertility/motility_correlations",
    "fertility/motility_stability",
    "fertility/mouse_protamine_blocks",
    "footprint_architecture/lacuna_size_distribution",
    "footprint_architecture/nucleosome_phasing",
    "footprint_architecture/nucleosomes_per_lacuna",
    "footprint_architecture/protamine_edge_nucleosome_profile",
    "footprint_architecture/read_accessibility",
    "genome_context/region_occupancy",
    "promoter_regulation/curated_go_enrichment",
    "promoter_regulation/gene_set_enrichment",
    "promoter_regulation/tss_lacuna_enrichment",
    "read_heatmaps/dafseq_chr22_heatmaps",
    "read_heatmaps/dafseq_chr5_heatmaps",
    "read_heatmaps/dafseq_mei4_heatmaps",
    "read_heatmaps/dafseq_uba1_heatmaps",
    "read_heatmaps/figure3_tss_heatmaps",
    "read_heatmaps/motility_heatmaps",
    "read_heatmaps/mouse_prm2_heatmaps",
    "regulatory_elements/motif_retention",
    "regulatory_elements/promoter_fire_histone_overlap",
    "testis_transition/nucleosome_fraction",
    "testis_transition/protamine_fraction",
    "testis_transition/transition_states",
    "testis_transition/transition_trajectory",
]


def fail(message: str) -> None:
    raise AssertionError(message)


def check_structure() -> None:
    forbidden_roots = {"notebooks", "exploratory", "working"}
    present = {path.name.lower() for path in ROOT.iterdir() if path.is_dir()}
    if overlap := forbidden_roots & present:
        fail(f"legacy top-level directories present: {sorted(overlap)}")
    notebooks = sorted(path.relative_to(ROOT) for path in ROOT.rglob("*.ipynb"))
    if notebooks:
        fail(f"notebooks present: {notebooks}")
    compiled = sorted(
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.name == "__pycache__" or path.suffix == ".pyc"
    )
    if compiled:
        fail(f"generated Python cache artifacts present: {compiled}")

    text_suffixes = {".cff", ".md", ".py", ".txt", ".yaml", ".yml"}
    host_path_markers = ("/" + "mnt/", "/" + "home/")
    legacy_name_markers = ("testes" + "_ml", "Revisions" + "/")
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in text_suffixes:
            continue
        source = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            compile(source, str(path), "exec")
        if any(marker in source for marker in host_path_markers):
            fail(f"absolute host path in {path.relative_to(ROOT)}")
        if any(marker in source for marker in legacy_name_markers):
            fail(f"legacy working-directory name in {path.relative_to(ROOT)}")


def load_source(relative: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "source" / relative, sep="\t")


def check_source_data() -> None:
    source_root = ROOT / "data" / "source"
    actual = {
        str(path.relative_to(source_root))
        for path in source_root.rglob("*")
        if path.is_file()
    }
    expected = set(EXPECTED_SOURCE_ROWS)
    if actual != expected:
        fail(
            "source-data inventory differs; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )

    for relative, expected_rows in EXPECTED_SOURCE_ROWS.items():
        observed = len(load_source(relative))
        if observed != expected_rows:
            fail(f"{relative}: expected {expected_rows:,} rows, found {observed:,}")

    transcripts = load_source(
        "promoter_regulation/tss_lacuna_transcript_statistics.tsv"
    )
    enriched = int(
        ((transcripts["q_value"] < 0.05) & (transcripts["log2_or"] > 0)).sum()
    )
    if enriched != 1_970 or transcripts["p_global"].nunique() != 1:
        fail("promoter-lacuna statistics failed release invariants")

    go = load_source("promoter_regulation/curated_go_enrichment.tsv")
    if int((go["q_value"] < 0.20).sum()) != 4:
        fail("curated GO table failed release invariants")

    transition = load_source("testis_transition/transition_embedding.tsv.gz")
    if set(transition["state"].astype(int)) != set(range(6)):
        fail("transition table does not contain the six expected states")

    mouse = load_source("fertility/mouse_protamine_blocks_per_read.tsv.gz")
    if set(mouse["sample"]) != {"WT_FS2", "WT_2", "KO_208", "KO_206"}:
        fail("mouse table contains an unexpected sample set")

    accessibility = load_source("footprint_architecture/read_accessibility.tsv")
    observed_accessibility = float(
        accessibility.loc[
            accessibility["category"] == "one_or_more_lacunae", "pct_reads"
        ].iloc[0]
    )
    if not np.isclose(observed_accessibility, 66.6545, rtol=0, atol=1e-10):
        fail("read-accessibility table failed release invariants")

    edge_profile = load_source(
        "footprint_architecture/protamine_edge_nucleosome_profile.tsv.gz"
    )
    if (
        edge_profile["distance_bin"].nunique() != 40
        or edge_profile["position_bp"].min() != -100
        or edge_profile["position_bp"].max() != 4_999
    ):
        fail("protamine-edge profile failed release invariants")

    heatmap_manifest = load_source("read_heatmaps/heatmap_manifest.tsv")
    expected_heatmap_dimensions = {
        "tss_depleted": (1_000, 4_000),
        "tss_enriched": (1_000, 4_000),
        "tss_asxl2": (23, 4_000),
        "tss_ino80d": (25, 4_000),
        "mouse_wt": (500, 5_000),
        "mouse_prm2_ko": (500, 5_000),
        "motility_2pct": (100, 5_000),
        "motility_1p2pct": (100, 5_000),
        "daf_chr5": (3_468, 4_654),
        "daf_chr22": (2_741, 4_061),
        "daf_uba1": (4_733, 4_300),
        "daf_mei4": (7_101, 4_500),
    }
    matrix_dimensions = {
        str(row.matrix_id): (int(row.matrix_n_rows), int(row.width_bp))
        for row in heatmap_manifest.drop_duplicates("matrix_id").itertuples(index=False)
    }
    if matrix_dimensions != expected_heatmap_dimensions:
        fail("read-heatmap matrix dimensions differ from the frozen release")
    expected_composite_counts = {
        "figure3_tss_heatmaps": 4,
        "mouse_prm2_heatmaps": 2,
        "motility_heatmaps": 2,
        "dafseq_chr5_heatmaps": 4,
        "dafseq_chr22_heatmaps": 4,
        "dafseq_uba1_heatmaps": 4,
        "dafseq_mei4_heatmaps": 4,
    }
    if heatmap_manifest.groupby("composite_id").size().to_dict() != expected_composite_counts:
        fail("read-heatmap composite inventory differs from the frozen release")
    heatmap_text = heatmap_manifest.to_csv(sep="\t", index=False)
    private_heatmap_labels = ("PS00754", "PS00755", "WT_2", "KO_208", "MEIS4")
    if any(label in heatmap_text for label in private_heatmap_labels):
        fail("read-heatmap manifest contains an internal sample or legacy label")

    hia5_m6a = load_source("assay_controls/hia5_m6a_per_read.tsv.gz")
    expected_hia5_conditions = {
        "dose": {0.0, 0.5, 1.0, 2.0, 4.0},
        "time": {5.0, 10.0, 15.0, 20.0, 25.0, 30.0},
    }
    observed_hia5_conditions = {
        cohort: set(group["condition_value"].astype(float))
        for cohort, group in hia5_m6a.groupby("cohort")
    }
    if observed_hia5_conditions != expected_hia5_conditions:
        fail("Hia5 control conditions differ from the frozen release")
    if set(hia5_m6a.groupby(["cohort", "condition_value"]).size()) != {3_000}:
        fail("Hia5 control table does not contain 3,000 reads per condition")
    private_control_columns = {"name", "sample", "barcode", "chrom", "start", "end"}
    if private_control_columns.intersection(hia5_m6a.columns):
        fail("Hia5 control table contains a private identifier or coordinate column")

    hia5_histograms = load_source("assay_controls/hia5_footprint_histograms.tsv")
    if set(hia5_histograms.groupby(["cohort", "condition_value"]).size()) != {100}:
        fail("Hia5 footprint histograms do not contain 100 bins per condition")
    expected_hia5_footprint_counts = {
        ("dose", 0.0): 7_029,
        ("dose", 0.5): 2_315,
        ("dose", 1.0): 1_630,
        ("dose", 2.0): 1_089,
        ("dose", 4.0): 2_015,
        ("time", 5.0): 6_386,
        ("time", 10.0): 2_968,
        ("time", 15.0): 2_718,
        ("time", 20.0): 1_505,
        ("time", 25.0): 1_190,
        ("time", 30.0): 1_640,
    }
    observed_hia5_footprint_counts = {
        (str(cohort), float(value)): int(group["n_total"].iloc[0])
        for (cohort, value), group in hia5_histograms.groupby(
            ["cohort", "condition_value"]
        )
    }
    if observed_hia5_footprint_counts != expected_hia5_footprint_counts:
        fail("Hia5 footprint counts differ from the frozen release")

    technical_histograms = load_source(
        "assay_controls/technical_replicate_histograms.tsv"
    )
    expected_technical_groups = {
        ("footprint_size", 1.0): 11_043,
        ("footprint_size", 2.0): 10_657,
        ("accessible_patch_size", 1.0): 1_877,
        ("accessible_patch_size", 2.0): 1_851,
    }
    observed_technical_groups = {
        (str(metric), float(value)): int(group["n_total"].iloc[0])
        for (metric, value), group in technical_histograms.groupby(
            ["metric", "condition_value"]
        )
    }
    if observed_technical_groups != expected_technical_groups:
        fail("technical-replicate size counts differ from the frozen release")
    if set(
        technical_histograms.groupby(["metric", "condition_value"]).size()
    ) != {100}:
        fail("technical-replicate histograms do not contain 100 bins per group")

    motility = load_source("fertility/motility_metrics.tsv")
    expected_correlations = {
        "prot_frac": -0.8071532805049727,
        "nuc_per_kb": 0.49560885653131215,
        "gap_frac": 0.8243388728588625,
    }
    for metric, expected_r in expected_correlations.items():
        observed_r = float(pearsonr(motility["motility"], motility[metric]).statistic)
        if not np.isclose(observed_r, expected_r, rtol=0, atol=1e-12):
            fail(f"{metric}: expected r={expected_r}, found {observed_r}")


def check_execution() -> float:
    with tempfile.TemporaryDirectory(prefix="protamine-analysis-") as temp_dir:
        output_root = Path(temp_dir) / "results"
        environment = os.environ.copy()
        environment.update(
            {
                "MPLBACKEND": "Agg",
                "MPLCONFIGDIR": str(Path(temp_dir) / "matplotlib"),
                "PROTAMINE_OUTPUT_ROOT": str(output_root),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        started = time.monotonic()
        result = subprocess.run(
            [sys.executable, "-m", "analysis.reproduce"],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        elapsed = time.monotonic() - started
        if result.returncode:
            print(result.stdout, file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            fail(f"bundled workflows exited with status {result.returncode}")

        expected_outputs = {
            f"{stem}.{extension}"
            for stem in PLOT_STEMS
            for extension in ("png", "pdf")
        }
        expected_outputs.add("testis_transition/transition_trajectory.tsv")
        actual_outputs = {
            str(path.relative_to(output_root))
            for path in output_root.rglob("*")
            if path.is_file()
        }
        if actual_outputs != expected_outputs:
            fail(
                "generated-output inventory differs; "
                f"missing={sorted(expected_outputs - actual_outputs)}, "
                f"unexpected={sorted(actual_outputs - expected_outputs)}"
            )
        empty = [
            path for path in output_root.rglob("*")
            if path.is_file() and path.stat().st_size < 100
        ]
        if empty:
            fail(f"empty generated outputs: {empty}")
        return elapsed


def main() -> int:
    try:
        check_structure()
        check_source_data()
        elapsed = check_execution()
    except (AssertionError, OSError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print(
        "Package check passed: 15 workflows, 25 compact source tables, "
        f"59 generated files ({elapsed:.1f} s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
