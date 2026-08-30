#!/usr/bin/env python3
"""Run the bundled-data reproduction workflows."""

from __future__ import annotations

import argparse
import subprocess
import sys


WORKFLOWS = {
    "early_chromatin": "analysis.footprint_architecture.early_chromatin_distributions",
    "footprint_architecture": "analysis.footprint_architecture.nucleosome_phasing",
    "testis_transition": "analysis.testis_transition.transition_embedding",
    "promoter_lacunae": "analysis.promoter_regulation.tss_lacuna_enrichment",
    "go_enrichment": "analysis.promoter_regulation.go_enrichment",
    "gene_set_enrichment": "analysis.promoter_regulation.gene_set_enrichment",
    "promoter_chromatin": "analysis.regulatory_elements.promoter_fire_histone_overlap",
    "motif_retention": "analysis.regulatory_elements.motif_retention",
    "genome_context": "analysis.genome_context.region_occupancy",
    "mouse_prm2": "analysis.fertility.mouse_protamine_blocks",
    "motility_correlations": "analysis.fertility.motility_correlations",
    "motility_stability": "analysis.fertility.motility_stability",
    "assay_controls": "analysis.assay_controls.titration_and_replicates",
    "assay_periodicity": "analysis.assay_validation.dafseq_periodicity",
    "read_heatmaps": "analysis.read_heatmaps.footprint_heatmaps",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(WORKFLOWS),
        help="run only the named workflows; default is all",
    )
    parser.add_argument("--list", action="store_true", help="list workflow names and exit")
    args = parser.parse_args()

    if args.list:
        for name in WORKFLOWS:
            print(name)
        return

    selected = args.only or list(WORKFLOWS)
    failures: list[str] = []
    for index, name in enumerate(selected, start=1):
        module = WORKFLOWS[name]
        print(f"[{index}/{len(selected)}] {name}", flush=True)
        result = subprocess.run([sys.executable, "-m", module], check=False)
        if result.returncode:
            failures.append(name)
    if failures:
        raise SystemExit(f"failed workflows: {', '.join(failures)}")
    print(f"completed {len(selected)} workflows")


if __name__ == "__main__":
    main()
