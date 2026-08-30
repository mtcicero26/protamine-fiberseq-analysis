#!/usr/bin/env python3
"""Run or redraw preranked gene-set enrichment for promoter lacunae."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.shared.paths import RESULTS_ROOT, SOURCE_DATA_ROOT
from analysis.shared.plotting import save_figure
import matplotlib.pyplot as plt


DEFAULT_LIBRARIES = [
    "GO_Biological_Process_2023",
    "Reactome_2022",
    "MSigDB_Hallmark_2020",
]


def build_ranked_list(statistics: pd.DataFrame) -> pd.Series:
    """Build the signed −log10(P) ranking used for preranked GSEA."""
    genes = statistics.loc[statistics["biotype"] == "mRNA"].copy()
    genes = genes.sort_values("p_value").drop_duplicates("gene_id")
    effect = genes["log2_or"].to_numpy(dtype=float)
    sign = np.where(effect < 0, -1.0, 1.0)
    primary = -np.log10(np.clip(genes["p_value"], 1e-300, None)) * sign
    epsilon = 1e-3 * np.max(np.abs(primary)) / max(np.max(np.abs(effect)), 1e-12)
    ranking = pd.Series(primary + epsilon * effect, index=genes["gene_id"])
    return ranking.sort_values(ascending=False)


def run_prerank(statistics_path: Path, output_dir: Path) -> pd.DataFrame:
    """Recompute the combined GSEA table with the production parameters."""
    try:
        import gseapy as gp
    except ImportError as error:
        raise SystemExit("Install the optional 'gseapy' dependency to recompute GSEA") from error

    ranking = build_ranked_list(pd.read_csv(statistics_path, sep="\t"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[pd.DataFrame] = []
    for library in DEFAULT_LIBRARIES:
        result = gp.prerank(
            rnk=ranking,
            gene_sets=library,
            threads=4,
            permutation_num=1000,
            min_size=100,
            max_size=2000,
            seed=42,
            verbose=False,
            outdir=None,
        ).res2d.copy()
        result["library"] = library
        result.to_csv(output_dir / f"gene_set_enrichment_{library}.tsv", sep="\t", index=False)
        results.append(result)
    return pd.concat(results, ignore_index=True).sort_values("FDR q-val")


def plot_summary(data: pd.DataFrame, output_stem: Path, top_k: int) -> None:
    positive = data.loc[data["NES"] > 0].nsmallest(top_k, "FDR q-val").iloc[::-1]
    negative = data.loc[data["NES"] < 0].nsmallest(top_k, "FDR q-val").iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5.2, top_k * 0.42)))
    for ax, subset, color, title in (
        (axes[0], positive, "#d55e00", "Positive normalized enrichment score"),
        (axes[1], negative, "#0072b2", "Negative normalized enrichment score"),
    ):
        y = np.arange(len(subset))
        values = -np.log10(np.clip(subset["FDR q-val"], 1e-300, None))
        ax.barh(y, values, color=color, alpha=0.88)
        labels = [term[:58] + ("…" if len(term) > 58 else "") for term in subset["Term"]]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        for index, (_, row) in enumerate(subset.iterrows()):
            ax.text(values.iloc[index], index, f"  NES={row['NES']:+.2f}", va="center", fontsize=7)
        ax.set_xlabel("−log10(FDR q value)")
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Preranked gene-set enrichment from promoter lacuna statistics")
    fig.tight_layout()
    save_figure(fig, output_stem)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=SOURCE_DATA_ROOT / "promoter_regulation" / "gene_set_enrichment.tsv",
        help="precomputed combined GSEA summary",
    )
    parser.add_argument(
        "--gene-statistics",
        type=Path,
        help="optional gene-level statistics; supplying this reruns GSEA",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_ROOT / "promoter_regulation" / "gene_set_enrichment",
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.gene_statistics:
        data = run_prerank(args.gene_statistics, args.output.parent / "gsea_tables")
        data.to_csv(args.output.with_name(args.output.name + "_summary.tsv"), sep="\t", index=False)
    else:
        data = pd.read_csv(args.input, sep="\t")
    plot_summary(data, args.output, args.top_k)
    print(f"wrote {args.output}.png/.pdf")


if __name__ == "__main__":
    main()
