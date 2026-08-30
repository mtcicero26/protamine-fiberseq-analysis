#!/usr/bin/env python3
"""Compute or redraw transcript-level promoter-lacuna enrichment.

The default command redraws the analysis from the bundled transcript-level
statistics.  Supplying ``--cache-dir`` recomputes per-transcript counts and
binomial tests from a per-read Parquet cache, using the bundled non-TSS
empirical background unless another null is requested explicitly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib  # bind 'matplotlib' for rcParams
matplotlib.rcParams['pdf.fonttype'] = 42  # editable text in Illustrator (not Type 3 outlined glyphs)
matplotlib.rcParams['ps.fonttype']  = 42
import numpy as np
import pandas as pd
from scipy.stats import binomtest, false_discovery_control

from analysis.shared.paths import RESULTS_ROOT, SOURCE_DATA_ROOT
from analysis.shared.plotting import save_figure


def load_cache(cache_dir: Path, labels: set[str], lacuna_col: str) -> pd.DataFrame:
    """Load all per-sample shards from the cache, filter by sample label.
    `lacuna_col` is the boolean column to use for the lacuna stat:
      - has_lacuna_at_tss        (strict: gap covers TSS exactly)
      - has_lacuna_near_tss_250  (relaxed: gap intersects TSS ± 250 bp)
      - has_lacuna_near_tss_500  (more relaxed: gap intersects TSS ± 500 bp)
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SystemExit(
            "Install the optional pyarrow dependency to read a per-read cache"
        ) from error

    parts = []
    cols = ['transcript_id', 'gene_id', 'biotype', 'chrom', 'tss', 'strand',
            'sample', 'label', lacuna_col,
            'frac_prot_500', 'frac_prot_1000', 'frac_prot_1500']
    for sample_dir in sorted(cache_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        for shard in sorted(sample_dir.glob('shard_*.parquet')):
            t = pq.read_table(shard, columns=cols)
            df = t.to_pandas()
            if labels:
                df = df.loc[df['label'].isin(labels)]
            # rename to a stable name downstream
            if lacuna_col != 'has_lacuna_at_tss':
                df = df.rename(columns={lacuna_col: 'has_lacuna_at_tss'})
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def per_transcript_stats(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Aggregate per-(transcript_id) lacuna counts and mean protamine
    fraction for the chosen analysis window."""
    col = {500: 'frac_prot_500', 1000: 'frac_prot_1000',
           1500: 'frac_prot_1500'}[window]
    g = (df.groupby('transcript_id', observed=True)
           .agg(gene_id=('gene_id', 'first'),
                biotype=('biotype', 'first'),
                chrom=('chrom', 'first'),
                tss=('tss', 'first'),
                strand=('strand', 'first'),
                n_reads=('has_lacuna_at_tss', 'size'),
                n_lacuna=('has_lacuna_at_tss', 'sum'),
                mean_frac_prot=(col, 'mean'))
           .reset_index())
    g['frac_lacuna'] = g['n_lacuna'] / g['n_reads']
    return g


def plot_volcano(stats: pd.DataFrame, out_path_no_ext: Path,
                 cohort_label: str, null_label: str,
                 annotate_genes: list[str] | None = None,
                 featured_q_enriched: float = 0.20,
                 featured_q_depleted: float = 0.30):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    x = stats['frac_lacuna'].to_numpy() * 100
    y = -np.log10(np.clip(stats['p_value'].to_numpy(), 1e-300, None))
    q = stats['q_value'].to_numpy()
    log2or = stats['log2_or'].to_numpy()

    # 3-tier shading: q<0.05 (saturated), q<0.10 (mid), q<featured (lightest).
    # The wider featured tiers preserve the thresholds used for downstream
    # functional-enrichment summaries.
    pos = log2or > 0
    neg = log2or < 0

    en_05 = pos & (q < 0.05)
    en_10 = pos & (q >= 0.05) & (q < 0.10)
    en_ft = pos & (q >= 0.10) & (q < featured_q_enriched)
    de_05 = neg & (q < 0.05)
    de_10 = neg & (q >= 0.05) & (q < 0.10)
    de_ft = neg & (q >= 0.10) & (q < featured_q_depleted)
    is_ns = ~(en_05 | en_10 | en_ft | de_05 | de_10 | de_ft)

    # ns first (background), then loose tiers, then strict on top
    ax.scatter(x[is_ns], y[is_ns], s=4, alpha=0.18, color='#bbbbbb',
               label=f'ns (n={is_ns.sum():,})', rasterized=True)
    ax.scatter(x[de_ft], y[de_ft], s=6, alpha=0.5, color='#cfdef7',
               label=f'depleted q<{featured_q_depleted} (n={de_ft.sum():,})',
               rasterized=True)
    ax.scatter(x[en_ft], y[en_ft], s=6, alpha=0.5, color='#fcd9b8',
               label=f'enriched q<{featured_q_enriched} (n={en_ft.sum():,})',
               rasterized=True)
    ax.scatter(x[de_10], y[de_10], s=10, alpha=0.7, color='#aec7e8',
               label=f'depleted q<0.10 (n={de_10.sum():,})', rasterized=True)
    ax.scatter(x[en_10], y[en_10], s=10, alpha=0.7, color='#ffb499',
               label=f'enriched q<0.10 (n={en_10.sum():,})', rasterized=True)
    ax.scatter(x[de_05], y[de_05], s=14, alpha=0.9, color='#1f77b4',
               label=f'depleted q<0.05 (n={de_05.sum():,})', rasterized=True)
    ax.scatter(x[en_05], y[en_05], s=14, alpha=0.9, color='#d62728',
               label=f'enriched q<0.05 (n={en_05.sum():,})', rasterized=True)

    p_null = stats['p_global'].iloc[0] * 100
    ax.axvline(p_null, ls='--', color='black', lw=0.8,
               label=f'null rate ({null_label}: {p_null:.2f}%)')

    if annotate_genes:
        # one annotation per gene = the most-significant TSS for that gene
        gene_first = (stats.sort_values('p_value')
                            .drop_duplicates('gene_id'))
        texts = []
        anchor_x, anchor_y = [], []
        for g_ in annotate_genes:
            sub = gene_first.loc[gene_first['gene_id'] == g_]
            if sub.empty:
                continue
            r = sub.iloc[0]
            xv = float(r['frac_lacuna']) * 100
            yv = -np.log10(max(float(r['p_value']), 1e-300))
            ax.scatter([xv], [yv], s=70, facecolor='none',
                       edgecolor='black', linewidth=1.5, zorder=10)
            t = ax.text(xv, yv, g_, fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  fc='white', ec='black', lw=0.5, alpha=0.9),
                        zorder=11)
            texts.append(t)
            anchor_x.append(xv)
            anchor_y.append(yv)
        # auto-space labels to avoid overlap
        try:
            from adjustText import adjust_text
            adjust_text(texts, x=anchor_x, y=anchor_y, ax=ax,
                        arrowprops=dict(arrowstyle='-', color='black',
                                        lw=0.6, alpha=0.7),
                        expand=(1.4, 1.6),
                        force_text=(0.7, 0.9),
                        force_static=(0.4, 0.5))
        except ImportError:
            pass
    ax.set_xlabel(f'% reads with a lacuna near TSS')
    ax.set_ylabel('-log10 p-value (binomial vs null rate)')
    ax.set_title(f'TSS lacuna enrichment — {cohort_label}')
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    fig.tight_layout()
    save_figure(fig, out_path_no_ext)
    plt.close(fig)
    print(f'  saved {out_path_no_ext.name}.png / .pdf', file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--stats', type=Path,
        default=SOURCE_DATA_ROOT / 'promoter_regulation' /
                'tss_lacuna_transcript_statistics.tsv',
        help='bundled per-transcript statistics for a direct numerical redraw',
    )
    ap.add_argument(
        '--cache-dir', type=Path,
        help='optional per-read parquet cache; supplying it rebuilds statistics',
    )
    ap.add_argument('--labels', default='high,low,repl,new',
                    help='comma-list of sample labels to include in the pool')
    ap.add_argument('--lacuna', default='has_lacuna_near_tss_250',
                    choices=['has_lacuna_at_tss',
                             'has_lacuna_near_tss_250',
                             'has_lacuna_near_tss_500'],
                    help='which lacuna definition to use for the binary stat')
    ap.add_argument('--window', type=int, default=1000, choices=[500, 1000, 1500])
    ap.add_argument('--min-reads', type=int, default=20)
    ap.add_argument('--max-reads', type=int, default=100,
                    help='drop TSSs with more than this many reads — '
                         'usually indicates a repetitive locus / multi-mapper')
    ap.add_argument('--biotypes', default='mRNA',
                    help='comma-list of transcript biotypes to keep '
                         '(default: mRNA only). Pass "all" to disable '
                         'the filter.')
    ap.add_argument('--out', type=Path,
                    default=RESULTS_ROOT / 'promoter_regulation' /
                            'tss_lacuna_enrichment')
    ap.add_argument('--annotate',
                    default='SOX2,MEI4,CBX3,ADGB,INO80D,E2F3,'
                            'OR2L3,XPA,APOBEC1,ASXL2',
                    help='comma-list of gene symbols to circle and label '
                         'on the volcano. Defaults: 6 lacuna-rich + 4 '
                         'lacuna-poor featured candidates. Pass "" to disable.')
    ap.add_argument(
        '--null-table', type=Path,
        default=SOURCE_DATA_ROOT / 'promoter_regulation' /
                'non_tss_empirical_null.tsv',
        help='empirical non-TSS background table used with --null auto',
    )
    ap.add_argument('--null', default='auto',
                    help="null rate source: 'auto' (the POOLED row of "
                         "--null-table), 'pooled' (mean of the selected "
                         "cache rows), or a numeric value (for example 0.07)")
    args = ap.parse_args()

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    annotate = [s.strip() for s in args.annotate.split(',') if s.strip()]
    if args.cache_dir is None:
        stats = pd.read_csv(args.stats, sep='\t')
        required = {'frac_lacuna', 'p_value', 'p_global', 'log2_or', 'q_value'}
        if missing := required.difference(stats.columns):
            raise ValueError(f'{args.stats} lacks columns: {sorted(missing)}')
        plot_volcano(
            stats,
            out,
            cohort_label='human sperm',
            null_label='empirical non-TSS background',
            annotate_genes=annotate or None,
        )
        n_enriched = int(((stats['q_value'] < 0.05) &
                          (stats['log2_or'] > 0)).sum())
        n_depleted = int(((stats['q_value'] < 0.05) &
                          (stats['log2_or'] < 0)).sum())
        print(f'loaded {len(stats):,} transcript rows; q<0.05: '
              f'{n_enriched:,} enriched, {n_depleted:,} depleted')
        return

    cache_dir = args.cache_dir

    labels = set(s.strip() for s in args.labels.split(','))
    biotypes = (None if args.biotypes.strip().lower() == 'all'
                else set(s.strip() for s in args.biotypes.split(',')))

    print(f'loading cache from {cache_dir} (labels = {labels}, '
          f'lacuna = {args.lacuna})...', file=sys.stderr)
    df = load_cache(cache_dir, labels, args.lacuna)
    if df.empty:
        print(
            'No cache rows were found; supply a cache generated by the '
            'promoter-lacuna preprocessing workflow.',
            file=sys.stderr,
        )
        sys.exit(1)
    if biotypes is not None:
        df = df.loc[df['biotype'].isin(biotypes)]
        print(f'  {len(df):,} (TSS, read) rows after biotype filter '
              f'({sorted(biotypes)})', file=sys.stderr)
    else:
        print(f'  {len(df):,} (TSS, read) rows (all biotypes)',
              file=sys.stderr)

    print(f'\naggregating per-transcript (window=±{args.window})...',
          file=sys.stderr)
    stats = per_transcript_stats(df, args.window)
    n_before = len(stats)
    stats = stats.loc[stats['n_reads'] >= args.min_reads]
    if args.max_reads:
        stats = stats.loc[stats['n_reads'] <= args.max_reads]
    stats = stats.reset_index(drop=True)
    print(f'  {len(stats):,} transcripts with {args.min_reads} <= n_reads '
          f'<= {args.max_reads} (dropped {n_before - len(stats):,})',
          file=sys.stderr)

    # Null-rate modes: empirical non-TSS background (default), an explicitly
    # requested pooled-cache rate, or a caller-supplied numeric value.
    null_label = 'pooled cache'
    if args.null == 'pooled':
        p_null = float(df['has_lacuna_at_tss'].mean())
    elif args.null == 'auto':
        if not args.null_table.exists():
            raise FileNotFoundError(
                f'Empirical null table not found: {args.null_table}. '
                'Supply --null-table, --null pooled, or a numeric --null.'
            )
        ndf = pd.read_csv(args.null_table, sep='\t')
        pooled = ndf.loc[ndf['sample'] == 'POOLED']
        if len(pooled) != 1:
            raise ValueError(f'{args.null_table} must contain one POOLED row')
        row = pooled.iloc[0]
        col = {'has_lacuna_at_tss': 'rate_strict',
               'has_lacuna_near_tss_250': 'rate_250',
               'has_lacuna_near_tss_500': 'rate_500'}[args.lacuna]
        p_null = float(row[col])
        null_label = f'non-TSS empirical (n={int(row["n_pairs"]):,} pairs)'
    else:
        p_null = float(args.null)
        null_label = f'fixed = {p_null}'
    print(f'  null rate ({null_label}) = {p_null*100:.2f}%', file=sys.stderr)

    # binomial test per transcript
    print('\ntesting per transcript...', file=sys.stderr)
    pvals = np.empty(len(stats))
    for i in range(len(stats)):
        n = int(stats.at[i, 'n_reads'])
        k = int(stats.at[i, 'n_lacuna'])
        # two-sided to get both enriched and depleted
        pvals[i] = binomtest(k, n, p=p_null, alternative='two-sided').pvalue
    stats['p_value'] = pvals
    stats['p_global'] = p_null
    # log2 odds ratio of (transcript rate / null rate); gives sign of effect
    eps = 1e-9
    stats['log2_or'] = np.log2((stats['frac_lacuna'] + eps) / (p_null + eps))
    # BH FDR
    q = false_discovery_control(stats['p_value'].to_numpy(), method='bh')
    stats['q_value'] = q
    stats = stats.sort_values('p_value').reset_index(drop=True)
    stats.to_csv(str(out) + '_stats.tsv', sep='\t', index=False)
    print(f'  wrote {out}_stats.tsv ({len(stats):,} transcripts)',
          file=sys.stderr)
    n_sig = (stats['q_value'] < 0.05).sum()
    n_enriched = ((stats['q_value'] < 0.05) & (stats['log2_or'] > 0)).sum()
    n_depleted = ((stats['q_value'] < 0.05) & (stats['log2_or'] < 0)).sum()
    print(f'  significant @ FDR 5%: {n_sig:,}  '
          f'(enriched={n_enriched:,}, depleted={n_depleted:,})',
          file=sys.stderr)

    plot_volcano(stats, out,
                 cohort_label=' + '.join(sorted(labels)) +
                              f' / {args.lacuna.replace("has_lacuna_", "")}',
                 null_label=null_label,
                 annotate_genes=annotate or None)

    # also dump the top hits for sanity
    top = stats.nsmallest(20, 'p_value')[
        ['transcript_id', 'gene_id', 'biotype', 'chrom', 'tss',
         'n_reads', 'n_lacuna', 'frac_lacuna', 'log2_or',
         'p_value', 'q_value']
    ]
    print('\ntop 20 hits (per-transcript):', file=sys.stderr)
    print(top.to_string(index=False), file=sys.stderr)

    # gene-level summary: collapse identical-TSS transcripts (same gene+tss)
    # → one row, keeping the most-significant transcript_id per gene
    gene_stats = (stats.sort_values('p_value')
                       .drop_duplicates(['gene_id', 'chrom', 'tss'])
                       .reset_index(drop=True))
    gene_stats.to_csv(str(out) + '_gene_stats.tsv', sep='\t', index=False)
    n_gsig = (gene_stats['q_value'] < 0.05).sum()
    print(f'\ngene-level (collapsed): {len(gene_stats):,} unique TSSs, '
          f'{n_gsig:,} significant @ FDR 5%', file=sys.stderr)
    top_genes = gene_stats.nsmallest(20, 'p_value')[
        ['gene_id', 'biotype', 'chrom', 'tss', 'n_reads', 'n_lacuna',
         'frac_lacuna', 'log2_or', 'p_value', 'q_value']
    ]
    print('\ntop 20 hits (gene-level):', file=sys.stderr)
    print(top_genes.to_string(index=False), file=sys.stderr)


if __name__ == '__main__':
    main()
