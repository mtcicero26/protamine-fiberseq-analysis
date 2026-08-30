#!/usr/bin/env python3
"""Compare modification-position autocorrelation across assay contexts.

The two views pair DAF-based deamination measurements and Hia5-based m6A
measurements in somatic chromatin with the corresponding measurements inside
sperm protamine footprints.  Dual y-axes retain the native scale of each
autocorrelation calculation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from analysis.shared.paths import RESULTS_ROOT, SOURCE_DATA_ROOT
from analysis.shared.plotting import save_figure
import matplotlib.pyplot as plt

MAX_LAG = 1000       # bp — enough for ~5 nucleosome-repeat cycles
NRL     = 180        # nominal nucleosome repeat length


def _load_scDAF_pooled(path: Path, max_lag: int):
    """Whole-read scDAF pooled autocorr at per-bp resolution (very smooth
    thanks to ~150M pair signals per lag)."""
    df = pd.read_csv(path, sep='\t')
    df = df.loc[(df.lag_bp >= 1) & (df.lag_bp <= max_lag)]
    return df.lag_bp.to_numpy(), df['pooled'].to_numpy()


def _load_inout(path: Path, dataset: str, context: str, scale: str, max_lag: int):
    df = pd.read_csv(path, sep='\t')
    sub = df.loc[(df.scale == scale)
                  & (df.context == context)
                  & (df.dataset == dataset)]
    sub = sub.loc[(sub.lag_bp >= 1) & (sub.lag_bp <= max_lag)].sort_values('lag_bp')
    return sub.lag_bp.to_numpy(), sub.autocorr.to_numpy()


def _panel_dual(ax_left, curve_left, curve_right, title, max_lag=MAX_LAG):
    """Two curves on one panel, dual y-axes. Curves are (label, x, y, color)."""
    l_label, l_x, l_y, l_color = curve_left
    r_label, r_x, r_y, r_color = curve_right

    ax_right = ax_left.twinx()
    ax_left.plot(l_x, l_y,  color=l_color, lw=1.2, label=l_label)
    ax_right.plot(r_x, r_y, color=r_color, lw=1.2, label=r_label)

    for x in range(NRL, max_lag + 1, NRL):
        ax_left.axvline(x, color='#33A040', lw=0.4, ls=':', alpha=0.55)
    ax_left.axhline(0, color=l_color, lw=0.3, alpha=0.4)

    ax_left.set_xlabel('lag (bp)', fontsize=9)
    ax_left.set_ylabel(f'autocorr\n({l_label})', fontsize=8.5, color=l_color)
    ax_right.set_ylabel(f'autocorr\n({r_label})', fontsize=8.5, color=r_color)
    ax_left.tick_params(axis='y',  colors=l_color, labelsize=8)
    ax_right.tick_params(axis='y', colors=r_color, labelsize=8)
    ax_left.tick_params(axis='x', labelsize=8)
    ax_left.set_xlim(0, max_lag)
    ax_left.set_title(title, fontsize=10, loc='left')
    ax_left.grid(alpha=0.12)
    ax_left.spines['top'].set_visible(False)
    ax_right.spines['top'].set_visible(False)

    # combined legend
    lines = [
        plt.Line2D([0], [0], color=l_color, lw=1.4, label=l_label),
        plt.Line2D([0], [0], color=r_color, lw=1.4, label=r_label),
    ]
    ax_left.legend(handles=lines, loc='upper right', fontsize=8,
                     frameon=True, framealpha=0.9)


def _render(sc_path: Path, fiberseq_path: Path, max_lag: int, out_stem: Path):
    sc_x, sc_y     = _load_scDAF_pooled(sc_path, max_lag)
    daf_in_x, daf_in_y  = _load_inout(fiberseq_path, 'DAF-seq',       'in_protamine', 'short', max_lag)
    gm_x, gm_y          = _load_inout(fiberseq_path, 'GM12878 all',   'all',          'short', max_lag)
    m6a_in_x, m6a_in_y  = _load_inout(fiberseq_path, 'm6a_200 sperm', 'in_protamine', 'short', max_lag)

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 4.8))

    _panel_dual(
        axes[0],
        ('scDAF-seq (HG002, whole read)',       sc_x, sc_y,   '#1f77b4'),
        ('DAF-seq (sperm, in-protamine)',       daf_in_x, daf_in_y, '#d62728'),
        'A · DAF-based deamination periodicity',
        max_lag=max_lag,
    )
    _panel_dual(
        axes[1],
        ('Hia5 m6A Fiber-seq (GM12878, whole read)',    gm_x, gm_y,   '#1f77b4'),
        ('Hia5 m6A Fiber-seq (m6a_200 sperm, in-protamine)', m6a_in_x, m6a_in_y, '#d62728'),
        'B · Hia5 m6A periodicity',
        max_lag=max_lag,
    )

    fig.suptitle(
        f'Modification-position autocorrelation 0–{max_lag} bp, same enzyme × chromatin state\n'
        'green dotted ticks = multiples of the ~180 bp nucleosome-repeat length',
        fontsize=10, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=3.2)
    save_figure(fig, out_stem, dpi=250)
    plt.close(fig)
    print(f'wrote {out_stem.name}.png/.pdf', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--scdaf-input', type=Path,
        default=SOURCE_DATA_ROOT / 'assay_validation' / 'scdafseq_autocorrelation.tsv')
    parser.add_argument(
        '--fiberseq-input', type=Path,
        default=SOURCE_DATA_ROOT / 'assay_validation' / 'fiberseq_autocorrelation.tsv')
    parser.add_argument(
        '--output-dir', type=Path,
        default=RESULTS_ROOT / 'assay_validation')
    args = parser.parse_args()
    _render(args.scdaf_input, args.fiberseq_input, MAX_LAG,
            args.output_dir / 'periodicity')
    _render(args.scdaf_input, args.fiberseq_input, 200,
            args.output_dir / 'periodicity_zoom_200bp')


if __name__ == '__main__':
    main()
