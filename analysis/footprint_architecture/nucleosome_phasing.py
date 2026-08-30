#!/usr/bin/env python3
"""Nucleosome phasing metaprofile: anchor at the 5' edge of each nucleosome,
plot probability of being inside another nucleosome at each downstream
offset (0..3000 bp).

Compares m6a_200 human sperm with GM12878 and marks 180-bp and 195-bp
nucleosome-repeat-length guides for visual reference.

Data sources (no BAM needed — fp BEDs have all nucleosome calls):
  <fp-dir>/m6a_200_fp.bed
  <fp-dir>/GM12878_fp.bed

The bundled compact table is used by default. Pass ``--fp-dir`` to rebuild the
profiles from primary footprint calls.

Output: ``results/footprint_architecture/nucleosome_phasing.{png,pdf}``.
"""
from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.shared.paths import RESULTS_ROOT, SOURCE_DATA_ROOT
from analysis.shared.plotting import save_figure
import matplotlib.pyplot as plt

SOURCE_DATA = SOURCE_DATA_ROOT / 'footprint_architecture' / 'nucleosome_phasing.tsv'

NUC_MIN, NUC_MAX = 90, 200      # nucleosome footprint size range
MAX_OFFSET       = 3000          # bp downstream to profile
SMOOTH_BIN       = 5             # 5-bp smoothing


def parse_fp_blocks(fields):
    """Returns list of (block_start_rel, block_size) for one read.
    Both m6a_200_fp.bed and GM12878_fp.bed put blockStarts in the
    penultimate column and blockSizes in the last column."""
    try:
        starts = [int(x) for x in fields[-2].rstrip(',').split(',') if x]
        sizes  = [int(x) for x in fields[-1].rstrip(',').split(',') if x]
    except (ValueError, IndexError):
        return []
    if len(starts) != len(sizes):
        return []
    return list(zip(starts, sizes))


def process_fp_bed(bed_path: Path, n_limit: int | None, label: str):
    """Stream the fp BED, accumulate downstream-nucleosome-density profile."""
    counter = np.zeros(MAX_OFFSET + 1, dtype=np.int64)
    denom   = np.zeros(MAX_OFFSET + 1, dtype=np.int64)
    n_reads_seen = 0
    n_reads_kept = 0
    n_nucs_total = 0

    open_fn = gzip.open if str(bed_path).endswith('.gz') else open
    print(f'[{label}] streaming {bed_path}', file=sys.stderr)
    with open_fn(str(bed_path), 'rt') as fh:
        for line in fh:
            if line.startswith('chrom') or line.startswith('#'): continue
            n_reads_seen += 1
            if n_limit is not None and n_reads_kept >= n_limit: break
            f = line.rstrip('\n').split('\t')
            if len(f) < 10: continue
            try:
                read_start = int(f[1]); read_end = int(f[2])
            except ValueError: continue
            read_len = read_end - read_start
            if read_len < 1500: continue
            blocks = parse_fp_blocks(f)
            if not blocks: continue
            # extract nucleosome 5' edges (sorted)
            nuc_edges = sorted(s for s, sz in blocks if NUC_MIN <= sz < NUC_MAX)
            if len(nuc_edges) < 3: continue
            # build a per-bp "inside any nuc" mask in read-local coords
            mask = np.zeros(read_len, dtype=bool)
            for s, sz in blocks:
                if NUC_MIN <= sz < NUC_MAX:
                    end = min(s + sz, read_len)
                    if s < read_len:
                        mask[s:end] = True
            # for each anchor nuc 5' edge, accumulate downstream profile
            for anchor in nuc_edges:
                # window [anchor, anchor + MAX_OFFSET]
                hi = min(anchor + MAX_OFFSET + 1, read_len)
                width = hi - anchor
                if width < 200: continue   # need at least some downstream
                counter[:width] += mask[anchor:hi]
                denom[:width]   += 1
                n_nucs_total += 1
            n_reads_kept += 1
            if n_reads_kept % 10000 == 0:
                print(f'  [{label}] kept {n_reads_kept:,} reads  '
                      f'({n_nucs_total:,} anchor nucs)', file=sys.stderr)
    print(f'[{label}] DONE: {n_reads_kept:,} reads / {n_nucs_total:,} '
          f'anchor nucs (saw {n_reads_seen:,} lines)',
          file=sys.stderr)
    with np.errstate(divide='ignore', invalid='ignore'):
        profile = counter / np.maximum(denom, 1)
    return profile, denom, n_reads_kept, n_nucs_total


def smooth(y, w):
    if w <= 1: return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode='same')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-reads', type=int,
                    help='optional cap; omit to stream all eligible reads')
    ap.add_argument('--smooth-bin', type=int, default=SMOOTH_BIN)
    ap.add_argument('--fp-dir', type=Path,
                    help='directory containing m6a_200_fp.bed and GM12878_fp.bed; '
                         'omit to redraw from --source-data')
    ap.add_argument('--source-data', type=Path, default=SOURCE_DATA,
                    help='compact profile TSV used when --fp-dir is omitted')
    ap.add_argument('--write-source-data', type=Path,
                    help='optional TSV path for profiles rebuilt with --fp-dir')
    ap.add_argument('--out-dir', type=Path,
                    default=RESULTS_ROOT / 'footprint_architecture')
    args = ap.parse_args()

    profiles = {}
    rows = []
    colors = {'m6a_200 sperm': '#d62728', 'GM12878': '#1f77b4'}
    if args.fp_dir is not None:
        datasets = [
            ('m6a_200 sperm', args.fp_dir / 'm6a_200_fp.bed'),
            ('GM12878',       args.fp_dir / 'GM12878_fp.bed'),
        ]
        for label, bed_path in datasets:
            if not bed_path.is_file():
                raise FileNotFoundError(bed_path)
            profile, denom, n_r, n_nuc = process_fp_bed(
                bed_path, args.max_reads, label
            )
            smoothed = smooth(profile, args.smooth_bin)
            profiles[label] = (smoothed, denom, n_r, n_nuc, colors[label])
            for off, p in enumerate(smoothed):
                rows.append((label, off, p, int(denom[off])))
        if args.write_source_data is not None:
            args.write_source_data.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                rows, columns=['dataset', 'offset_bp', 'p_in_nuc', 'denom']
            ).to_csv(args.write_source_data, sep='\t', index=False)
    else:
        if not args.source_data.is_file():
            raise FileNotFoundError(
                f'Missing {args.source_data}; pass --fp-dir to rebuild profiles'
            )
        source = pd.read_csv(args.source_data, sep='\t')
        required = {'dataset', 'offset_bp', 'p_in_nuc', 'denom'}
        if missing := required.difference(source.columns):
            raise ValueError(f'{args.source_data} lacks columns: {sorted(missing)}')
        for label, frame in source.groupby('dataset', sort=False):
            frame = frame.sort_values('offset_bp')
            profile = frame['p_in_nuc'].to_numpy(dtype=float)
            denom = frame['denom'].to_numpy(dtype=int)
            profiles[label] = (
                profile, denom, None, int(denom[0]), colors.get(label, '#333333')
            )

    # ============================================================
    # plot
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A — both overlaid, full range
    ax = axes[0]
    for label, (smoothed, denom, n_r, n_nuc, color) in profiles.items():
        legend_label = label if n_r is None else (
            f'{label}  (n={n_r:,} reads, {n_nuc:,} anchors)'
        )
        ax.plot(np.arange(len(smoothed)), smoothed, label=legend_label,
                color=color, linewidth=0.9)
    ax.set_xlabel('offset from nucleosome 5′ edge (bp)', fontsize=10)
    ax.set_ylabel('P(inside another nucleosome)', fontsize=10)
    ax.set_title('A. Downstream nucleosome density (full range)',
                  fontsize=11, loc='left')
    ax.grid(alpha=0.2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9, loc='upper right')

    # Panel B — zoom 0-1500 with phasing peaks marked
    ax = axes[1]
    zoom_hi = 1500
    for label, (smoothed, denom, n_r, n_nuc, color) in profiles.items():
        x = np.arange(zoom_hi + 1)
        ax.plot(x, smoothed[:zoom_hi + 1], label=label, color=color,
                 linewidth=1.0)
    # expected phasing peaks (5' edge + ~180 bp NRL)
    for k, NRL in enumerate([180, 195]):
        for n in range(1, 8):
            ax.axvline(n * NRL, color='gray', alpha=0.25,
                        linestyle=':' if k == 0 else '--',
                        linewidth=0.6)
    ax.set_xlabel('offset from nucleosome 5′ edge (bp)', fontsize=10)
    ax.set_ylabel('P(inside another nucleosome)', fontsize=10)
    ax.set_title('B. Zoom 0–1500 bp · dotted=180 bp NRL, dashed=195 bp NRL',
                  fontsize=11, loc='left')
    ax.grid(alpha=0.2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9, loc='upper right')

    fig.suptitle("Nucleosome phasing metaprofile anchored at the 5′ edge",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / 'nucleosome_phasing'
    save_figure(fig, stem)
    plt.close(fig)

    print(f'wrote {stem.name}.png/.pdf', file=sys.stderr)


if __name__ == '__main__':
    main()
