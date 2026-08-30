# Protamine Fiber-seq analysis

This repository contains the paper-specific Python workflows for the principal
quantitative analyses in *Protamine lacunae preserve the paternal chromatin
landscape in sperm*. It is a script-only release: there are no notebooks or
exploratory working directories.

The bundled compact source tables allow every default workflow to run without
access to primary human sequencing files. Primary-data access and release
status are documented in [DATA_ACCESS.md](DATA_ACCESS.md). The companion
[FiberFormer repository](https://github.com/mtcicero26/fiberformer) contains the
model architecture, training and inference code, reported checkpoint, and a
synthetic inference example.

## Quick start

The release was tested on Linux x86-64 under WSL2 with Python 3.10. A CPU is
sufficient for all bundled workflows; no specialized hardware is required.

```bash
git clone https://github.com/mtcicero26/protamine-fiberseq-analysis.git
cd protamine-fiberseq-analysis
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m analysis.reproduce
```

Dependency installation normally takes 2–5 minutes. The complete bundled-data
run writes PNG and editable-text PDF outputs below `results/`. Reference
outputs are included in that directory. On the release-preparation host, the
expanded automated run completed in about two minutes and used less than 1 GB
of memory.

Run the automated source-data and execution check with:

```bash
make check
```

## Bundled workflows

| Workflow | Analysis | Output stem(s) |
| --- | --- | --- |
| `early_chromatin` | Read accessibility, lacuna size, nucleosomes per lacuna, and protamine-edge organization | `results/footprint_architecture/{read_accessibility,lacuna_size_distribution,nucleosomes_per_lacuna,protamine_edge_nucleosome_profile}` |
| `footprint_architecture` | Nucleosome phasing in sperm and somatic chromatin | `results/footprint_architecture/nucleosome_phasing` |
| `testis_transition` | FiberFormer transition states, chromatin overlays, and pseudotime trajectory | `results/testis_transition/transition_*`, `protamine_fraction`, `nucleosome_fraction` |
| `promoter_lacunae` | Transcript-level promoter-lacuna enrichment | `results/promoter_regulation/tss_lacuna_enrichment` |
| `go_enrichment` | Curated developmental and chromatin GO enrichment | `results/promoter_regulation/curated_go_enrichment` |
| `gene_set_enrichment` | Preranked broad gene-set enrichment | `results/promoter_regulation/gene_set_enrichment` |
| `promoter_chromatin` | FIRE and histone-mark overlap at promoters | `results/regulatory_elements/promoter_fire_histone_overlap` |
| `motif_retention` | Motif-associated chromatin retention | `results/regulatory_elements/motif_retention` |
| `genome_context` | Regional nucleosome and protamine occupancy | `results/genome_context/region_occupancy` |
| `mouse_prm2` | Per-read protamine blocks in wild-type and *Prm2*-knockout sperm | `results/fertility/mouse_protamine_blocks` |
| `motility_correlations` | Human sperm chromatin metrics and motility | `results/fertility/motility_correlations` |
| `motility_stability` | Sequencing-depth sensitivity of motility correlations | `results/fertility/motility_stability` |
| `assay_controls` | Hia5 dose/time titrations and technical-replicate footprint/lacuna distributions | `results/assay_controls/{hia5_titration,technical_replicates}` |
| `assay_periodicity` | DAF-seq and Hia5 modification-position autocorrelation | `results/assay_validation/periodicity*` |
| `read_heatmaps` | Deidentified TSS-centered, mouse, motility, and representative paired-donor DAF-seq footprint heatmaps | `results/read_heatmaps/*_heatmaps` |

List the workflow names or run a subset:

```bash
python -m analysis.reproduce --list
python -m analysis.reproduce --only promoter_lacunae read_heatmaps mouse_prm2
```

Each module can also be called directly and documents its input overrides with
`--help`. For example:

```bash
python -m analysis.fertility.motility_correlations --help
python -m analysis.testis_transition.transition_embedding --help
```

The generated files are numerical analysis exports. Multi-panel layout and
typography for the manuscript were assembled separately from these outputs.

## Recomputing from upstream data

The default commands consume the compact tables in `data/source/`. Several
modules also expose an upstream-data mode:

- `analysis.footprint_architecture.nucleosome_phasing --fp-dir ...` rebuilds
  the phasing profiles from footprint BED files.
- `analysis.footprint_architecture.early_chromatin_distributions
  --fiberhmm-bed ...` rebuilds the aggregate Figure 1 distributions from
  uniformly sampled FiberHMM BED12 calls (one million reads for distributions
  and 100,000 reads for the protamine-edge profile by default).
- `analysis.promoter_regulation.tss_lacuna_enrichment --cache-dir ...`
  recomputes transcript statistics from a per-read Parquet cache.
- `analysis.promoter_regulation.gene_set_enrichment --gene-statistics ...`
  reruns preranked GSEA.
- `analysis.regulatory_elements.promoter_fire_histone_overlap` accepts gene
  statistics, FIRE BED, and five named histone-mark BED files.
- `analysis.fertility.mouse_protamine_blocks --cache-dir ...` rebuilds the
  per-read mouse table from footprint-block caches.

The `read_heatmaps` workflow reconstructs twelve frozen, deidentified base
matrices from run-length footprint intervals and applies the released row
orders. It validates SHA-256 checksums for every base matrix and displayed
panel before plotting. The bundled set covers the main TSS, mouse, and
low-motility read views plus paired-donor DAF-seq examples at chromosome 5,
chromosome 22, UBA1, and MEI4. These processed inputs contain no read names,
donor codes, genomic coordinates, or sequence records.

The bundled assay controls include five Hia5 amount conditions, six Hia5
reaction times, paired technical-replicate size distributions, and DAF-seq and
Hia5 modification-position periodicity. The titration source tables retain
only anonymous per-read m6A percentages and aggregate footprint-size bins;
the replicate tables contain aggregate size bins only.

Install `requirements-optional.txt` for the Parquet and GSEA modes. Full-scale
paths are supplied through command-line arguments; outputs can be redirected
with `PROTAMINE_OUTPUT_ROOT`.

## Repository layout

```text
analysis/       named executable analysis modules
data/source/    compact processed inputs used by the default commands
metadata/       accessions and software versions
results/        reference PNG/PDF outputs
tools/          release validation
```

The compact tables contain aggregate, feature-level, or deidentified
read-level values and do not contain sequence reads or direct participant
identifiers. See [data/README.md](data/README.md) for the input inventory.

## License and citation

Code is released under the MIT License. Citation metadata are provided in
`CITATION.cff`; please also cite the associated manuscript and the archived
release when available.
