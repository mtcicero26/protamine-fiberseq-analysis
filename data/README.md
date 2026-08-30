# Compact source data

These tables are the processed numerical inputs used by the default analysis
commands. They contain no primary sequence reads or direct participant
identifiers.

| File | Rows | Contents |
| --- | ---: | --- |
| `source/assay_controls/hia5_footprint_histograms.tsv` | 1,100 | Aggregate 100-bin footprint-size distributions for five Hia5 amounts and six reaction times |
| `source/assay_controls/hia5_m6a_per_read.tsv.gz` | 33,000 | Anonymous per-read m6A percentages for the Hia5 dose and time controls |
| `source/assay_controls/technical_replicate_histograms.tsv` | 400 | Aggregate footprint- and accessible-patch-size distributions for two technical replicates |
| `source/assay_validation/fiberseq_autocorrelation.tsv` | 5,410 | Fiber-seq and DAF-seq autocorrelation profiles |
| `source/assay_validation/scdafseq_autocorrelation.tsv` | 1,001 | Pooled and per-cell scDAF-seq autocorrelation |
| `source/fertility/motility_metrics.tsv` | 13 | Per-sample chromatin and CASA motility metrics |
| `source/fertility/motility_stability.tsv` | 600 | Depth-resampling correlation results; sensitivity columns omit PS00754 |
| `source/fertility/mouse_protamine_blocks_per_read.tsv.gz` | 10,354 | Deidentified per-read protamine-block counts; up to 3,000 sampled reads per animal |
| `source/footprint_architecture/lacuna_size_distribution.tsv.gz` | 21,807 | Aggregate lacuna-size counts from 1,000,000 sampled sperm reads |
| `source/footprint_architecture/nucleosome_phasing.tsv` | 6,002 | Sperm and somatic nucleosome phasing profiles |
| `source/footprint_architecture/nucleosomes_per_lacuna.tsv.gz` | 4,057 | Aggregate nucleosome-count distributions by 100-bp lacuna-size bin |
| `source/footprint_architecture/protamine_edge_nucleosome_profile.tsv.gz` | 204,000 | Forty deidentified, distance-ordered protamine-edge footprint profiles |
| `source/footprint_architecture/read_accessibility.tsv` | 2 | Fraction of sampled sperm reads containing one or more accessible lacunae |
| `source/genome_context/region_occupancy.tsv` | 9 | Regional footprint occupancy and confidence intervals |
| `source/promoter_regulation/curated_go_enrichment.tsv` | 14 | Curated GO enrichment statistics |
| `source/promoter_regulation/gene_set_enrichment.tsv` | 559 | Broad preranked GSEA results |
| `source/promoter_regulation/non_tss_empirical_null.tsv` | 21 | Empirical non-TSS lacuna background rates |
| `source/promoter_regulation/tss_lacuna_gene_statistics.tsv` | 42,107 | Transcript results collapsed to unique gene/TSS coordinates; q values are inherited from the transcript-level analysis |
| `source/promoter_regulation/tss_lacuna_transcript_statistics.tsv` | 127,751 | Transcript-level promoter-lacuna statistics |
| `source/read_heatmaps/footprint_heatmap_intervals.tsv.gz` | 57,912 | Nonzero footprint-size runs from twelve deidentified heatmap matrices; accessible bases are implicit zeros |
| `source/read_heatmaps/heatmap_manifest.tsv` | 24 | Panel dimensions, display labels, selection/sort rules, layouts, and SHA-256 checksums |
| `source/read_heatmaps/heatmap_row_order.tsv.gz` | 25,535 | Anonymous frozen row orders and cluster ranks for each displayed heatmap view |
| `source/regulatory_elements/motif_retention_effects.tsv` | 163 | Motif retention effects and statistical tests |
| `source/regulatory_elements/promoter_fire_histone_overlap.tsv` | 40 | Promoter FIRE/histone overlap counts and percentages |
| `source/testis_transition/transition_embedding.tsv.gz` | 173,141 | Deidentified embedding coordinates, states, and chromatin fractions; read and donor identifiers are omitted |

Compressed tables use ordinary gzip-compressed TSV format and can be read
directly with pandas.

## Reference definitions

- Promoter and gene-coordinate analyses use GRCh38.p14
  (`GCF_000001405.40`) with NCBI RefSeq annotation release
  `GCF_000001405.40-RS_2023_10`.
- HG002 regional analyses use the HG002 v1.1a personal assembly. Centromere,
  satellite, centromere-dip-region (CDR), and rDNA annotations derive from the
  CenSatData HG002 resources; gene bodies use the HG002 v1.1 JHU annotation.
- Motif analyses use JASPAR 2024 CORE vertebrate matrices. FIREs and sperm
  histone-mark intervals are study-derived resources deposited with the study
  data; the bundled table contains their processed overlap summary.
- `scdafseq_autocorrelation.tsv` retains the 12 deidentified single-cell
  profiles as well as the pooled column used in the default plot.
