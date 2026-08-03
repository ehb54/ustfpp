# PEARC 2025 replication dataset

This package contains the processed datasets associated with the 2025 paper,
*Optimizing UltraScan Job Scheduling with Deep Learning-Based Performance
Prediction*.

## Contents

| File | Method | Records | Known role |
|---|---|---:|---|
| `2dsa_filtered.csv.gz` | 2DSA | 265,346 | Archived grid-search candidate; matches Table 1 |
| `ga_filtered.csv.gz` | GA | 14,006 | Archived grid-search candidate; matches Table 1 |
| `pcsa_filtered.csv.gz` | PCSA | 45,340 | Archived grid-search candidate; matches Table 1 |
| `2dsa_preprocessed.csv.gz` | 2DSA | 227,361 | Confirmed preprocessing output and feature-analysis input |

The first three counts match Table 1 and the filenames referenced by the
archived grid-search code. The paper's Section 2.2 says that removing
negative-valued 2DSA rows reduced that dataset from 265,346 to 227,361 rows;
`2dsa_preprocessed.csv.gz` preserves that post-filter population. All files are
gzip-compressed without changing their CSV content.

Both 2DSA files are included deliberately. The manuscript describes the
227,361-row file as the output of preprocessing, and the saved feature-analysis
configuration identifies it as that analysis's input. The reported job count and
archived grid-search code point to the 265,346-row file, but the original selected
model artifact is unavailable, so its exact training input cannot be independently
verified. This package preserves both populations rather than overstating the
available provenance.

## 2025 prediction target

The models reported in the 2025 paper predict `CPUTime`. The processed files also
retain `max_rss` and `wallTime` because they were recorded outputs in the source
dataset. They were not prediction targets for the reported 2025 grid search;
prediction of maximum memory usage was identified as future work.

The paper says `max_rss` and `wallTime` were removed from feature analysis to
avoid post-execution data leakage. However, the archived feature-analysis code
constructs an ignore list without applying it to the matrices passed to the
models, and the saved feature-importance outputs contain both columns. The
preserved artifacts therefore indicate that these post-execution variables were
unintentionally included in that feature analysis. This issue is separate from
the CPU-time grid-search models, whose preprocessing code drops both columns.

## Scrubbing boundary

The release contains processed configuration columns and the recorded `CPUTime`,
`max_rss`, and `wallTime` outputs. It excludes raw source labels, submitter
information, email addresses, job identifiers, and timestamps. Cluster identity
is represented only by the numeric values already present in the filtered
datasets; no code-to-name mapping is included.

## Verification

See `manifest.json` for exact columns and SHA-256 checksums of both the compressed
and uncompressed content.

To inspect a file without replacing it:

```bash
gzip -cd 2dsa_filtered.csv.gz | head
```
