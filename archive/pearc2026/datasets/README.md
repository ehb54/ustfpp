# PEARC 2026 replication dataset

This package contains the cleaned datasets used by the PEARC 2026 paper. Each
file includes submission-time configuration features, encoded cluster identity,
timestamps, and performance targets.

## Contents

| File | Method | Expected records |
|---|---|---:|
| `2dsa_cleaned.csv.gz` | 2DSA | 283,412 |
| `ga_cleaned.csv.gz` | GA | 15,581 |
| `pcsa_cleaned.csv.gz` | PCSA | 42,536 |

These counts match Table 1 of the manuscript. The paper's selected models use
the raw feature space.

## Scrubbing boundary

The files exclude raw source labels, submitter information, email addresses, and
job identifiers. Cluster identity is represented by numeric codes, and no
code-to-name mapping is included.

The `submitTime`, `startTime`, `endTime`, and `updateTime` columns are retained.
They are required to reproduce temporal ordering, drift analyses, and the paper's
record-order versus temporal comparisons. They should receive an explicit data-
owner review before public release because they preserve historical operational
timing.

## Verification

See `manifest.json` for exact columns and SHA-256 checksums of both the compressed
and uncompressed content.

To inspect a file without replacing it:

```bash
gzip -cd 2dsa_cleaned.csv.gz | head
```

## Release status

Candidate for public release. A redistribution license and confirmation from the
data owner/collaborators are still required.
