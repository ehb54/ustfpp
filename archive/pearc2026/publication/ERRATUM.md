# Erratum

**Paper:** Householder, Zou, and Brookes. *Evaluating Deep Learning–Based Performance Prediction for
UltraScan Workflows Under Temporal and Observability Constraints.* PEARC '26.

**Issued:** 2026-07-25

This note corrects narrative text in the published paper. **No table, figure, result, or conclusion
is affected.** The corrections concern sentences in Sections 4.2 and 4.4 that restate rounded table
entries as if they were exact values.

---

## 1. Section 4.2 — sufficiency rates stated as 80% and 100%

Table 4 reports runtime sufficiency to one decimal place as `0.8`, `1.0`, and `1.0`. These entries
are correct roundings. The prose in Section 4.2 restates them as literal percentages, which
overstates two of the three values:

| Family | Paper text (§4.2) | Table 4 (published, 1 d.p.) | Exact value |
| ------ | ----------------- | --------------------------- | ----------- |
| 2DSA   | "sufficiency falls to 80%" | 0.8 | **83.94%** |
| GA     | "100% of accepted jobs meet their conservatively buffered requirements" | 1.0 | **97.03%** |
| PCSA   | "100% of accepted jobs meet the conservatively buffered runtime requirements" | 1.0 | **95.01%** |

The affected sentences should be read as:

- PCSA: "…accepts 89.8% of test jobs, and 95.0% of accepted jobs meet the conservatively buffered
  runtime requirements."
- GA: "…accepts 80.5% of GA jobs, and 97.0% of accepted jobs meet their conservatively buffered
  requirements."
- 2DSA: "Among this restricted subset, sufficiency falls to 83.9%, below the near-unity sufficiency
  observed for PCSA and GA."

Neither GA nor PCSA achieves literal 100% sufficiency. The phrase "near-unity" remains an accurate
characterization of both.

## 2. Section 4.4 — memory sufficiency stated as exceeding 90%

Section 4.4 states that memory "sufficiency rates exceed 90% for all methods." Table 5 reports
`90.0` for both 2DSA and GA. The exact values are:

| Family | Table 5 (published, 1 d.p.) | Exact value |
| ------ | --------------------------- | ----------- |
| 2DSA   | 90.0 | **90.04%** |
| GA     | 90.0 | **89.96%** |
| PCSA   | 97.3 | **97.28%** |

GA memory sufficiency is marginally below 90%, not above it. The sentence should read
"…sufficiency rates are approximately 90% or above for all methods." The target-specific stability
contrast drawn in Section 4.4 is unaffected.

---

## Scope of impact

The following are **correct as published** and require no revision:

- **Table 4** and **Table 5** — correct at the precision reported.
- **Acceptance rates** (27.8%, 80.5%, 89.8%) — unaffected; identical for both prediction targets.
- **Buffer inflation factors** (4.0×, 0.6×, 1.0×) — unaffected. The 2DSA result rests on buffer
  inflation, not on the sufficiency rate.
- **Section 4.3** attribution of 2DSA drift to Cluster 29 — unaffected.
- **Abstract, Section 5 (Conclusion), and all figures** — contain no affected quantities.
- The three-regime finding (PCSA stable, GA sufficient at conservative cost, 2DSA drifted) — the
  corrected values preserve every ordering and every qualitative claim.

## Reproducing these values

The exact figures above are produced by the analysis scripts in this repository and are recorded in
their output artifacts:

```bash
python analysis/08_simulation/cpu_sufficiency_simulation.py     # -> results/cpu_sufficiency_summary.csv
python analysis/08_simulation/memory_sufficiency_simulation.py  # -> results/table_4_memory_sufficiency.csv
```

Sufficiency is `n_sufficient / n_accepted` (runtime) and `n_sufficient / n_gated_in` (memory):

| Family | Runtime | Memory |
| ------ | ------- | ------ |
| 2DSA   | 9,923 / 11,822 = 0.83937 | 10,645 / 11,822 = 0.90044 |
| GA     | 1,827 / 1,883 = 0.97026 | 1,694 / 1,883 = 0.89963 |
| PCSA   | 5,445 / 5,731 = 0.95010 | 5,575 / 5,731 = 0.97278 |

## Cause

Both corrections have the same origin: prose was written from the one-decimal table entries rather
than from the underlying result files. Values in future work should be cited from
`analysis/08_simulation/results/`, not from the published tables.
