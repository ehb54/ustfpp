# PEARC 2026 final results

This is the compact, reproducibility-relevant record of the 2026 work.

- `selected_models/` contains the 12 selected raw-feature models, their fitted
  scalers, per-model grid-search metrics, and checkpoint metadata.
- `aggregate/` contains the 768 raw CPU/memory grid-search runs and their
  corresponding tail-error summaries.
- `tables/` contains the selected-model table and the paper-facing temporal,
  Z-score, and simulation tables.
- `figures/` contains the final z-score stratification figures.
- `run_log.csv` is the preserved run log.

The prior bulk grid-run copies were verified against the dated server run. They
were redundant: the server version added only checkpoint files, while every file
in `selected_models/` was verified to be present in that run. Those two bulk
copies were removed after this compact record was assembled.

The selected models use the raw feature space.
