# PEARC 2026 workflow

This directory separates the preserved model-building run from preprocessing
and later analysis. It intentionally contains source, configuration, and
environment files only; generated data, models, tables, and figures belong in
`../datasets/` and `../results/`.

## Contents

- `model_building/` preserves the worker, launcher, the 12 raw CPU/memory
  configurations, and the three raw feature-column lists used for the paper.
- `preprocessing/` contains the extraction, cleaning, and verification code and
  its configuration.
- `analysis/` contains the retained analysis, aggregation, simulation, and
  paper-output source files. Files under this directory may postdate the model
  run; they are preserved as analysis code, not represented as training code.
- `environment/requirements.txt` is the available project dependency list. It
  is not a fully locked environment or a record of the server modules.

## Replication notes

The configurations use the cleaned datasets in `../../datasets/` when run from
`model_building/`; their historical relative output paths write below
`model_building/results/`. To avoid changing the preserved record, run any new
experiments in a separate working copy and redirect outputs there.
