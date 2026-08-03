# PEARC 2025 workflow

This directory preserves the best available paper-era source snapshot. The
files were recovered from the last repository state before the final paper PDF
was created. Generated datasets, models, plots, and result tables are stored
outside this directory.

## Contents

- `preprocess/` formats and filters the source data.
- `feature/` contains the feature-analysis implementation and configuration.
- `hyperparameter_architecture_analysis.py`, `run_model.py`, `split_config.py`,
  and `configs/` contain the model-building workflow.
- `util/` and `visual/` contain result collection and visualization utilities.
- `legacy_scripts/` contains older launch and conversion helpers formerly kept
  loose in the archive. Their presence is historical; they are not asserted to
  have produced the published results. `mpgverbatim.py` is a notebook-style
  TensorFlow example, not a runnable Python script.
- `requirements.txt` is the dependency list preserved with the snapshot.

## Provenance limit

No run manifest links one exact source revision, configuration, input file, and
selected model artifact for the published 2025 models. These are therefore the
closest recoverable paper-era scripts, not a claimed bit-for-bit record of the
original training run. In particular, the two preserved 2DSA populations in
`../datasets/` must remain distinct for the reasons documented there.

Several paths in the scripts reflect the repository layout at the time. Preserve
this copy unchanged; if a runnable restoration needs path updates, make those in
a separate wrapper or working copy.

The files in `configs/` use a JSON-like format with `#` comments. They are not
strict JSON, by design of the historical snapshot; use the matching 2025
launcher/parser rather than a generic JSON reader.
