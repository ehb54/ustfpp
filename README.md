# UltraScan TensorFlow Job Performance Prediction (USTFPP)

USTFPP is the working repository for predicting UltraScan job performance from
job configuration metadata. The repository is now prepared for the 2027
iteration while preserving the completed 2025 and 2026 work under `archive/`.

## Archived work

- [`archive/pearc2025/`](archive/pearc2025/) preserves the available 2025
  workflow, scrubbed datasets, historical intermediates, results, and author
  presentation. It supports inspection and partial reruns, but no preserved run
  artifact establishes the exact selected-model training input.
- [`archive/pearc2026/`](archive/pearc2026/) preserves scrubbed datasets,
  selected raw-feature models, aggregate metrics, final tables and figures,
  workflow source, correction notice, and author presentation. Its selected
  models can be evaluated against the archived datasets; exact full grid-search
  retraining is not claimed.

Raw LIMS exports and identifying source data are intentionally excluded. The
archived datasets are scrubbed processed releases; see each archive's
`datasets/README.md` and `manifest.json` for contents and checksums.

## Starting the next iteration

The root-level `preprocess/`, `feature/`, and `grid_search/` directories are
the reusable foundation for new work. They intentionally contain source and
configuration rather than historical generated data, models, or paper outputs.

Create an environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Use the preprocessing configuration in `preprocess/` to work from approved
private raw inputs. Generated working data and figures should remain outside
the preserved year archives unless they are curated as a final archive artifact.

## Repository layout

```
ustfpp/
├── archive/        # Completed 2025 and 2026 preservation records
├── feature/        # Reusable feature-analysis source
├── grid_search/    # Reusable model-search worker and launcher
├── presentation/   # Current presentation assets
├── preprocess/     # Reusable extraction and preprocessing source
└── requirements.txt
```
