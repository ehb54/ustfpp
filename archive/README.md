# Publication archive

This directory is being organized as the immutable record of completed USTFPP
iterations.

Each year uses the same structure:

- `publication/` - paper citation/link, presentation, corrections, and final
  publication assets; publisher PDFs are not stored in the repository
- `datasets/` - scrubbed replication datasets and their manifests
- `workflow/` - the scripts, configurations, and environment information needed
  to reproduce that year's work
- `results/` - selected models, aggregate metrics, and final tables and figures

Raw or identifying source data does not belong here. It should be retained in
approved private research storage. Bulk intermediates belong here only when they
cannot be reproduced from the archived workflow and datasets.

The repository root remains the working area for the next iteration. Once a
year's archive is verified, generated data and results for that year can be
removed from the root workflow directories.
