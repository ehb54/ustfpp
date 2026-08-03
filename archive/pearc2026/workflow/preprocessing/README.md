# PEARC 2026 preprocessing workflow

This directory preserves the extraction, cleaning, and verification source used
for the 2026 work. It is a historical snapshot, not a self-contained raw-data
release.

`extract_data.py` creates per-method datasets from approved private source
metadata. `preprocess_filter.py` applies the archived
`preprocess_filter_config.json`, and `verify_cleaned.py` checks the resulting
cleaned data. The public, scrubbed outputs of this process are in
`../../datasets/` as gzip-compressed CSV files with checksums in `manifest.json`.

The raw metadata input and its extraction environment are not included. Do not
write newly generated files into this archive; use a separate working copy for
any rerun or extension.
