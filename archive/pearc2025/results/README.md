# PEARC 2025 preserved results

The compact evidence currently available consists of feature-analysis outputs,
top-five feature-grid-search tables, and one historical deployed 2DSA inference
bundle. The original selected-model artifact and a complete run manifest were
not preserved, so this directory does not claim to contain the final model used
for the published results.

`filtered/` contains historical intermediate preprocessing and feature-derived
data. The four feature-specific 2DSA files are retained because the preserved
model-search configurations name them as inputs. The uncompressed 2DSA
preprocessing output was removed because it duplicated
`../datasets/2dsa_preprocessed.csv.gz` exactly.

`deployed_model/` is a preserved 2DSA inference bundle with a model, scaler,
sample input, and inference script. It is historical deployment evidence, not
a verified selected model for the paper.
