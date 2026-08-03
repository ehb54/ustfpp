#!/usr/bin/env python3
"""
Generate Table 3.3: Parameter Z-Scores and Prediction Error under Temporal Evaluation
Analysis-only script using pre-computed predictions from Section 3.2.

python analysis/07_comparative_analysis/03_zscore_error_analysis.py \
  --out_dir analysis/07_comparative_analysis/results/03/zscore
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_raw_feature_columns(path: Path) -> List[str]:
    """Load raw feature column names from text file."""
    if not path.exists():
        raise FileNotFoundError(f"Raw feature column list not found: {path}")
    with open(path, "r") as f:
        columns = [line.strip() for line in f if line.strip()]
    return columns


def temporal_split_indices(n: int) -> Tuple[int, int, int]:
    """
    Compute train/val/test split indices matching training logic.
    Returns (train_size, val_size, test_size).
    """
    train_size = int(n * 0.70)
    val_size = int(n * 0.15)
    test_size = n - train_size - val_size
    return train_size, val_size, test_size


def compute_max_abs_zscore(
        X_train: np.ndarray,
        X_test: np.ndarray,
) -> np.ndarray:
    """
    Compute max absolute Z-score across all parameters for each test sample.

    Args:
        X_train: Training data (n_train, n_features)
        X_test: Test data (n_test, n_features)

    Returns:
        max_abs_z: Array of shape (n_test,) with max |Z| per test sample
    """
    # Compute mean and std from training data
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0, ddof=1)

    # Avoid division by zero - skip features with zero variance
    valid_features = sigma > 0

    if not np.any(valid_features):
        # All features have zero variance - return zeros
        return np.zeros(X_test.shape[0])

    # Compute Z-scores for test data (only for valid features)
    Z = np.zeros_like(X_test)
    Z[:, valid_features] = (X_test[:, valid_features] - mu[valid_features]) / sigma[valid_features]

    # Compute max absolute Z-score per sample
    max_abs_z = np.max(np.abs(Z), axis=1)

    return max_abs_z


def bin_zscore(z: float) -> str:
    """Assign Z-score to bin."""
    if z <= 1:
        return "<=1"
    elif z <= 2:
        return "(1,2]"
    elif z <= 3:
        return "(2,3]"
    else:
        return ">3"


def analyze_method_target(
        method: str,
        target: str,
        data_csv: Path,
        raw_columns_file: Path,
        preds_csv: Path,
) -> pd.DataFrame:
    """
    Analyze Z-score vs prediction error for one (method, target) pair.

    Returns DataFrame with columns: method, target, z_bin, median_abs_error, p90_abs_error, count
    """
    # Load dataset
    if not data_csv.exists():
        raise FileNotFoundError(f"Dataset not found: {data_csv}")
    df = pd.read_csv(data_csv)

    # Verify required columns
    if "submitTime" not in df.columns:
        raise ValueError(f"Missing submitTime column in {data_csv}")

    # Load raw feature columns
    raw_columns = load_raw_feature_columns(raw_columns_file)
    missing_features = [c for c in raw_columns if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"Missing raw feature columns in {data_csv}: {missing_features}"
        )

    # Build X (raw features only)
    X = df[raw_columns].values
    submit_times = df["submitTime"].values

    # Sort by submitTime ascending
    sort_idx = np.argsort(submit_times)
    X = X[sort_idx]

    # Temporal split
    n_total = len(X)
    train_size, val_size, test_size = temporal_split_indices(n_total)
    test_start = train_size + val_size

    X_train = X[:train_size]
    X_test = X[test_start:]

    # Compute max absolute Z-scores for test set
    max_abs_z = compute_max_abs_zscore(X_train, X_test)

    # Load predictions
    if not preds_csv.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_csv}")
    preds_df = pd.read_csv(preds_csv)

    # Verify preds_df has expected number of rows
    if len(preds_df) != test_size:
        raise ValueError(
            f"Prediction file {preds_csv} has {len(preds_df)} rows, "
            f"expected {test_size} (test set size)"
        )

    # Verify required columns in preds
    required_preds_cols = ["idx_in_sorted_df", "error"]
    missing_preds_cols = [c for c in required_preds_cols if c not in preds_df.columns]
    if missing_preds_cols:
        raise ValueError(
            f"Missing columns in {preds_csv}: {missing_preds_cols}"
        )

    # Compute absolute error
    abs_error = np.abs(preds_df["error"].values)

    # Assign Z-score bins
    z_bins = [bin_zscore(z) for z in max_abs_z]

    # Create analysis dataframe
    analysis_df = pd.DataFrame({
        "max_abs_z": max_abs_z,
        "z_bin": z_bins,
        "abs_error": abs_error,
    })

    # Compute metrics per bin
    bin_order = ["<=1", "(1,2]", "(2,3]", ">3"]
    results = []

    for z_bin in bin_order:
        bin_data = analysis_df[analysis_df["z_bin"] == z_bin]

        if len(bin_data) == 0:
            # Empty bin - use NaN
            results.append({
                "method": method.upper(),
                "target": target,
                "z_bin": z_bin,
                "median_abs_error": np.nan,
                "p90_abs_error": np.nan,
                "count": 0,
            })
        else:
            results.append({
                "method": method.upper(),
                "target": target,
                "z_bin": z_bin,
                "median_abs_error": np.median(bin_data["abs_error"]),
                "p90_abs_error": np.percentile(bin_data["abs_error"], 90),
                "count": len(bin_data),
            })

    return pd.DataFrame(results)


def print_summary(results_df: pd.DataFrame):
    """Print console summary showing trends per method/target."""
    print("\n" + "=" * 80)
    print("Z-SCORE BIN TRENDS (Median Absolute Error)")
    print("=" * 80)

    for method in ["2DSA", "GA", "PCSA"]:
        print(f"\n{method}:")

        for target in ["cpu", "memory"]:
            subset = results_df[
                (results_df["method"] == method) & (results_df["target"] == target)
                ].copy()

            if len(subset) == 0:
                continue

            # Ensure proper ordering
            bin_order = ["<=1", "(1,2]", "(2,3]", ">3"]
            subset["z_bin"] = pd.Categorical(subset["z_bin"], categories=bin_order, ordered=True)
            subset = subset.sort_values("z_bin")

            target_label = "CPUTime" if target == "cpu" else "max_rss"
            print(f"  {target_label}:")

            for _, row in subset.iterrows():
                if row["count"] == 0:
                    print(f"    {row['z_bin']:8s}: no samples")
                else:
                    print(
                        f"    {row['z_bin']:8s}: "
                        f"median={row['median_abs_error']:10.2f}, "
                        f"p90={row['p90_abs_error']:10.2f}, "
                        f"n={row['count']:5d}"
                    )

            # Check for monotonic trend
            valid_rows = subset[subset["count"] > 0]
            if len(valid_rows) >= 2:
                medians = valid_rows["median_abs_error"].values
                is_increasing = all(medians[i] <= medians[i + 1] for i in range(len(medians) - 1))
                is_decreasing = all(medians[i] >= medians[i + 1] for i in range(len(medians) - 1))

                if is_increasing:
                    print(f"    → Monotonic INCREASE (error rises with Z-score)")
                elif is_decreasing:
                    print(f"    → Monotonic DECREASE (error falls with Z-score)")
                else:
                    print(f"    → Non-monotonic trend")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Table 3.3: Parameter Z-Scores and Prediction Error"
    )
    parser.add_argument(
        "--preds_dir",
        type=Path,
        default=Path("archive/pearc2026/results/reproduced"),
        help="Directory containing preds_*.csv files from Section 3.2",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("archive/pearc2026/results/reproduced"),
        help="Output directory for table",
    )
    args = parser.parse_args()

    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Fixed data sources
    data_csvs = {
        "2dsa": Path("archive/pearc2026/datasets/2dsa_cleaned.csv.gz"),
        "ga": Path("archive/pearc2026/datasets/ga_cleaned.csv.gz"),
        "pcsa": Path("archive/pearc2026/datasets/pcsa_cleaned.csv.gz"),
    }

    raw_columns_files = {
        "2dsa": Path("archive/pearc2026/workflow/model_building/columns_files/raw_config_columns_2dsa.txt"),
        "ga": Path("archive/pearc2026/workflow/model_building/columns_files/raw_config_columns_ga.txt"),
        "pcsa": Path("archive/pearc2026/workflow/model_building/columns_files/raw_config_columns_pcsa.txt"),
    }

    methods = ["2dsa", "ga", "pcsa"]
    targets = ["cpu", "memory"]

    # Collect all results
    all_results = []

    print("Computing Z-score vs error analysis...")

    for method in methods:
        for target in targets:
            preds_csv = args.preds_dir / f"preds_{method}_{target}.csv"

            print(f"\nProcessing {method.upper()} - {target}...")

            result_df = analyze_method_target(
                method=method,
                target=target,
                data_csv=data_csvs[method],
                raw_columns_file=raw_columns_files[method],
                preds_csv=preds_csv,
            )

            all_results.append(result_df)

    # Combine all results
    final_df = pd.concat(all_results, ignore_index=True)

    # Write output
    output_csv = args.out_dir / "table_3_3_zscore_error.csv"
    final_df.to_csv(output_csv, index=False)

    print(f"\n✓ Table 3.3 written to: {output_csv}")

    # Print summary
    print_summary(final_df)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
