#!/usr/bin/env python3
"""
Generate Table 3.2: Contrasting Predictability of Runtime and Memory
Evaluates pre-trained TEMPORAL models on TEMPORAL test split.

python analysis/07_comparative_analysis/03_runtime_vs_memory_temporal_eval.py \
  --results_root results/pearc2026/selected_models \
  --out_dir analysis/07_comparative_analysis/results/03

"""

import argparse
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score

# Expected temporal models from Table 2.5
EXPECTED = {
    ("2dsa", "cpu"): "64x32_std_adam_bs64_elu_do0.2",
    ("2dsa", "memory"): "128x64_std_rmsprop_bs128_elu_do0.2",
    ("ga", "cpu"): "128x64_std_adam_bs128_relu_do0.2",
    ("ga", "memory"): "128x64_std_adam_bs64_relu_do0.2",
    ("pcsa", "cpu"): "128x64_minmax_adam_bs64_relu_do0.2",
    ("pcsa", "memory"): "64x32_std_rmsprop_bs128_relu_do0.2",
}


def run_dir_to_model_id(dir_name: str) -> str:
    """
    Parse run directory name and return canonical model_id.

    Input format: {method}_arch_{a1}_{a2}_scaler_{standard|minmax}_opt_{adam|rmsprop}_batch_{64|128}_act_{relu|elu}_drop_{0.2|0.3}
    Output format: {a1}x{a2}_{scalerToken}_{optToken}_bs{batch}_{act}_do{drop}

    Mappings:
    - scaler: standard -> std, minmax -> minmax
    - optimizer: rms -> rmsprop, adam -> adam
    """
    # Pattern to extract components
    pattern = r'(?P<method>\w+)_arch_(?P<a1>\d+)_(?P<a2>\d+)_scaler_(?P<scaler>\w+)_opt_(?P<opt>\w+)_batch_(?P<batch>\d+)_act_(?P<act>\w+)_drop_(?P<drop>[\d.]+)'

    match = re.match(pattern, dir_name)
    if not match:
        raise ValueError(f"Cannot parse directory name: {dir_name}")

    parts = match.groupdict()

    # Map scaler token
    scaler_map = {"standard": "std", "minmax": "minmax"}
    scaler_token = scaler_map.get(parts["scaler"])
    if scaler_token is None:
        raise ValueError(f"Unknown scaler: {parts['scaler']}")

    # Map optimizer token
    opt_map = {"adam": "adam", "rms": "rmsprop", "rmsprop": "rmsprop"}
    opt_token = opt_map.get(parts["opt"])
    if opt_token is None:
        raise ValueError(f"Unknown optimizer: {parts['opt']}")

    # Construct model_id
    model_id = f"{parts['a1']}x{parts['a2']}_{scaler_token}_{opt_token}_bs{parts['batch']}_{parts['act']}_do{parts['drop']}"

    return model_id


def select_run_directory(method: str, target: str, results_root: Path) -> Path:
    """
    Deterministically select the run directory matching EXPECTED[(method, target)].

    Raises RuntimeError if no match or multiple matches found.
    """
    expected_id = EXPECTED[(method, target)]
    method_dir = results_root / method / target / "raw" / "temporal"

    if not method_dir.exists():
        raise FileNotFoundError(f"Temporal directory not found: {method_dir}")

    # List all subdirectories
    run_dirs = [d for d in method_dir.iterdir() if d.is_dir()]

    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {method_dir}")

    # Compute model_id for each directory
    dir_to_id = {}
    for run_dir in run_dirs:
        try:
            model_id = run_dir_to_model_id(run_dir.name)
            dir_to_id[run_dir] = model_id
        except ValueError as e:
            # Skip directories that don't match the expected pattern
            continue

    # Find matches
    matches = [d for d, mid in dir_to_id.items() if mid == expected_id]

    if len(matches) == 0:
        # No match found - print diagnostic table
        print(f"\nERROR: No run directory matches expected model_id for ({method}, {target})")
        print(f"Expected: {expected_id}")
        print("\nAvailable directories:")
        print(f"{'Directory Name':<80} {'Computed Model ID':<50}")
        print("-" * 130)
        for run_dir in sorted(run_dirs):
            computed_id = dir_to_id.get(run_dir, "PARSE_FAILED")
            print(f"{run_dir.name:<80} {computed_id:<50}")
        raise RuntimeError(f"No matching run directory for ({method}, {target})")

    if len(matches) > 1:
        # Multiple matches found - print diagnostic table
        print(f"\nERROR: Multiple run directories match expected model_id for ({method}, {target})")
        print(f"Expected: {expected_id}")
        print("\nMatching directories:")
        for match in matches:
            print(f"  {match.name}")
        raise RuntimeError(f"Multiple matching run directories for ({method}, {target})")

    return matches[0]


def load_raw_feature_columns(path: Path) -> List[str]:
    """Load raw feature column names from text file."""
    if not path.exists():
        raise FileNotFoundError(f"Raw feature column list not found: {path}")
    with open(path, "r") as f:
        columns = [line.strip() for line in f if line.strip()]
    return columns


def discover_model_artifacts(models_dir: Path) -> Tuple[Path, Path]:
    """
    Discover exactly one .keras and one .pkl file in models directory.
    Fails loudly if count != 1 for either.
    """
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    keras_files = list(models_dir.glob("*.keras"))
    pkl_files = list(models_dir.glob("*_scaler.pkl"))

    if len(keras_files) != 1:
        raise ValueError(
            f"Expected exactly 1 .keras file in {models_dir}, found {len(keras_files)}"
        )
    if len(pkl_files) != 1:
        raise ValueError(
            f"Expected exactly 1 *_scaler.pkl file in {models_dir}, found {len(pkl_files)}"
        )

    return keras_files[0], pkl_files[0]


def temporal_split_indices(n: int) -> Tuple[int, int, int]:
    """
    Compute train/val/test split indices matching training logic.
    Returns (train_size, val_size, test_size).
    """
    train_size = int(n * 0.70)
    val_size = int(n * 0.15)
    test_size = n - train_size - val_size
    return train_size, val_size, test_size


def evaluate_model(
        method: str,
        target: str,
        data_csv: Path,
        raw_columns_file: Path,
        results_root: Path,
        out_dir: Path,
) -> Dict:
    """
    Evaluate a single (method, target) model on temporal test split.
    Returns metrics dict.
    """
    # Target column mapping
    target_col = "CPUTime" if target == "cpu" else "max_rss"

    # Load dataset
    if not data_csv.exists():
        raise FileNotFoundError(f"Dataset not found: {data_csv}")
    df = pd.read_csv(data_csv)

    # Verify required columns
    required_cols = ["submitTime", "CPUTime", "max_rss"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {data_csv}: {missing}")

    # Load raw feature columns
    raw_columns = load_raw_feature_columns(raw_columns_file)
    missing_features = [c for c in raw_columns if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"Missing raw feature columns in {data_csv}: {missing_features}"
        )

    # Build X, y
    X = df[raw_columns].values
    y = df[target_col].values
    submit_times = df["submitTime"].values

    # Sort by submitTime ascending
    sort_idx = np.argsort(submit_times)
    X = X[sort_idx]
    y = y[sort_idx]
    submit_times = submit_times[sort_idx]

    # Temporal split
    n_total = len(X)
    train_size, val_size, test_size = temporal_split_indices(n_total)
    test_start = train_size + val_size

    X_test = X[test_start:]
    y_test = y[test_start:]
    submit_times_test = submit_times[test_start:]

    # Select run directory deterministically
    run_dir = select_run_directory(method, target, results_root)
    models_dir = run_dir / "models"

    model_path, scaler_path = discover_model_artifacts(models_dir)

    # Load scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Scale test data
    X_test_scaled = scaler.transform(X_test)

    # Load model and predict
    model = tf.keras.models.load_model(model_path)
    y_pred = model.predict(X_test_scaled, verbose=0)
    y_pred = y_pred.flatten()

    # Compute metrics
    error = y_pred - y_test
    mae = np.mean(np.abs(error))
    std = np.std(error)
    r2 = r2_score(y_test, y_pred)

    # Console summary
    print(f"\n{method.upper()} - {target}:")
    print(f"  n_test={test_size}, MAE={mae:.2f}, Std={std:.2f}, R²={r2:.4f}")

    # Save per-job predictions
    preds_df = pd.DataFrame({
        "idx_in_sorted_df": np.arange(test_start, n_total),
        "submitTime": submit_times_test,
        "y_true": y_test,
        "y_pred": y_pred,
        "error": error,
    })
    preds_csv = out_dir / f"preds_{method}_{target}.csv"
    preds_df.to_csv(preds_csv, index=False)

    return {
        "method": method,
        "target": target,
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "n_total": n_total,
        "n_test": test_size,
        "train_size": train_size,
        "val_size": val_size,
        "MAE": mae,
        "Std": std,
        "R2": r2,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate Table 3.2: Runtime vs Memory Temporal Evaluation"
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        default=Path("archive/pearc2026/results/selected_models"),
        help="Root directory containing model results",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("archive/pearc2026/results/reproduced"),
        help="Output directory for tables",
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

    # Collect results
    all_results = []

    for method in methods:
        for target in targets:
            result = evaluate_model(
                method=method,
                target=target,
                data_csv=data_csvs[method],
                raw_columns_file=raw_columns_files[method],
                results_root=args.results_root,
                out_dir=args.out_dir,
            )
            all_results.append(result)

    # Build main table (3 rows × 6 metric columns)
    table_rows = []
    for method in methods:
        row = {"method": method.upper()}

        # Find results for this method
        cpu_res = next(r for r in all_results if r["method"] == method and r["target"] == "cpu")
        mem_res = next(r for r in all_results if r["method"] == method and r["target"] == "memory")

        row["MAE_CPUTime"] = cpu_res["MAE"]
        row["MAE_max_rss"] = mem_res["MAE"]
        row["Std_CPUTime"] = cpu_res["Std"]
        row["Std_max_rss"] = mem_res["Std"]
        row["R2_CPUTime"] = cpu_res["R2"]
        row["R2_max_rss"] = mem_res["R2"]

        table_rows.append(row)

    table_df = pd.DataFrame(table_rows)
    table_csv = args.out_dir / "table_3_2_temporal.csv"
    table_df.to_csv(table_csv, index=False)
    print(f"\n✓ Main table written to: {table_csv}")

    # Build audit table (6 rows)
    audit_rows = []
    for result in all_results:
        audit_rows.append({
            "method": result["method"].upper(),
            "target": result["target"],
            "run_dir": result["run_dir"],
            "model_path": result["model_path"],
            "scaler_path": result["scaler_path"],
            "n_total": result["n_total"],
            "n_test": result["n_test"],
            "train_size": result["train_size"],
            "val_size": result["val_size"],
        })

    audit_df = pd.DataFrame(audit_rows)
    audit_csv = args.out_dir / "table_3_2_temporal_audit.csv"
    audit_df.to_csv(audit_csv, index=False)
    print(f"✓ Audit table written to: {audit_csv}")
    print(f"\n✓ Per-job predictions written to: {args.out_dir}/preds_{{method}}_{{target}}.csv")


if __name__ == "__main__":
    main()
