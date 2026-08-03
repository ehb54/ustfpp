#!/usr/bin/env python3
"""
CPUTime Sufficiency Simulation under Conservative Z-Score Gating
Evaluates pre-trained temporal CPUTime models with Z-score gating and buffering.

python analysis/08_simulation/cpu_sufficiency_simulation.py \
    --out_dir analysis/08_simulation/results
"""

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


def load_selected_models(selected_model_table_path: Path, target: str = "CPUTime") -> Dict[str, Dict]:
    """
    Load selected models for the specified target from the selected model table.

    Args:
        selected_model_table_path: Path to selected_model_table.csv
        target: Target variable name (default: "CPUTime")

    Returns:
        Dict mapping method (lowercase) to model configuration dict
    """
    if not selected_model_table_path.exists():
        raise FileNotFoundError(
            f"Selected model table not found: {selected_model_table_path}\n"
            f"Please ensure the file exists at this location."
        )

    df = pd.read_csv(selected_model_table_path)

    # Verify required columns
    required_cols = ["method", "target", "split_mode", "model_id"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in selected model table: {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Filter for CPUTime target and Temporal-order split (case-insensitive)
    df_target = df[df["target"].str.lower() == target.lower()].copy()

    if len(df_target) == 0:
        available_targets = df["target"].unique().tolist()
        raise ValueError(
            f"No models found for target '{target}' in selected model table.\n"
            f"Available targets: {available_targets}"
        )

    # Filter for temporal split mode
    df_temporal = df_target[df_target["split_mode"].str.lower() == "temporal-order"].copy()

    if len(df_temporal) == 0:
        available_splits = df_target["split_mode"].unique().tolist()
        raise ValueError(
            f"No Temporal-order models found for target '{target}'.\n"
            f"Available split modes for {target}: {available_splits}"
        )

    # Build models dict
    models = {}
    for _, row in df_temporal.iterrows():
        method = row["method"].lower()
        models[method] = {
            "model_id": row["model_id"],
            "split_mode": row["split_mode"],
            "grid_config_name": row.get("grid_config_name", ""),
        }

    print(f"Loaded {len(models)} selected models for target '{target}':")
    for method, info in models.items():
        print(f"  {method.upper()}: {info['model_id']}")

    return models


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


def model_id_to_dir_name(method: str, model_id: str) -> str:
    """
    Convert model_id to directory name pattern.

    Input: "128x64_std_rmsprop_bs128_elu_do0.2"
    Output: "2dsa_arch_128_64_scaler_standard_opt_rmsprop_batch_128_act_elu_drop_0.2"
    """
    # Parse model_id
    # Format: {arch}_{scaler}_{opt}_bs{batch}_{act}_do{drop}
    pattern = r'(?P<arch>\d+x\d+)_(?P<scaler>\w+)_(?P<opt>\w+)_bs(?P<batch>\d+)_(?P<act>\w+)_do(?P<drop>[\d.]+)'
    match = re.match(pattern, model_id)

    if not match:
        raise ValueError(f"Cannot parse model_id: {model_id}")

    parts = match.groupdict()

    # Convert arch: 128x64 -> 128_64
    arch = parts["arch"].replace("x", "_")

    # Convert scaler: std -> standard, minmax -> minmax
    scaler_map = {"std": "standard", "minmax": "minmax"}
    scaler = scaler_map.get(parts["scaler"])
    if scaler is None:
        raise ValueError(f"Unknown scaler: {parts['scaler']}")

    # Optimizer: rmsprop/adam -> rmsprop/adam
    opt = parts["opt"]

    # Construct directory name
    dir_name = f"{method}_arch_{arch}_scaler_{scaler}_opt_{opt}_batch_{parts['batch']}_act_{parts['act']}_drop_{parts['drop']}"

    return dir_name


def find_model_directory(method: str, model_id: str, results_root: Path, target: str = "cpu") -> Path:
    """Find the model directory matching the expected model_id."""
    expected_dir_name = model_id_to_dir_name(method, model_id)
    method_dir = results_root / method / target / "raw" / "temporal"

    if not method_dir.exists():
        raise FileNotFoundError(f"Temporal directory not found: {method_dir}")

    # Look for exact match
    model_dir = method_dir / expected_dir_name

    if not model_dir.exists():
        # List available directories for debugging
        available = [d.name for d in method_dir.iterdir() if d.is_dir()]
        raise FileNotFoundError(
            f"Expected model directory not found: {model_dir}\n"
            f"Available directories:\n" + "\n".join(f"  {d}" for d in available)
        )

    return model_dir


def discover_model_artifacts(models_dir: Path, method: str, model_id: str) -> Tuple[Path, Path]:
    """
    Discover .keras and .pkl files in models directory.
    If multiple, select based on filename patterns.
    """
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    keras_files = list(models_dir.glob("*.keras"))
    pkl_files = list(models_dir.glob("*_scaler.pkl"))

    if len(keras_files) == 0:
        raise ValueError(f"No .keras file found in {models_dir}")
    if len(pkl_files) == 0:
        raise ValueError(f"No *_scaler.pkl file found in {models_dir}")

    # If single files, use them
    if len(keras_files) == 1 and len(pkl_files) == 1:
        return keras_files[0], pkl_files[0]

    # Multiple files - select based on naming patterns
    method_upper = method.upper()

    # Parse model_id for tokens
    tokens = model_id.replace("x", "_").replace("bs", "_").replace("do", "_").split("_")

    # Select keras file
    keras_file = None
    for kf in keras_files:
        name = kf.name
        if method_upper in name or method in name:
            keras_file = kf
            break

    if keras_file is None:
        keras_file = keras_files[0]

    # Select scaler file
    scaler_file = None
    for sf in pkl_files:
        if sf.name.endswith("_scaler.pkl"):
            scaler_file = sf
            break

    if scaler_file is None:
        scaler_file = pkl_files[0]

    return keras_file, scaler_file


def compute_max_abs_zscore(X_train: np.ndarray, X_data: np.ndarray) -> np.ndarray:
    """
    Compute max absolute Z-score for each row in X_data.
    Statistics computed from X_train only.
    """
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0, ddof=1)

    # Avoid division by zero
    valid_features = sigma > 0

    Z = np.zeros_like(X_data)
    Z[:, valid_features] = (X_data[:, valid_features] - mu[valid_features]) / sigma[valid_features]

    max_abs_z = np.max(np.abs(Z), axis=1)

    return max_abs_z


def compute_calibration_buffer(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        max_abs_z: np.ndarray,
        z_gate: float,
        alpha: float,
) -> float:
    """
    Compute calibration buffer as quantile of residual on accepted calibration jobs.

    Args:
        y_true: True CPUTime values (calibration split)
        y_pred: Predicted CPUTime values (calibration split)
        max_abs_z: Max absolute z-scores (calibration split)
        z_gate: Z-score gate threshold
        alpha: Quantile level for buffer (e.g., 0.95 for P95)

    Returns:
        Buffer value (additive)
    """
    # Filter to accepted jobs only
    accepted = max_abs_z <= z_gate

    if not np.any(accepted):
        print(f"  Warning: No calibration jobs accepted at z_gate={z_gate}, using buffer=0")
        return 0.0

    # Compute residuals for accepted jobs
    residuals = y_true[accepted] - y_pred[accepted]

    # Compute quantile
    buffer = np.percentile(residuals, alpha * 100)

    # Clamp to non-negative
    buffer = max(0.0, buffer)

    n_accepted = np.sum(accepted)
    print(f"  Calibration: {n_accepted} accepted jobs, alpha={alpha:.2f} -> buffer={buffer:.2f}")

    return buffer


def simulate_cpu_sufficiency(
        method: str,
        data_csv: Path,
        raw_columns_file: Path,
        results_root: Path,
        out_dir: Path,
        selected_models: Dict[str, Dict],
        z_gate: float = 3.0,
        alpha: float = 0.95,
) -> Dict:
    """
    Run CPUTime sufficiency simulation for one method.
    Returns summary statistics dict.
    """
    print(f"\n{'=' * 80}")
    print(f"Processing {method.upper()}")
    print(f"{'=' * 80}")

    # Check if method has selected model
    if method not in selected_models:
        raise ValueError(
            f"No selected model found for method '{method}'.\n"
            f"Available methods: {list(selected_models.keys())}"
        )

    # Load dataset
    if not data_csv.exists():
        raise FileNotFoundError(f"Dataset not found: {data_csv}")
    df = pd.read_csv(data_csv)

    # Verify required columns
    if "submitTime" not in df.columns:
        raise ValueError(f"Missing submitTime column in {data_csv}")

    # Find CPUTime column
    if "cputime" in df.columns:
        target_col = "cputime"
    elif "CPUTime" in df.columns:
        target_col = "CPUTime"
    else:
        available_cols = [c for c in df.columns if "cpu" in c.lower() or "time" in c.lower()]
        raise ValueError(
            f"Neither cputime nor CPUTime found in {data_csv}.\n"
            f"Columns containing 'cpu' or 'time': {available_cols}"
        )

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
    df_sorted = df.iloc[sort_idx].reset_index(drop=True)

    # Temporal split
    n_total = len(X)
    train_size, val_size, test_size = temporal_split_indices(n_total)
    test_start = train_size + val_size
    calib_end = train_size + val_size  # Calibration = train + val

    X_train = X[:train_size]
    X_calib = X[:calib_end]
    y_calib = y[:calib_end]
    X_test = X[test_start:]
    y_test = y[test_start:]
    df_test = df_sorted.iloc[test_start:].reset_index(drop=True)

    print(f"Dataset size: {n_total} (train={train_size}, val={val_size}, test={test_size})")
    print(f"Calibration size: {calib_end} (train + val)")

    # Load model
    model_id = selected_models[method]["model_id"]
    model_dir = find_model_directory(method, model_id, results_root, target="cpu")
    models_dir = model_dir / "models"

    model_path, scaler_path = discover_model_artifacts(models_dir, method, model_id)

    print(f"Model directory: {model_dir}")
    print(f"Model file: {model_path.name}")
    print(f"Scaler file: {scaler_path.name}")

    # Load scaler and model
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    model = tf.keras.models.load_model(model_path)

    # Make predictions on calibration and test sets
    X_calib_scaled = scaler.transform(X_calib)
    y_pred_calib = model.predict(X_calib_scaled, verbose=0).flatten()

    X_test_scaled = scaler.transform(X_test)
    y_pred_test = model.predict(X_test_scaled, verbose=0).flatten()

    # Compute Z-scores for calibration and test sets
    max_abs_z_calib = compute_max_abs_zscore(X_train, X_calib)
    max_abs_z_test = compute_max_abs_zscore(X_train, X_test)

    # Compute buffer from calibration split only (NO test leakage)
    buffer = compute_calibration_buffer(
        y_true=y_calib,
        y_pred=y_pred_calib,
        max_abs_z=max_abs_z_calib,
        z_gate=z_gate,
        alpha=alpha,
    )

    # Apply gating and buffering to test set
    gate_status = []
    t_req_vals = []
    sufficient = []
    overhead_abs = []
    overhead_ratio = []

    for i in range(len(y_test)):
        z_val = max_abs_z_test[i]

        # Gate: accepted if z <= z_gate
        if z_val <= z_gate:
            gate = "accepted"

            # Requested time = prediction + buffer
            t_req = y_pred_test[i] + buffer

            # Sufficiency
            is_sufficient = t_req >= y_test[i]

            # Overhead (absolute)
            overhead_val = t_req - y_test[i]

            # Overhead ratio (handle division by zero)
            if y_test[i] > 0:
                overhead_r = t_req / y_test[i]
            else:
                overhead_r = np.nan

            gate_status.append(gate)
            t_req_vals.append(t_req)
            sufficient.append(is_sufficient)
            overhead_abs.append(overhead_val)
            overhead_ratio.append(overhead_r)
        else:
            gate = "rejected"
            gate_status.append(gate)
            t_req_vals.append(np.nan)
            sufficient.append(np.nan)
            overhead_abs.append(np.nan)
            overhead_ratio.append(np.nan)

    # Build audit dataframe
    audit_df = df_test.copy()
    audit_df["max_abs_z"] = max_abs_z_test
    audit_df["gate_status"] = gate_status
    audit_df["pred_cputime"] = y_pred_test
    audit_df["buffer"] = buffer
    audit_df["t_req"] = t_req_vals
    audit_df["observed_cputime"] = y_test
    audit_df["sufficient"] = sufficient
    audit_df["overhead_abs"] = overhead_abs
    audit_df["overhead_ratio"] = overhead_ratio
    audit_df["model_path"] = str(model_path)
    audit_df["scaler_path"] = str(scaler_path)
    audit_df["z_gate"] = z_gate
    audit_df["alpha"] = alpha

    # Write audit CSV
    audit_csv = out_dir / f"audit_cputime_{method}.csv"
    audit_df.to_csv(audit_csv, index=False)
    print(f"Audit CSV: {audit_csv}")

    # Compute summary statistics
    n_accepted = sum(1 for g in gate_status if g == "accepted")
    n_rejected = sum(1 for g in gate_status if g == "rejected")

    accepted_sufficient = [s for s in sufficient if pd.notna(s)]
    n_sufficient = sum(accepted_sufficient)
    n_insufficient = len(accepted_sufficient) - n_sufficient

    # Overhead statistics (for accepted jobs only)
    accepted_overhead_abs = [o for o in overhead_abs if pd.notna(o)]
    accepted_overhead_ratio = [o for o in overhead_ratio if pd.notna(o)]

    summary = {
        "method": method.upper(),
        "n_total": test_size,
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "accepted_fraction": n_accepted / test_size if test_size > 0 else 0,
        "rejected_fraction": n_rejected / test_size if test_size > 0 else 0,
        "n_sufficient": n_sufficient,
        "n_insufficient": n_insufficient,
        "sufficiency_rate": n_sufficient / n_accepted if n_accepted > 0 else 0,
        "insufficiency_rate": n_insufficient / n_accepted if n_accepted > 0 else 0,
        "buffer": buffer,
        "overhead_abs_median": np.median(accepted_overhead_abs) if accepted_overhead_abs else 0,
        "overhead_abs_mean": np.mean(accepted_overhead_abs) if accepted_overhead_abs else 0,
        "overhead_abs_p90": np.percentile(accepted_overhead_abs, 90) if accepted_overhead_abs else 0,
        "overhead_ratio_median": np.median(accepted_overhead_ratio) if accepted_overhead_ratio else 0,
        "overhead_ratio_p90": np.percentile(accepted_overhead_ratio, 90) if accepted_overhead_ratio else 0,
        "z_gate": z_gate,
        "alpha": alpha,
        "model_id": model_id,
    }

    print(f"Accepted: {n_accepted}/{test_size} ({summary['accepted_fraction']:.1%})")
    print(f"Sufficient: {n_sufficient}/{n_accepted} ({summary['sufficiency_rate']:.1%})")
    print(f"Insufficient: {n_insufficient}/{n_accepted} ({summary['insufficiency_rate']:.1%})")
    print(f"Buffer: {buffer:.2f}")
    print(f"Overhead (abs) - median: {summary['overhead_abs_median']:.2f}, p90: {summary['overhead_abs_p90']:.2f}")
    print(f"Overhead (ratio) - median: {summary['overhead_ratio_median']:.2f}, p90: {summary['overhead_ratio_p90']:.2f}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="CPUTime Sufficiency Simulation with Z-Score Gating"
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        default=Path("archive/pearc2026/results/selected_models"),
        help="Root directory containing model results",
    )
    parser.add_argument(
        "--selected_model_table",
        type=Path,
        default=Path("archive/pearc2026/results/tables/selected_model_table.csv"),
        help="Path to selected model table CSV",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("archive/pearc2026/results/reproduced"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--z_gate",
        type=float,
        default=3.0,
        help="Z-score gate threshold (default: 3.0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.95,
        help="Quantile level for buffer calibration (default: 0.95)",
    )
    args = parser.parse_args()

    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load selected models
    selected_models = load_selected_models(args.selected_model_table, target="CPUTime")

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

    # Filter to methods that have selected models
    available_methods = [m for m in methods if m in selected_models]

    if not available_methods:
        raise ValueError(
            f"No selected models found for any of the expected methods: {methods}\n"
            f"Available methods in selected model table: {list(selected_models.keys())}"
        )

    print(f"\n{'=' * 80}")
    print(f"CPUTime Sufficiency Simulation")
    print(f"{'=' * 80}")
    print(f"Configuration:")
    print(f"  Z-gate threshold: {args.z_gate}")
    print(f"  Buffer quantile (alpha): {args.alpha}")
    print(f"  Methods: {[m.upper() for m in available_methods]}")
    print(f"  Selected model table: {args.selected_model_table}")
    print(f"  Results root: {args.results_root}")
    print(f"  Output directory: {args.out_dir}")

    # Run simulations
    summaries = []

    for method in available_methods:
        summary = simulate_cpu_sufficiency(
            method=method,
            data_csv=data_csvs[method],
            raw_columns_file=raw_columns_files[method],
            results_root=args.results_root,
            out_dir=args.out_dir,
            selected_models=selected_models,
            z_gate=args.z_gate,
            alpha=args.alpha,
        )
        summaries.append(summary)

    # Write summary table
    summary_df = pd.DataFrame(summaries)
    summary_csv = args.out_dir / "cpu_sufficiency_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Write configuration metadata
    config = {
        "z_gate": args.z_gate,
        "alpha": args.alpha,
        "selected_model_table": str(args.selected_model_table),
        "results_root": str(args.results_root),
        "methods": available_methods,
        "target": "CPUTime",
    }
    config_json = args.out_dir / "cpu_sufficiency_config.json"
    with open(config_json, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"✓ Summary table written to: {summary_csv}")
    print(f"✓ Configuration written to: {config_json}")
    print(f"✓ Audit CSVs written to: {args.out_dir}/audit_cputime_{{method}}.csv")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
