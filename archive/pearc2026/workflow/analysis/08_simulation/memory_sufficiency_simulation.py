#!/usr/bin/env python3
"""
Memory Sufficiency Simulation under Conservative Z-Score Gating
Evaluates pre-trained temporal max_rss models with Z-score gating and buffering.

python analysis/08_simulation/memory_sufficiency_simulation.py \\n    --out_dir analysis/08_simulation/results
"""

import argparse
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

# Fixed model selection for temporal max_rss models
EXPECTED_MODELS = {
    "2dsa": "128x64_std_rmsprop_bs128_elu_do0.2",
    "ga": "128x64_std_adam_bs64_relu_do0.2",
    "pcsa": "64x32_std_rmsprop_bs128_relu_do0.2",
}

# P90 absolute error buffers per method and Z-bin
BUFFERS = {
    "2dsa": {
        "<=1": 53.74,
        "(1,2]": 3446.64,
        "(2,3]": 4589.72,
        ">3": 333.98,
    },
    "ga": {
        "<=1": 147587.32,
        "(1,2]": 192923.83,
        "(2,3]": 344939.79,
        ">3": 136189.58,
    },
    "pcsa": {
        "<=1": 0.0,  # No samples - will warn
        "(1,2]": 531.73,
        "(2,3]": 7634.11,
        ">3": 13379.85,
    },
}


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


def find_model_directory(method: str, model_id: str, results_root: Path) -> Path:
    """Find the model directory matching the expected model_id."""
    expected_dir_name = model_id_to_dir_name(method, model_id)
    method_dir = results_root / method / "memory" / "raw" / "temporal"

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


def assign_z_bin(z: float) -> str:
    """Assign Z-score to bin."""
    if z <= 1:
        return "<=1"
    elif z <= 2:
        return "(1,2]"
    elif z <= 3:
        return "(2,3]"
    else:
        return ">3"


def simulate_memory_sufficiency(
        method: str,
        data_csv: Path,
        raw_columns_file: Path,
        results_root: Path,
        out_dir: Path,
) -> Dict:
    """
    Run memory sufficiency simulation for one method.
    Returns summary statistics dict.
    """
    print(f"\n{'=' * 80}")
    print(f"Processing {method.upper()}")
    print(f"{'=' * 80}")

    # Load dataset
    if not data_csv.exists():
        raise FileNotFoundError(f"Dataset not found: {data_csv}")
    df = pd.read_csv(data_csv)

    # Verify required columns
    if "submitTime" not in df.columns:
        raise ValueError(f"Missing submitTime column in {data_csv}")

    # Find max_rss column
    if "max_rss" in df.columns:
        target_col = "max_rss"
    elif "maxRSS" in df.columns:
        target_col = "maxRSS"
    else:
        raise ValueError(f"Neither max_rss nor maxRSS found in {data_csv}")

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

    X_train = X[:train_size]
    X_test = X[test_start:]
    y_test = y[test_start:]
    df_test = df_sorted.iloc[test_start:].reset_index(drop=True)

    print(f"Dataset size: {n_total} (train={train_size}, val={val_size}, test={test_size})")

    # Compute Z-scores for test set
    max_abs_z = compute_max_abs_zscore(X_train, X_test)
    z_bins = [assign_z_bin(z) for z in max_abs_z]

    # Load model
    model_id = EXPECTED_MODELS[method]
    model_dir = find_model_directory(method, model_id, results_root)
    models_dir = model_dir / "models"

    model_path, scaler_path = discover_model_artifacts(models_dir, method, model_id)

    print(f"Model directory: {model_dir}")
    print(f"Model file: {model_path.name}")
    print(f"Scaler file: {scaler_path.name}")

    # Load scaler and model
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    X_test_scaled = scaler.transform(X_test)

    model = tf.keras.models.load_model(model_path)
    y_pred = model.predict(X_test_scaled, verbose=0).flatten()

    # Apply gating and buffering
    buffers = BUFFERS[method]

    gate_status = []
    buffer_vals = []
    adjusted_vals = []
    sufficient = []
    shortfall = []

    for i in range(len(y_test)):
        z_bin = z_bins[i]
        z_val = max_abs_z[i]

        # Gate: gated_in if z <= 3, gated_out if z > 3
        if z_val <= 3:
            gate = "gated_in"

            # Get buffer
            buffer = buffers.get(z_bin, 0.0)
            if z_bin == "<=1" and method == "pcsa" and buffer == 0.0:
                print(f"Warning: PCSA z_bin <=1 has no samples, using buffer=0")

            # Adjusted prediction
            adjusted = y_pred[i] + buffer

            # Sufficiency
            is_sufficient = adjusted >= y_test[i]
            shortfall_val = max(0, y_test[i] - adjusted)

            gate_status.append(gate)
            buffer_vals.append(buffer)
            adjusted_vals.append(adjusted)
            sufficient.append(is_sufficient)
            shortfall.append(shortfall_val)
        else:
            gate = "gated_out"
            gate_status.append(gate)
            buffer_vals.append(np.nan)
            adjusted_vals.append(np.nan)
            sufficient.append(np.nan)
            shortfall.append(np.nan)

    # Build audit dataframe
    audit_df = df_test.copy()
    audit_df["max_abs_z"] = max_abs_z
    audit_df["z_bin"] = z_bins
    audit_df["gate_status"] = gate_status
    audit_df["pred_max_rss"] = y_pred
    audit_df["buffer"] = buffer_vals
    audit_df["adjusted_max_rss"] = adjusted_vals
    audit_df["observed_max_rss"] = y_test
    audit_df["sufficient"] = sufficient
    audit_df["shortfall"] = shortfall
    audit_df["model_path"] = str(model_path)
    audit_df["scaler_path"] = str(scaler_path)

    # Write audit CSV
    audit_csv = out_dir / f"audit_memory_{method}.csv"
    audit_df.to_csv(audit_csv, index=False)
    print(f"Audit CSV: {audit_csv}")

    # Compute summary statistics
    n_gated_in = sum(1 for g in gate_status if g == "gated_in")
    n_gated_out = sum(1 for g in gate_status if g == "gated_out")

    gated_in_sufficient = [s for s in sufficient if pd.notna(s)]
    n_sufficient = sum(gated_in_sufficient)
    n_insufficient = len(gated_in_sufficient) - n_sufficient

    gated_in_shortfall = [s for s in shortfall if pd.notna(s) and s > 0]

    summary = {
        "method": method.upper(),
        "n_total": test_size,
        "n_gated_in": n_gated_in,
        "n_gated_out": n_gated_out,
        "gated_in_rate": n_gated_in / test_size if test_size > 0 else 0,
        "gated_out_rate": n_gated_out / test_size if test_size > 0 else 0,
        "n_sufficient": n_sufficient,
        "n_insufficient": n_insufficient,
        "sufficient_rate_of_gated_in": n_sufficient / n_gated_in if n_gated_in > 0 else 0,
        "insufficient_rate_of_gated_in": n_insufficient / n_gated_in if n_gated_in > 0 else 0,
        "mean_shortfall": np.mean(gated_in_shortfall) if gated_in_shortfall else 0,
        "p95_shortfall": np.percentile(gated_in_shortfall, 95) if gated_in_shortfall else 0,
        "max_shortfall": np.max(gated_in_shortfall) if gated_in_shortfall else 0,
    }

    print(f"Gated in: {n_gated_in}/{test_size} ({summary['gated_in_rate']:.1%})")
    print(f"Sufficient: {n_sufficient}/{n_gated_in} ({summary['sufficient_rate_of_gated_in']:.1%})")
    print(f"Insufficient: {n_insufficient}/{n_gated_in} ({summary['insufficient_rate_of_gated_in']:.1%})")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Memory Sufficiency Simulation with Z-Score Gating"
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
        help="Output directory for results",
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

    # Run simulations
    summaries = []

    for method in methods:
        summary = simulate_memory_sufficiency(
            method=method,
            data_csv=data_csvs[method],
            raw_columns_file=raw_columns_files[method],
            results_root=args.results_root,
            out_dir=args.out_dir,
        )
        summaries.append(summary)

    # Write summary table
    summary_df = pd.DataFrame(summaries)
    summary_csv = args.out_dir / "table_4_memory_sufficiency.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"\n{'=' * 80}")
    print(f"✓ Summary table written to: {summary_csv}")
    print(f"✓ Audit CSVs written to: {args.out_dir}/audit_memory_{{method}}.csv")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
