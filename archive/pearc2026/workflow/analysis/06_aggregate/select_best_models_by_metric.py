""""
Selects best-performing predictive models based on evaluation metrics and
produces a consolidated summary table. This class compares candidate models
across targets and feature spaces, applies consistent selection criteria, and
emits a structured representation suitable for downstream analysis and
publication.

runs from project root directory:
python analysis/06_aggregate/select_best_models_by_metric.py \
--runs_csv analysis/06_aggregate/results/runs.csv \
--out_csv analysis/06_aggregate/results/02/selected_model_table.csv
"""
#!/usr/bin/env python3

import argparse
import sys
import os
import re
import pandas as pd


def normalize_label(value, mapping):
    """Normalize a label using a case-insensitive mapping."""
    if pd.isna(value):
        return value
    val_lower = str(value).strip().lower()
    return mapping.get(val_lower, value)


def parse_grid_config_name(grid_config_name):
    """
    Parse configuration details from grid_config_name.

    Expected format: {method}_arch_{layers}_scaler_{scaler}_opt_{optimizer}_batch_{batch}_act_{activation}_drop_{dropout}
    Example: 2dsa_arch_128_64_scaler_minmax_opt_rmsprop_batch_64_act_relu_drop_0.3

    Returns: dict with keys: architecture, scaler, optimizer, batch_size, activation, dropout
    """
    config_str = str(grid_config_name).strip()

    # Extract using regex patterns
    patterns = {
        'architecture': r'arch_([^_]+(?:_\d+)*)',  # e.g., "128_64"
        'scaler': r'scaler_([^_]+)',  # e.g., "minmax"
        'optimizer': r'opt_([^_]+)',  # e.g., "rmsprop"
        'batch_size': r'batch_(\d+)',  # e.g., "64"
        'activation': r'act_([^_]+)',  # e.g., "relu"
        'dropout': r'drop_([\d.]+)',  # e.g., "0.3"
    }

    config = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, config_str)
        if match:
            config[key] = match.group(1)
        else:
            raise ValueError(f"Could not parse {key} from grid_config_name: {grid_config_name}")

    return config


def parse_layers(config_architecture):
    """
    Parse config_architecture into "x"-joined layer sizes.

    Accepts formats like:
    - "[128, 64]" or "(128, 64)" (list/tuple string representation)
    - "128,64" or "128, 64" (comma-separated)
    - "128_64" (underscore-separated)
    - "128 64" (space-separated)

    Returns: "128x64" (no brackets, x-joined)
    """
    arch_str = str(config_architecture).strip()

    # Remove brackets/parentheses
    arch_str = arch_str.strip('[](){}')

    # Extract all integers
    numbers = re.findall(r'\d+', arch_str)

    if not numbers:
        raise ValueError(f"Could not parse architecture: {config_architecture}")

    return 'x'.join(numbers)


def norm_scaler(config_scaler, grid_config_name):
    """
    Normalize scaler name to canonical token.

    Mappings:
    - "standard", "std", "zscore", "z-score", "z_score" → "std"
    - "minmax", "min-max", "min_max" → "minmax"

    Raises ValueError for unrecognized scaler values.
    """
    scaler_lower = str(config_scaler).strip().lower().replace('-', '_')

    # Standard scaler variants
    if scaler_lower in ['standard', 'std', 'zscore', 'z_score']:
        return 'std'

    # MinMax scaler variants
    if scaler_lower in ['minmax', 'min_max']:
        return 'minmax'

    # Unrecognized scaler
    raise ValueError(
        f"Unrecognized scaler value: '{config_scaler}' "
        f"in grid_config_name: {grid_config_name}"
    )


def norm_optimizer(config_optimizer):
    """
    Normalize optimizer name.

    Mappings:
    - "rms", "rms_prop", "rms-prop" → "rmsprop"
    - Others: lowercase, stripped (e.g., "adam" → "adam")

    Returns: normalized optimizer token
    """
    opt_lower = str(config_optimizer).strip().lower().replace('-', '_')

    # RMSprop variants
    if opt_lower in ['rms', 'rms_prop', 'rmsprop']:
        return 'rmsprop'

    # Everything else: use as-is (lowercase)
    return opt_lower


def fmt_dropout(config_dropout):
    """
    Format dropout value as minimal decimal string.

    Examples:
    - 0.30 → "0.3"
    - 0.2  → "0.2"
    - 0.0  → "0"
    - 0.05 → "0.05"

    Returns: string like "0.3" (no trailing zeros)
    """
    dropout_val = float(config_dropout)

    # Format with minimal precision (remove trailing zeros)
    formatted = f"{dropout_val:g}"

    return formatted


def create_short_model_id(row):
    """
    Create standardized model identifier from configuration details.

    Format: <layers>_<scaler>_<optimizer>_bs<batch>_<activation>_do<dropout>

    Example: 128x64_std_rmsprop_bs64_relu_do0.3

    Normalization rules:
    - layers: parsed from grid_config_name, x-joined integers (e.g., "128x64")
    - scaler: "std" or "minmax" (strict mapping, raises error on unknown values)
    - optimizer: normalized (e.g., "rmsprop", "adam")
    - batch: integer with "bs" prefix (e.g., "bs64")
    - activation: lowercase (e.g., "relu", "elu")
    - dropout: minimal decimal with "do" prefix (e.g., "do0.3", "do0.2")
    """
    # Parse config from grid_config_name (more reliable than config_* columns)
    config = parse_grid_config_name(row['grid_config_name'])

    # Parse architecture into layers
    layers = parse_layers(config['architecture'])

    # Normalize scaler (strict validation)
    scaler = norm_scaler(config['scaler'], row['grid_config_name'])

    # Normalize optimizer
    optimizer = norm_optimizer(config['optimizer'])

    # Format batch size
    batch = f"bs{int(config['batch_size'])}"

    # Normalize activation (lowercase)
    activation = str(config['activation']).strip().lower()

    # Format dropout
    dropout = f"do{fmt_dropout(config['dropout'])}"

    return f"{layers}_{scaler}_{optimizer}_{batch}_{activation}_{dropout}"


def main():
    parser = argparse.ArgumentParser(
        description="Select 12 models from grid-search results for publication table"
    )
    parser.add_argument(
        "--runs_csv",
        type=str,
        required=True,
        help="Path to runs.csv containing grid-search results"
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="selected_models_table.csv",
        help="Output CSV path (default: selected_models_table.csv)"
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="2dsa,ga,pcsa",
        help="Comma-separated list of methods (default: auto-detect, or 2dsa,ga,pcsa)"
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="cpu,memory",
        help="Comma-separated list of targets (default: cpu,memory)"
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="record,temporal",
        help="Comma-separated list of split modes (default: auto-detect, or record,temporal)"
    )

    args = parser.parse_args()

    # Label normalization mappings
    method_mapping = {"2dsa": "2DSA", "ga": "GA", "pcsa": "PCSA"}
    split_mapping = {"record": "Record-order", "temporal": "Temporal-order"}
    target_mapping = {"cpu": "CPUTime", "memory": "max_rss"}

    # Load data
    print(f"Loading {args.runs_csv}...")
    df = pd.read_csv(args.runs_csv)
    initial_rows = len(df)
    print(f"  Initial rows: {initial_rows}")

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include=['object']).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # Ensure val_mae is numeric
    df['val_mae'] = pd.to_numeric(df['val_mae'], errors='coerce')

    # Drop rows with NaN val_mae
    nan_rows = df['val_mae'].isna().sum()
    if nan_rows > 0:
        print(f"  Dropping {nan_rows} rows with NaN val_mae")
        df = df.dropna(subset=['val_mae'])

    cleaned_rows = len(df)
    print(f"  Rows after cleaning: {cleaned_rows}\n")

    # Auto-detect actual values in the data
    actual_methods = sorted([str(v).lower() for v in df['method'].dropna().unique()])
    actual_targets = sorted([str(v) for v in df['target'].dropna().unique()])
    actual_splits = sorted([str(v).lower() for v in df['split_mode'].dropna().unique()])
    actual_features = sorted([str(v).lower() for v in df['feature_space'].dropna().unique()])

    print("Auto-detected values in data:")
    print(f"  Methods: {actual_methods}")
    print(f"  Targets: {actual_targets}")
    print(f"  Splits: {actual_splits}")
    print(f"  Feature spaces: {actual_features}")
    print()

    # Parse expected combinations (use auto-detected if defaults provided)
    if args.methods == "2dsa,ga,pcsa":
        expected_methods = actual_methods
        print(f"Using auto-detected methods: {expected_methods}")
    else:
        expected_methods = [m.strip().lower() for m in args.methods.split(",")]
        print(f"Using specified methods: {expected_methods}")

    if args.targets == "cpu,memory":
    # Use the two published prediction targets.
        expected_targets = ['cpu', 'memory']
        print(f"Using default targets: {expected_targets}")
    else:
        expected_targets = [t.strip() for t in args.targets.split(",")]
        print(f"Using specified targets: {expected_targets}")

    if args.splits == "record,temporal":
        expected_splits = actual_splits
        print(f"Using auto-detected splits: {expected_splits}")
    else:
        expected_splits = [s.strip().lower() for s in args.splits.split(",")]
        print(f"Using specified splits: {expected_splits}")

    print()

    # Normalize method and split_mode for grouping (case-insensitive)
    df['method_lower'] = df['method'].str.lower()
    df['split_mode_lower'] = df['split_mode'].str.lower()

    # Filter to expected combinations (use .copy() to avoid SettingWithCopyWarning)
    # The published models use the raw feature space.
    df_filtered = df[
        df['method_lower'].isin(expected_methods) &
        df['target'].isin(expected_targets) &
        df['split_mode_lower'].isin(expected_splits) &
        (df['feature_space'].str.lower() == 'raw')
        ].copy()

    if len(df_filtered) == 0:
        print("ERROR: No rows match the expected method/target/split combinations.")
        print(f"\nExpected:")
        print(f"  Methods: {expected_methods}")
        print(f"  Targets: {expected_targets}")
        print(f"  Splits: {expected_splits}")
        print(f"  Feature space: raw only")
        sys.exit(1)

    print(f"Filtered to {len(df_filtered)} rows matching expected combinations (raw features only)\n")

    # Define tie-breaking columns in order
    tiebreak_cols = [
        'grid_config_name',
        'config_architecture',
        'config_scaler',
        'config_optimizer',
        'config_batch_size',
        'config_activation',
        'config_dropout'
    ]

    # Round val_mae to 6 decimals for tie detection
    df_filtered['val_mae_rounded'] = df_filtered['val_mae'].round(6)

    # Sort by group keys, then by val_mae, then by tie-breaking columns
    sort_cols = ['method_lower', 'target', 'split_mode_lower', 'val_mae_rounded'] + tiebreak_cols
    df_sorted = df_filtered.sort_values(by=sort_cols).reset_index(drop=True)

    # Select one row per group
    selected = df_sorted.groupby(
        ['method_lower', 'target', 'split_mode_lower'],
        as_index=False,
        dropna=False
    ).first()

    # Verify we have exactly 12 groups
    expected_count = len(expected_methods) * len(expected_targets) * len(expected_splits)
    if len(selected) != expected_count:
        print(f"ERROR: Expected {expected_count} groups, but found {len(selected)}.")
        print("Missing or extra groups detected.")

        # Show which groups we have
        print("\nGroups found:")
        for _, row in selected.iterrows():
            print(f"  {row['method_lower']}, {row['target']}, {row['split_mode_lower']}")

        # Show which groups are expected
        print("\nGroups expected:")
        for m in expected_methods:
            for t in expected_targets:
                for s in expected_splits:
                    found = ((selected['method_lower'] == m) &
                             (selected['target'] == t) &
                             (selected['split_mode_lower'] == s)).any()
                    status = "✓" if found else "✗"
                    print(f"  {status} {m}, {t}, {s}")

        sys.exit(1)

    # Normalize labels for output
    selected['method'] = selected['method_lower'].apply(
        lambda x: normalize_label(x, method_mapping)
    )
    selected['target'] = selected['target'].apply(
        lambda x: normalize_label(x, target_mapping)
    )
    selected['split_mode'] = selected['split_mode_lower'].apply(
        lambda x: normalize_label(x, split_mapping)
    )

    # Create standardized model identifier
    selected['model_id'] = selected.apply(create_short_model_id, axis=1)

    # Safety checks on model_id
    # 1. Check for null/empty values
    null_ids = selected['model_id'].isna() | (selected['model_id'] == '')
    if null_ids.any():
        print("ERROR: Found null or empty model_id values:")
        for _, row in selected[null_ids].iterrows():
            print(f"  {row['method']}/{row['target']}/{row['split_mode']}: {row['grid_config_name']}")
        sys.exit(1)

    # 2. Check for duplicate model_id values
    duplicate_ids = selected['model_id'].duplicated(keep=False)
    if duplicate_ids.any():
        print("ERROR: Found duplicate model_id values:")
        for model_id in selected[duplicate_ids]['model_id'].unique():
            print(f"\n  Duplicate model_id: '{model_id}'")
            dup_rows = selected[selected['model_id'] == model_id]
            for _, row in dup_rows.iterrows():
                print(f"    - {row['method']}/{row['target']}/{row['split_mode']}: {row['grid_config_name']}")
        sys.exit(1)

    # Print selection summary
    print("Selected models:")
    print("-" * 100)
    for _, row in selected.sort_values(['method', 'target', 'split_mode']).iterrows():
        print(f"{row['method']:6s} | {row['target']:10s} | {row['split_mode']:15s} | "
              f"{row['model_id']:35s} | val_mae={row['val_mae']:.6f}")
    print("-" * 100)
    print()

    # Prepare output columns in specified order
    output_cols = [
        'method',
        'target',
        'split_mode',
        'model_id',
        'grid_config_name',
        'feature_space',
        'config_architecture',
        'config_scaler',
        'config_optimizer',
        'config_batch_size',
        'config_activation',
        'config_dropout',
        'val_mae',
        'val_error_std',
        'epochs'
    ]

    # Verify all columns exist
    missing_cols = [col for col in output_cols if col not in selected.columns]
    if missing_cols:
        print(f"ERROR: Missing columns in input data: {missing_cols}")
        sys.exit(1)

    output_df = selected[output_cols].copy()

    # Final validation: exactly 12 rows
    if len(output_df) != 12:
        print(f"ERROR: Final output has {len(output_df)} rows, expected exactly 12.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.out_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}\n")

    # Write output
    output_df.to_csv(args.out_csv, index=False)
    print(f"Successfully wrote {len(output_df)} selected models to {args.out_csv}")


if __name__ == "__main__":
    main()
