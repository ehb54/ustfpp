#!/usr/bin/env python3
"""
Aggregate Grid Search Error Analysis Results

Dynamically discovers and combines grid search error analysis CSVs from a
hierarchical directory structure with split_mode support.

Expected hierarchy (discovered dynamically):
  <input_dir>/
    └── <method>/          (e.g., 2dsa, ga, pcsa)
         └── <target>/     (cpu or memory)
              └── raw/
                   └── <split_mode>/  (e.g., record, temporal - DISCOVERED DYNAMICALLY)
                        └── <grid_config_dir>/
                             ├── <method>_grid_search_results.csv
                             └── error_analysis/
                                  ├── <prefix>_train_worst_predictions_wallTime.csv
                                  ├── <prefix>_val_worst_predictions_wallTime.csv
                                  ├── <prefix>_test_worst_predictions_wallTime.csv
                                  ├── <prefix>_train_worst_predictions_max_rss.csv
                                  ├── <prefix>_val_worst_predictions_max_rss.csv
                                  ├── <prefix>_test_worst_predictions_max_rss.csv
                                  ├── <prefix>_train_worst_predictions_CPUTime.csv
                                  ├── <prefix>_val_worst_predictions_CPUTime.csv
                                  └── <prefix>_test_worst_predictions_CPUTime.csv

Output structure (CSV only):
  <output_dir>/
    ├── runs.csv
    ├── worst_predictions.csv
    ├── tail_summary_by_run.csv
    ├── best_by_target.csv
    ├── tail_vs_mean.csv
    └── missing_outputs_report.csv
"""

import os
import sys
import re
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict

TARGETS = ["wallTime", "max_rss", "CPUTime"]
SPLITS = ["train", "val", "test"]


def parse_config_from_dirname(dirname):
    """
    Extract hyperparameter configuration from directory name.

    Example: 2dsa_arch_64_32_scaler_minmax_opt_adam_batch_64_act_elu_drop_0.2
    Returns: {
        'architecture': '64_32',
        'scaler': 'minmax',
        'optimizer': 'adam',
        'batch_size': 64,
        'activation': 'elu',
        'dropout': 0.2
    }
    """
    config = {}

    # Extract architecture
    arch_match = re.search(r'arch_(\d+(?:_\d+)*)', dirname)
    if arch_match:
        config['architecture'] = arch_match.group(1)

    # Extract scaler
    scaler_match = re.search(r'scaler_(\w+)', dirname)
    if scaler_match:
        config['scaler'] = scaler_match.group(1)

    # Extract optimizer
    opt_match = re.search(r'opt_(\w+)', dirname)
    if opt_match:
        config['optimizer'] = opt_match.group(1)

    # Extract batch size
    batch_match = re.search(r'batch_(\d+)', dirname)
    if batch_match:
        config['batch_size'] = int(batch_match.group(1))

    # Extract activation
    act_match = re.search(r'act_(\w+)', dirname)
    if act_match:
        config['activation'] = act_match.group(1)

    # Extract dropout
    drop_match = re.search(r'drop_([\d.]+)', dirname)
    if drop_match:
        config['dropout'] = float(drop_match.group(1))

    return config


def find_grid_config_directories(input_dir):
    """
    Find all grid configuration directories under input_dir.

    Discovers split_mode directories dynamically by traversing:
      <input_dir>/<method>/<target>/<feature_space>/<split_mode>/<grid_config_dir>

    Also handles legacy paths without split_mode for backward compatibility.

    Returns: List of tuples (grid_config_path, metadata_dict)
    """
    input_path = Path(input_dir)
    grid_configs = []

    # Traverse: method -> target -> feature_space -> split_mode -> grid_config_dir
    for method_dir in input_path.iterdir():
        if not method_dir.is_dir():
            continue

        for target_dir in method_dir.iterdir():
            if not target_dir.is_dir():
                continue

            for feature_space_dir in target_dir.iterdir():
                if not feature_space_dir.is_dir():
                    continue

                # Check if this level contains split_mode directories or grid_config directories
                subdirs = [d for d in feature_space_dir.iterdir() if d.is_dir()]

                for subdir in subdirs:
                    # Check if subdir looks like a grid_config (has hyperparameter markers)
                    # vs a split_mode directory (simple name like "record" or "temporal")
                    is_likely_config = any(marker in subdir.name for marker in
                                           ['arch_', 'scaler_', 'opt_', 'batch_'])

                    if is_likely_config:
                        # Legacy path: feature_space -> grid_config (no split_mode)
                        metadata = {
                            'method': method_dir.name,
                            'target': target_dir.name,
                            'feature_space': feature_space_dir.name,
                            'split_mode': 'record',  # Assume legacy = record
                            'grid_config_name': subdir.name
                        }
                        grid_configs.append((subdir, metadata))
                    else:
                        # New path: feature_space -> split_mode -> grid_config
                        split_mode = subdir.name

                        for grid_config_dir in subdir.iterdir():
                            if not grid_config_dir.is_dir():
                                continue

                            metadata = {
                                'method': method_dir.name,
                                'target': target_dir.name,
                                'feature_space': feature_space_dir.name,
                                'split_mode': split_mode,
                                'grid_config_name': grid_config_dir.name
                            }

                            grid_configs.append((grid_config_dir, metadata))

    return grid_configs


def select_results_row(df):
    """
    Select one row from results CSV.
    Logic: single row -> use it; else best val metric if available; else last row.
    """
    if len(df) == 0:
        return None
    if len(df) == 1:
        return df.iloc[0]

    # Try to find best validation metric
    val_cols = [c for c in df.columns if
                'val' in c.lower() and ('rmse' in c.lower() or 'mae' in c.lower() or 'loss' in c.lower())]
    if val_cols:
        # Use first validation metric column, prefer lower values
        best_idx = df[val_cols[0]].idxmin()
        return df.loc[best_idx]

    # Default to last row
    return df.iloc[-1]


def build_runs_csv(grid_configs):
    """
    Build runs.csv: one row per config folder.
    Columns: method, target, feature_space, split_mode, grid_config_name, hyperparams,
             scalar metrics from results CSV.
    """
    runs = []

    print("\n" + "=" * 80)
    print("BUILDING runs.csv")
    print("=" * 80)

    for grid_config_dir, metadata in sorted(grid_configs, key=lambda x: str(x[0])):
        method = metadata['method'].lower()
        grid_config_name = metadata['grid_config_name']

        # Determine results CSV name based on method
        results_csv = grid_config_dir / f'{method}_grid_search_results.csv'

        if not results_csv.exists():
            print(f"  WARNING: {grid_config_name} missing {method}_grid_search_results.csv")
            continue

        try:
            results_df = pd.read_csv(results_csv)
            selected_row = select_results_row(results_df)

            if selected_row is None:
                print(f"  WARNING: {grid_config_name} has empty results CSV")
                continue

            # Parse hyperparameters
            config = parse_config_from_dirname(grid_config_name)

            # Build run record
            run_record = {
                'method': metadata['method'],
                'target': metadata['target'],
                'feature_space': metadata['feature_space'],
                'split_mode': metadata['split_mode'],
                'grid_config_name': grid_config_name,
                'grid_config_path': str(grid_config_dir)
            }

            # Add hyperparams
            for key, value in config.items():
                run_record[f'config_{key}'] = value

            # Add all scalar columns from selected row
            for col in selected_row.index:
                if pd.api.types.is_numeric_dtype(type(selected_row[col])):
                    run_record[col] = selected_row[col]
                elif isinstance(selected_row[col], (int, float, np.number)):
                    run_record[col] = selected_row[col]

            runs.append(run_record)
            print(
                f"  ✓ {metadata['method']}/{metadata['target']}/{metadata['feature_space']}/{metadata['split_mode']}/{grid_config_name}")

        except Exception as e:
            print(f"  ✗ ERROR processing {grid_config_name}: {e}")
            continue

    runs_df = pd.DataFrame(runs)
    print(f"\nTotal runs collected: {len(runs_df)}")
    return runs_df


def build_worst_predictions_csv(grid_configs):
    """
    Build worst_predictions.csv: concatenate all worst-prediction files.
    Columns: method, target, feature_space, split_mode, grid_config_name, split,
             prediction_target, rank, true_value, predicted_value,
             absolute_error, relative_error, source_path.
    """
    all_predictions = []

    print("\n" + "=" * 80)
    print("BUILDING worst_predictions.csv")
    print("=" * 80)

    for grid_config_dir, metadata in sorted(grid_configs, key=lambda x: str(x[0])):
        grid_config_name = metadata['grid_config_name']
        error_analysis_dir = grid_config_dir / 'error_analysis'

        if not error_analysis_dir.exists():
            continue

        for split in SPLITS:
            for target in TARGETS:
                # Find matching worst predictions file
                pattern = f"*_{split}_worst_predictions_{target}.csv"
                matching_files = list(error_analysis_dir.glob(pattern))

                if not matching_files:
                    continue

                for csv_file in matching_files:
                    try:
                        df = pd.read_csv(csv_file)

                        # Verify expected columns
                        expected_cols = {'true_value', 'predicted_value', 'absolute_error', 'relative_error'}
                        if not expected_cols.issubset(df.columns):
                            print(f"  WARNING: {csv_file.name} missing expected columns")
                            continue

                        # Sort by absolute_error descending and assign rank
                        df = df.sort_values('absolute_error', ascending=False).reset_index(drop=True)
                        df['rank'] = range(1, len(df) + 1)

                        # Add metadata
                        df['method'] = metadata['method']
                        df['target'] = metadata['target']
                        df['feature_space'] = metadata['feature_space']
                        df['split_mode'] = metadata['split_mode']
                        df['grid_config_name'] = grid_config_name
                        df['grid_config_path'] = str(grid_config_dir)
                        df['split'] = split
                        df['prediction_target'] = target
                        df['source_path'] = str(csv_file)

                        all_predictions.append(df)
                        print(
                            f"  ✓ {metadata['method']}/{metadata['target']}/{metadata['feature_space']}/{metadata['split_mode']}/{grid_config_name} {split} {target}: {len(df)} predictions")

                    except Exception as e:
                        print(f"  ✗ ERROR loading {csv_file.name}: {e}")
                        continue

    if not all_predictions:
        print("\n  WARNING: No worst prediction files found!")
        return pd.DataFrame()

    worst_df = pd.concat(all_predictions, ignore_index=True)
    print(f"\nTotal predictions collected: {len(worst_df):,}")
    return worst_df


def build_tail_summary_csv(worst_df):
    """
    Build tail_summary_by_run.csv: one row per (method, target, feature_space, split_mode,
                                                  grid_config_name, split, prediction_target).
    Columns: method, target, feature_space, split_mode, grid_config_name, split, prediction_target,
             count_topN, mean/median/max absolute_error, mean/median/max relative_error.
    """
    print("\n" + "=" * 80)
    print("BUILDING tail_summary_by_run.csv")
    print("=" * 80)

    if worst_df.empty:
        print("  WARNING: No worst predictions data available")
        return pd.DataFrame()

    summary = worst_df.groupby(['method', 'target', 'feature_space', 'split_mode',
                                'grid_config_name', 'split', 'prediction_target']).agg(
        count_topN=('absolute_error', 'count'),
        mean_absolute_error=('absolute_error', 'mean'),
        median_absolute_error=('absolute_error', 'median'),
        max_absolute_error=('absolute_error', 'max'),
        mean_relative_error=('relative_error', 'mean'),
        median_relative_error=('relative_error', 'median'),
        max_relative_error=('relative_error', 'max')
    ).reset_index()

    print(f"Total summary rows: {len(summary)}")
    return summary


def build_best_by_target_csv(runs_df):
    """
    Build best_by_target.csv: one row per (method, target, feature_space, split_mode) combination.
    Choose best run by minimum test_rmse_<target> if present, else test_mae_<target>.
    Columns: method, target, feature_space, split_mode, prediction_target, best_grid_config_name,
             metric_used, metric_value, plus hyperparams.
    """
    print("\n" + "=" * 80)
    print("BUILDING best_by_target.csv")
    print("=" * 80)

    if runs_df.empty:
        print("  WARNING: No runs data available")
        return pd.DataFrame()

    best_runs = []

    # Group by method, target, feature_space, split_mode
    for (method, target, feature_space, split_mode), group in runs_df.groupby(
            ['method', 'target', 'feature_space', 'split_mode']
    ):
        for prediction_target in TARGETS:
            # Try RMSE first, then MAE
            rmse_col = f'test_rmse_{prediction_target}'
            mae_col = f'test_mae_{prediction_target}'

            metric_col = None
            if rmse_col in group.columns:
                metric_col = rmse_col
            elif mae_col in group.columns:
                metric_col = mae_col

            if metric_col is None:
                continue

            # Find best (minimum) metric
            valid_runs = group[group[metric_col].notna()]
            if valid_runs.empty:
                continue

            best_idx = valid_runs[metric_col].idxmin()
            best_run = valid_runs.loc[best_idx]

            best_record = {
                'method': method,
                'target': target,
                'feature_space': feature_space,
                'split_mode': split_mode,
                'prediction_target': prediction_target,
                'best_grid_config_name': best_run['grid_config_name'],
                'metric_used': metric_col,
                'metric_value': best_run[metric_col]
            }

            # Add hyperparams
            for col in best_run.index:
                if col.startswith('config_'):
                    best_record[col] = best_run[col]

            best_runs.append(best_record)
            print(
                f"  ✓ {method}/{target}/{feature_space}/{split_mode}/{prediction_target}: {best_run['grid_config_name']} ({metric_col}={best_run[metric_col]:.6f})")

    best_df = pd.DataFrame(best_runs)
    return best_df


def build_tail_vs_mean_csv(runs_df, tail_summary_df):
    """
    Build tail_vs_mean.csv: one row per (method, target, feature_space, split_mode,
                                         grid_config_name, prediction_target) using ONLY test split.
    Join runs.csv test metric with tail_summary_by_run.csv (test split).
    Columns: method, target, feature_space, split_mode, grid_config_name, prediction_target,
             test_metric_value, mean_absolute_error_topN, max_absolute_error.
    """
    print("\n" + "=" * 80)
    print("BUILDING tail_vs_mean.csv")
    print("=" * 80)

    if runs_df.empty or tail_summary_df.empty:
        print("  WARNING: Missing required data")
        return pd.DataFrame()

    # Filter tail summary to test split only
    test_tail = tail_summary_df[tail_summary_df['split'] == 'test'].copy()

    comparisons = []

    for prediction_target in TARGETS:
        # Get test metric column
        rmse_col = f'test_rmse_{prediction_target}'
        mae_col = f'test_mae_{prediction_target}'

        metric_col = None
        if rmse_col in runs_df.columns:
            metric_col = rmse_col
        elif mae_col in runs_df.columns:
            metric_col = mae_col

        if metric_col is None:
            continue

        # Get relevant tail data for this prediction target
        target_tail = test_tail[test_tail['prediction_target'] == prediction_target].copy()

        # Merge with runs
        for _, run in runs_df.iterrows():
            if pd.isna(run.get(metric_col)):
                continue

            tail_row = target_tail[
                (target_tail['method'] == run['method']) &
                (target_tail['target'] == run['target']) &
                (target_tail['feature_space'] == run['feature_space']) &
                (target_tail['split_mode'] == run['split_mode']) &
                (target_tail['grid_config_name'] == run['grid_config_name'])
                ]

            if tail_row.empty:
                continue

            tail_row = tail_row.iloc[0]

            comparisons.append({
                'method': run['method'],
                'target': run['target'],
                'feature_space': run['feature_space'],
                'split_mode': run['split_mode'],
                'grid_config_name': run['grid_config_name'],
                'prediction_target': prediction_target,
                'test_metric_used': metric_col,
                'test_metric_value': run[metric_col],
                'mean_absolute_error_topN': tail_row['mean_absolute_error'],
                'max_absolute_error': tail_row['max_absolute_error']
            })

    comparison_df = pd.DataFrame(comparisons)
    print(f"Total comparison rows: {len(comparison_df)}")
    return comparison_df


def build_missing_outputs_report(grid_configs):
    """
    Build missing_outputs_report.csv: one row per grid config.
    Boolean columns indicating presence/absence of results CSV and worst prediction files.
    """
    reports = []

    print("\n" + "=" * 80)
    print("BUILDING missing_outputs_report.csv")
    print("=" * 80)

    for grid_config_dir, metadata in sorted(grid_configs, key=lambda x: str(x[0])):
        method = metadata['method'].lower()
        grid_config_name = metadata['grid_config_name']

        report = {
            'method': metadata['method'],
            'target': metadata['target'],
            'feature_space': metadata['feature_space'],
            'split_mode': metadata['split_mode'],
            'grid_config_name': grid_config_name
        }

        # Check results CSV
        results_csv = grid_config_dir / f'{method}_grid_search_results.csv'
        report['has_results_csv'] = results_csv.exists()

        # Check error analysis directory
        error_analysis_dir = grid_config_dir / 'error_analysis'
        report['has_error_analysis_dir'] = error_analysis_dir.exists()

        # Check each worst predictions file
        for split in SPLITS:
            for target in TARGETS:
                pattern = f"*_{split}_worst_predictions_{target}.csv"
                col_name = f'has_{split}_{target}_worst_preds'

                if error_analysis_dir.exists():
                    matching = list(error_analysis_dir.glob(pattern))
                    report[col_name] = len(matching) > 0
                else:
                    report[col_name] = False

        reports.append(report)

    report_df = pd.DataFrame(reports)
    print(f"Total runs checked: {len(report_df)}")
    return report_df


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate grid search error analysis results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python aggregate_grid_search_errors.py --input-dir ./run_2025_12_27/results --output-dir aggregated/
"""
    )

    parser.add_argument('--input-dir', required=True,
                        help='Base directory containing results hierarchy')
    parser.add_argument('--output-dir', default='aggregated',
                        help='Output directory for aggregated CSV files (default: aggregated/)')

    args = parser.parse_args()

    # Validate input directory
    if not Path(args.input_dir).exists():
        print(f"ERROR: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GRID SEARCH ERROR ANALYSIS AGGREGATION")
    print("=" * 80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    # Find all grid configurations (discovers split_mode dynamically)
    print("\n" + "=" * 80)
    print("DISCOVERING GRID CONFIGURATIONS")
    print("=" * 80)
    grid_configs = find_grid_config_directories(args.input_dir)
    print(f"Found {len(grid_configs)} grid configurations")

    # Report discovered split modes
    split_modes = set(metadata['split_mode'] for _, metadata in grid_configs)
    print(f"Discovered split_modes: {sorted(split_modes)}")

    if not grid_configs:
        print("\nERROR: No grid configurations found!")
        sys.exit(1)

    # Build all CSV outputs
    runs_df = build_runs_csv(grid_configs)
    if not runs_df.empty:
        runs_path = output_path / 'runs.csv'
        runs_df.to_csv(runs_path, index=False)
        print(f"\n✓ Saved: runs.csv ({len(runs_df)} runs)")

    worst_df = build_worst_predictions_csv(grid_configs)
    if not worst_df.empty:
        worst_path = output_path / 'worst_predictions.csv'
        worst_df.to_csv(worst_path, index=False)
        print(f"✓ Saved: worst_predictions.csv ({len(worst_df):,} predictions)")

    tail_summary_df = build_tail_summary_csv(worst_df)
    if not tail_summary_df.empty:
        tail_path = output_path / 'tail_summary_by_run.csv'
        tail_summary_df.to_csv(tail_path, index=False)
        print(f"✓ Saved: tail_summary_by_run.csv ({len(tail_summary_df)} rows)")

    best_df = build_best_by_target_csv(runs_df)
    if not best_df.empty:
        best_path = output_path / 'best_by_target.csv'
        best_df.to_csv(best_path, index=False)
        print(f"✓ Saved: best_by_target.csv ({len(best_df)} rows)")

    tail_vs_mean_df = build_tail_vs_mean_csv(runs_df, tail_summary_df)
    if not tail_vs_mean_df.empty:
        comparison_path = output_path / 'tail_vs_mean.csv'
        tail_vs_mean_df.to_csv(comparison_path, index=False)
        print(f"✓ Saved: tail_vs_mean.csv ({len(tail_vs_mean_df)} comparisons)")

    missing_df = build_missing_outputs_report(grid_configs)
    if not missing_df.empty:
        missing_path = output_path / 'missing_outputs_report.csv'
        missing_df.to_csv(missing_path, index=False)
        print(f"✓ Saved: missing_outputs_report.csv ({len(missing_df)} runs)")

    print("\n" + "=" * 80)
    print("AGGREGATION COMPLETE")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Files created: 6 CSV files")


if __name__ == '__main__':
    main()
