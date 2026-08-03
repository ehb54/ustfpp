#!/usr/bin/env python3
"""
Temporal vs Record Split Comparison

Compares temporal split vs record split to quantify generalization
under ordering/time shift using aggregated results.

python analysis/07_comparative_analysis/06_temporal_vs_record.py \
  --runs_csv analysis/06_aggregate/results/runs.csv \
  --out_dir analysis/07_comparative_analysis/results/06

"""

import pandas as pd
import sys
from pathlib import Path


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Temporal vs record split comparison')
    parser.add_argument('--runs_csv', default='runs.csv', help='Path to runs.csv')
    parser.add_argument('--out_dir', default='.', help='Output directory')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top configs to evaluate')
    args = parser.parse_args()

    # Load data
    runs_path = Path(args.runs_csv)
    if not runs_path.exists():
        print(f"ERROR: File not found: {args.runs_csv}")
        sys.exit(1)

    df = pd.read_csv(runs_path)

    # Validate required columns
    required = ['method', 'target', 'feature_space', 'split_mode', 'grid_config_name', 'val_mae', 'test_mae']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        sys.exit(1)

    # Check for optional columns
    has_error_std = 'test_error_std' in df.columns

    # Verify split_mode contains record and temporal
    split_modes = set(df['split_mode'].unique())
    if 'record' not in split_modes or 'temporal' not in split_modes:
        print(f"ERROR: Expected split_mode to contain 'record' and 'temporal', found: {split_modes}")
        sys.exit(1)

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # TABLE A: Best Runs Side-by-Side
    # ========================================================================
    print("\n" + "=" * 80)
    print("TABLE A: BEST RUNS SIDE-BY-SIDE")
    print("=" * 80)

    # Select best run per (method, target, feature_space, split_mode)
    df_sorted = df.sort_values(
        by=['method', 'target', 'feature_space', 'split_mode', 'val_mae', 'test_mae', 'grid_config_name'],
        ascending=[True, True, True, True, True, True, True]
    )

    best_runs = df_sorted.groupby(['method', 'target', 'feature_space', 'split_mode']).first().reset_index()

    # Build comparison table
    methods = sorted(df['method'].unique())
    targets = sorted(df['target'].unique())
    feature_spaces = sorted(df['feature_space'].unique())

    table_a_rows = []

    for method in methods:
        for target in targets:
            for feature_space in feature_spaces:
                # Get record best
                record_run = best_runs[
                    (best_runs['method'] == method) &
                    (best_runs['target'] == target) &
                    (best_runs['feature_space'] == feature_space) &
                    (best_runs['split_mode'] == 'record')
                    ]

                # Get temporal best
                temporal_run = best_runs[
                    (best_runs['method'] == method) &
                    (best_runs['target'] == target) &
                    (best_runs['feature_space'] == feature_space) &
                    (best_runs['split_mode'] == 'temporal')
                    ]

                row = {
                    'method': method,
                    'target': target,
                    'feature_space': feature_space,
                }

                # Record metrics
                if len(record_run) > 0:
                    row['record_best_val_mae'] = record_run['val_mae'].iloc[0]
                    row['record_best_test_mae'] = record_run['test_mae'].iloc[0]
                    row['record_best_grid_config_name'] = record_run['grid_config_name'].iloc[0]
                    if has_error_std:
                        row['record_best_test_error_std'] = record_run['test_error_std'].iloc[0]
                else:
                    row['record_best_val_mae'] = None
                    row['record_best_test_mae'] = None
                    row['record_best_grid_config_name'] = None
                    if has_error_std:
                        row['record_best_test_error_std'] = None

                # Temporal metrics
                if len(temporal_run) > 0:
                    row['temporal_best_val_mae'] = temporal_run['val_mae'].iloc[0]
                    row['temporal_best_test_mae'] = temporal_run['test_mae'].iloc[0]
                    row['temporal_best_grid_config_name'] = temporal_run['grid_config_name'].iloc[0]
                    if has_error_std:
                        row['temporal_best_test_error_std'] = temporal_run['test_error_std'].iloc[0]
                else:
                    row['temporal_best_val_mae'] = None
                    row['temporal_best_test_mae'] = None
                    row['temporal_best_grid_config_name'] = None
                    if has_error_std:
                        row['temporal_best_test_error_std'] = None

                table_a_rows.append(row)

    table_a_df = pd.DataFrame(table_a_rows)

    # Save Table A
    table_a_path = out_dir / 'temporal_vs_record_best.csv'
    table_a_df.to_csv(table_a_path, index=False)
    print(f"✓ Saved: {table_a_path}")
    print(f"  Rows: {len(table_a_df)}")

    # ========================================================================
    # TABLE B: Generalization Gap
    # ========================================================================
    print("\n" + "=" * 80)
    print("TABLE B: GENERALIZATION GAP")
    print("=" * 80)

    table_b_rows = []

    for _, row_a in table_a_df.iterrows():
        row_b = {
            'method': row_a['method'],
            'target': row_a['target'],
            'feature_space': row_a['feature_space'],
        }

        # Compute gaps (temporal - record)
        # Positive means temporal performs worse
        if pd.notna(row_a['temporal_best_val_mae']) and pd.notna(row_a['record_best_val_mae']):
            row_b['gap_val_mae'] = row_a['temporal_best_val_mae'] - row_a['record_best_val_mae']
        else:
            row_b['gap_val_mae'] = None

        if pd.notna(row_a['temporal_best_test_mae']) and pd.notna(row_a['record_best_test_mae']):
            row_b['gap_test_mae'] = row_a['temporal_best_test_mae'] - row_a['record_best_test_mae']
        else:
            row_b['gap_test_mae'] = None

        if has_error_std:
            if pd.notna(row_a.get('temporal_best_test_error_std')) and pd.notna(
                    row_a.get('record_best_test_error_std')):
                row_b['gap_test_error_std'] = row_a['temporal_best_test_error_std'] - row_a[
                    'record_best_test_error_std']
            else:
                row_b['gap_test_error_std'] = None

        table_b_rows.append(row_b)

    table_b_df = pd.DataFrame(table_b_rows)

    # Save Table B
    table_b_path = out_dir / 'temporal_vs_record_gap.csv'
    table_b_df.to_csv(table_b_path, index=False)
    print(f"✓ Saved: {table_b_path}")
    print(f"  Rows: {len(table_b_df)}")

    # ========================================================================
    # TABLE C: Robustness Across Top-N Variants
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"TABLE C: ROBUSTNESS ACROSS TOP-{args.top_n} VARIANTS")
    print("=" * 80)

    table_c_rows = []

    for method in methods:
        for target in targets:
            for feature_space in feature_spaces:
                # Get all runs for this group
                group_record = df[
                    (df['method'] == method) &
                    (df['target'] == target) &
                    (df['feature_space'] == feature_space) &
                    (df['split_mode'] == 'record')
                    ]

                group_temporal = df[
                    (df['method'] == method) &
                    (df['target'] == target) &
                    (df['feature_space'] == feature_space) &
                    (df['split_mode'] == 'temporal')
                    ]

                # Sort and take top N
                record_sorted = group_record.sort_values(
                    by=['val_mae', 'test_mae', 'grid_config_name'],
                    ascending=[True, True, True]
                ).head(args.top_n)

                temporal_sorted = group_temporal.sort_values(
                    by=['val_mae', 'test_mae', 'grid_config_name'],
                    ascending=[True, True, True]
                ).head(args.top_n)

                row = {
                    'method': method,
                    'target': target,
                    'feature_space': feature_space,
                }

                # Record top-N stats
                if len(record_sorted) > 0:
                    row['record_topN_val_mae_median'] = record_sorted['val_mae'].median()
                    row['record_topN_val_mae_max'] = record_sorted['val_mae'].max()
                    row['record_topN_test_mae_median'] = record_sorted['test_mae'].median()
                    row['record_topN_test_mae_max'] = record_sorted['test_mae'].max()
                else:
                    row['record_topN_val_mae_median'] = None
                    row['record_topN_val_mae_max'] = None
                    row['record_topN_test_mae_median'] = None
                    row['record_topN_test_mae_max'] = None

                # Temporal top-N stats
                if len(temporal_sorted) > 0:
                    row['temporal_topN_val_mae_median'] = temporal_sorted['val_mae'].median()
                    row['temporal_topN_val_mae_max'] = temporal_sorted['val_mae'].max()
                    row['temporal_topN_test_mae_median'] = temporal_sorted['test_mae'].median()
                    row['temporal_topN_test_mae_max'] = temporal_sorted['test_mae'].max()
                else:
                    row['temporal_topN_val_mae_median'] = None
                    row['temporal_topN_val_mae_max'] = None
                    row['temporal_topN_test_mae_median'] = None
                    row['temporal_topN_test_mae_max'] = None

                # Compute gaps
                if row['temporal_topN_test_mae_median'] is not None and row['record_topN_test_mae_median'] is not None:
                    row['gap_topN_test_mae_median'] = row['temporal_topN_test_mae_median'] - row[
                        'record_topN_test_mae_median']
                else:
                    row['gap_topN_test_mae_median'] = None

                if row['temporal_topN_test_mae_max'] is not None and row['record_topN_test_mae_max'] is not None:
                    row['gap_topN_test_mae_max'] = row['temporal_topN_test_mae_max'] - row['record_topN_test_mae_max']
                else:
                    row['gap_topN_test_mae_max'] = None

                table_c_rows.append(row)

    table_c_df = pd.DataFrame(table_c_rows)

    # Save Table C
    table_c_path = out_dir / 'temporal_vs_record_topN.csv'
    table_c_df.to_csv(table_c_path, index=False)
    print(f"✓ Saved: {table_c_path}")
    print(f"  Rows: {len(table_c_df)}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEMPORAL VS RECORD COMPARISON COMPLETE")
    print("=" * 80)
    print(f"Output directory: {args.out_dir}")
    print("Files created:")
    print(f"  - temporal_vs_record_best.csv ({len(table_a_df)} rows)")
    print(f"  - temporal_vs_record_gap.csv ({len(table_b_df)} rows)")
    print(f"  - temporal_vs_record_topN.csv ({len(table_c_df)} rows)")


if __name__ == '__main__':
    main()
