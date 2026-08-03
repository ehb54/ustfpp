import os
import pandas as pd
import glob
from pathlib import Path

# Points to project root (one level up from the script)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, 'preprocess', 'raw')

# Output relative to script
OUTPUT_DIRECTORY = os.path.join(os.path.dirname(__file__), "results", "extracted")

# Define the column names in order
COLUMN_NAMES = [
    '@attributes.method', 'CPUCount',
    'edited_radial_points.0', 'edited_radial_points.1', 'edited_radial_points.2',
    'edited_radial_points.3', 'edited_radial_points.4', 'edited_radial_points.5',
    'edited_radial_points.6', 'edited_radial_points.7', 'edited_radial_points.8',
    'edited_radial_points.9',
    'edited_scans.0', 'edited_scans.1', 'edited_scans.2', 'edited_scans.3',
    'edited_scans.4', 'edited_scans.5', 'edited_scans.6', 'edited_scans.7',
    'edited_scans.8', 'edited_scans.9',
    'endTime',
    'job.cluster.@attributes.name',
    'job.jobParameters.bucket_fixed.@attributes.fixedtype',
    'job.jobParameters.bucket_fixed.@attributes.value',
    'job.jobParameters.bucket_fixed.@attributes.xtype',
    'job.jobParameters.bucket_fixed.@attributes.ytype',
    'job.jobParameters.conc_threshold.@attributes.value',
    'job.jobParameters.crossover.@attributes.value',
    'job.jobParameters.curve_type.@attributes.value',
    'job.jobParameters.curves_points.@attributes.value',
    'job.jobParameters.demes.@attributes.value',
    'job.jobParameters.elitism.@attributes.value',
    'job.jobParameters.ff0_grid_points.@attributes.value',
    'job.jobParameters.ff0_max.@attributes.value',
    'job.jobParameters.ff0_min.@attributes.value',
    'job.jobParameters.ff0_resolution.@attributes.value',
    'job.jobParameters.generations.@attributes.value',
    'job.jobParameters.gfit_iterations.@attributes.value',
    'job.jobParameters.k_grid.@attributes.value',
    'job.jobParameters.max_iterations.@attributes.value',
    'job.jobParameters.mc_iterations.@attributes.value',
    'job.jobParameters.meniscus_points.@attributes.value',
    'job.jobParameters.meniscus_range.@attributes.value',
    'job.jobParameters.migration.@attributes.value',
    'job.jobParameters.mutate_sigma.@attributes.value',
    'job.jobParameters.mutation.@attributes.value',
    'job.jobParameters.p_mutate_k.@attributes.value',
    'job.jobParameters.p_mutate_s.@attributes.value',
    'job.jobParameters.p_mutate_sk.@attributes.value',
    'job.jobParameters.plague.@attributes.value',
    'job.jobParameters.population.@attributes.value',
    'job.jobParameters.regularization.@attributes.value',
    'job.jobParameters.req_mgroupcount.@attributes.value',
    'job.jobParameters.rinoise_option.@attributes.value',
    'job.jobParameters.s_grid.@attributes.value',
    'job.jobParameters.s_grid_points.@attributes.value',
    'job.jobParameters.s_max.@attributes.value',
    'job.jobParameters.s_min.@attributes.value',
    'job.jobParameters.s_resolution.@attributes.value',
    'job.jobParameters.seed.@attributes.value',
    'job.jobParameters.solute_type.@attributes.value',
    'job.jobParameters.thr_deltr_ratio.@attributes.value',
    'job.jobParameters.tikreg_alpha.@attributes.value',
    'job.jobParameters.tikreg_option.@attributes.value',
    'job.jobParameters.tinoise_option.@attributes.value',
    'job.jobParameters.uniform_grid.@attributes.value',
    'job.jobParameters.vars_count.@attributes.value',
    'job.jobParameters.x_max.@attributes.value',
    'job.jobParameters.x_min.@attributes.value',
    'job.jobParameters.y_max.@attributes.value',
    'job.jobParameters.y_min.@attributes.value',
    'job.jobParameters.z_value.@attributes.value',
    'simpoints.0', 'simpoints.1', 'simpoints.2', 'simpoints.3', 'simpoints.4',
    'simpoints.5', 'simpoints.6', 'simpoints.7', 'simpoints.8', 'simpoints.9',
    'startTime', 'submitTime', 'updateTime',
    'CPUTime', 'max_rss', 'wallTime'
]


def validate_column_names(file_path, expected_columns):
    """
    Check if the first line of the file contains column headers and compare with expected columns.
    Returns a tuple of (missing_columns, extra_columns, has_header)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            fields = first_line.split()

            # Check if first line looks like a header (contains non-numeric values)
            has_header = any(
                not field.replace('.', '').replace('-', '').replace('@', '').isdigit() for field in fields[:5])

            if has_header:
                actual_columns = fields
                expected_set = set(expected_columns)
                actual_set = set(actual_columns)

                missing_columns = expected_set - actual_set
                extra_columns = actual_set - expected_set

                return list(missing_columns), list(extra_columns), True
            else:
                return [], [], False

    except Exception as e:
        print(f"Warning: Could not validate columns for {file_path}: {str(e)}")
        return [], [], False


def read_space_delimited(file_path):
    try:
        lines = []
        error_lines = []
        expected_fields = len(COLUMN_NAMES)

        # Validate column names first
        missing_cols, extra_cols, has_header = validate_column_names(file_path, COLUMN_NAMES)

        if has_header:
            print(f"\n{'=' * 80}")
            print(f"COLUMN VALIDATION for {os.path.basename(file_path)}")
            print(f"{'=' * 80}")

            if missing_cols:
                print(f"\n  MISSING COLUMNS ({len(missing_cols)}):")
                for col in sorted(missing_cols):
                    print(f"   - {col}")
            else:
                print("\n✓ No missing columns")

            if extra_cols:
                print(f"\n️  EXTRA COLUMNS ({len(extra_cols)}):")
                for col in sorted(extra_cols):
                    print(f"   + {col}")
            else:
                print("\n✓ No extra columns")

            print(f"\n{'=' * 80}\n")

        with open(file_path, 'r', encoding='utf-8') as f:
            # Skip header line if present
            if has_header:
                next(f)

            for line_num, line in enumerate(f, 2 if has_header else 1):
                # Split on whitespace (spaces AND tabs), filter out empty strings
                fields = line.strip().split()

                # Take only the expected number of fields
                # This handles the case where extra data (like version info) is appended
                fields = fields[:expected_fields]

                if len(fields) != expected_fields:
                    error_lines.append({
                        'file': file_path,
                        'line_number': line_num,
                        'raw_line': line.strip(),
                        'field_count': len(fields),
                        'expected_fields': expected_fields,
                        'error_type': 'incorrect_field_count'
                    })

                if '  ' in line:
                    error_lines.append({
                        'file': file_path,
                        'line_number': line_num,
                        'raw_line': line.strip(),
                        'field_count': len(fields),
                        'expected_fields': expected_fields,
                        'error_type': 'extra_spaces'
                    })

                lines.append(fields)

        df = pd.DataFrame(lines)
        df.columns = COLUMN_NAMES[:len(df.columns)]

        return df, '@attributes.method', error_lines

    except Exception as e:
        raise Exception(f"Failed to read file: {str(e)}")


def find_and_combine_metadata(data_dir):
    experiment_dfs = {
        '2DSA': [],
        'GA': [],
        'PCSA': []
    }
    metrics = {
        'total_files': 0,
        'files_with_errors': 0,
        'initial_counts': {'2DSA': 0, 'GA': 0, 'PCSA': 0},
        'records_removed': {'2DSA': 0, 'GA': 0, 'PCSA': 0},
        'final_counts': {'2DSA': 0, 'GA': 0, 'PCSA': 0},
        'column_counts': {'2DSA': 0, 'GA': 0, 'PCSA': 0}
    }

    all_errors = []
    error_files = []

    # Look for files in results/raw directory
    raw_dir = os.path.join(data_dir)
    pattern = os.path.join(raw_dir, '*')
    all_files = glob.glob(pattern)
    # Filter to only process files (not directories)
    all_files = [f for f in all_files if os.path.isfile(f)]
    metrics['total_files'] = len(all_files)

    print(f"Found {len(all_files)} file(s) in {raw_dir}")

    for file_path in all_files:
        print(f"\nProcessing file {metrics['total_files']}/{len(all_files)}: {file_path}")

        try:
            df, method_column, error_lines = read_space_delimited(file_path)
            all_errors.extend(error_lines)

            mappings = {
                '2DSA': [1, '1'],
                'GA': [3, '3'],
                'PCSA': [4, '4']
            }

            for exp_type, method_values in mappings.items():
                df_filtered = df[df[method_column].isin(method_values)].copy()

                if len(df_filtered) > 0:
                    experiment_dfs[exp_type].append(df_filtered)
                    metrics['initial_counts'][exp_type] += len(df_filtered)

        except Exception as e:
            error_message = f"Error processing {file_path}: {str(e)}"
            print(error_message)
            error_files.append((file_path, str(e)))
            metrics['files_with_errors'] += 1
            continue

    if all_errors:
        error_df = pd.DataFrame(all_errors)
        error_df.to_csv(os.path.join(OUTPUT_DIRECTORY, 'errors.csv'), index=False)

    combined_dfs = {}

    edited_columns = [f'edited_radial_points.{i}' for i in range(1, 10)]

    for exp_type, dfs in experiment_dfs.items():
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            # Convert columns to numeric
            combined_df[edited_columns] = combined_df[edited_columns].apply(pd.to_numeric, errors='coerce')

            # Filter rows where all specified edited_radial_points are zero
            initial_count = len(combined_df)
            filtered_df = combined_df[combined_df[edited_columns].sum(axis=1) == 0]
            metrics['records_removed'][exp_type] = initial_count - len(filtered_df)
            metrics['final_counts'][exp_type] = len(filtered_df)
            metrics['column_counts'][exp_type] = len(filtered_df.columns)

            combined_dfs[exp_type] = filtered_df

    return combined_dfs, metrics


def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory does not exist: {DATA_DIR}")
        return

    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    print(f"Starting search in: {DATA_DIR}")

    combined_data, metrics = find_and_combine_metadata(DATA_DIR)

    # Save metrics to CSV
    metrics_df = pd.DataFrame([{
        'total_files_processed': metrics['total_files'],
        'files_with_errors': metrics['files_with_errors'],
        '2DSA_initial_count': metrics['initial_counts']['2DSA'],
        '2DSA_records_removed': metrics['records_removed']['2DSA'],
        '2DSA_final_count': metrics['final_counts']['2DSA'],
        '2DSA_column_count': metrics['column_counts']['2DSA'],
        'GA_initial_count': metrics['initial_counts']['GA'],
        'GA_records_removed': metrics['records_removed']['GA'],
        'GA_final_count': metrics['final_counts']['GA'],
        'GA_column_count': metrics['column_counts']['GA'],
        'PCSA_initial_count': metrics['initial_counts']['PCSA'],
        'PCSA_records_removed': metrics['records_removed']['PCSA'],
        'PCSA_final_count': metrics['final_counts']['PCSA'],
        'PCSA_column_count': metrics['column_counts']['PCSA']
    }])

    metrics_df.to_csv(os.path.join(OUTPUT_DIRECTORY, 'extraction_metrics.csv'), index=False)

    if combined_data:
        for exp_type, df in combined_data.items():
            output_file = os.path.join(OUTPUT_DIRECTORY, f'{exp_type.lower()}_dataset.csv')
            df.to_csv(output_file, index=False)
            print(f"\n{exp_type} dataset saved to: {output_file}")
            print(f"Initial records: {metrics['initial_counts'][exp_type]}")
            print(f"Records removed: {metrics['records_removed'][exp_type]}")
            print(f"Final records: {metrics['final_counts'][exp_type]}")
            print(f"Number of columns: {metrics['column_counts'][exp_type]}")


if __name__ == "__main__":
    main()