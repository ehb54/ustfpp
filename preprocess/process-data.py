import os
import pandas as pd
import glob
from pathlib import Path

ROOT_DIRECTORY = "."
OUTPUT_DIRECTORY = "./results"

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
    'CPUTime', 'max_rss', 'wallTime'
]

def read_space_delimited(file_path):
    try:
        lines = []
        error_lines = []
        expected_fields = len(COLUMN_NAMES)

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                fields = [x for x in line.strip().split(' ') if x]

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

def find_and_combine_metadata(root_dir):
    experiment_dfs = {
        '2DSA': [],
        'GA': [],
        'PCSA': []
    }
    metrics = {
        'total_files': 0,
        'files_with_errors': 0,
        'experiment_counts': {'2DSA': 0, 'GA': 0, 'PCSA': 0},
        'filtered_counts': {'2DSA': 0, 'GA': 0, 'PCSA': 0}
    }

    all_errors = []
    error_files = []

    pattern = os.path.join(root_dir, '**/summary_metadata.csv')
    all_files = glob.glob(pattern, recursive=True)
    metrics['total_files'] = len(all_files)

    print(f"Found {len(all_files)} summary_metadata.csv files")

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
                    df_filtered.loc[:, 'source_file'] = file_path
                    df_filtered.loc[:, 'directory'] = os.path.dirname(file_path)
                    experiment_dfs[exp_type].append(df_filtered)
                    metrics['experiment_counts'][exp_type] += len(df_filtered)

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
            metrics['filtered_counts'][exp_type] = initial_count - len(filtered_df)

            combined_dfs[exp_type] = filtered_df

    return combined_dfs, metrics

def main():
    if not os.path.exists(ROOT_DIRECTORY):
        print(f"Error: Directory does not exist: {ROOT_DIRECTORY}")
        return

    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    print(f"Starting search in: {ROOT_DIRECTORY}")

    combined_data, metrics = find_and_combine_metadata(ROOT_DIRECTORY)

    # Save metrics to CSV
    metrics_df = pd.DataFrame([{
        'total_files': metrics['total_files'],
        'files_with_errors': metrics['files_with_errors'],
        '2DSA_total': metrics['experiment_counts']['2DSA'],
        '2DSA_filtered': metrics['filtered_counts']['2DSA'],
        'GA_total': metrics['experiment_counts']['GA'],
        'GA_filtered': metrics['filtered_counts']['GA'],
        'PCSA_total': metrics['experiment_counts']['PCSA'],
        'PCSA_filtered': metrics['filtered_counts']['PCSA']
    }])

    metrics_df.to_csv(os.path.join(OUTPUT_DIRECTORY, 'processing_metrics.csv'), index=False)

    if combined_data:
        for exp_type, df in combined_data.items():
            output_file = os.path.join(OUTPUT_DIRECTORY, f'{exp_type.lower()}-dataset.csv')
            df.to_csv(output_file, index=False)
            print(f"\n{exp_type} dataset saved to: {output_file}")
            print(f"Total size: {len(df)} rows × {len(df.columns)} columns")

if __name__ == "__main__":
    main()
