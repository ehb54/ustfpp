import pandas as pd
import os

def analyze_and_filter_columns(input_file, output_file):
    df = pd.read_csv(input_file)
    total_cols = len(df.columns)

    nunique = df.nunique()
    uniform_cols = nunique[nunique == 1].index
    cols_to_keep = nunique[nunique > 1].index

    filtered_df = df[cols_to_keep]
    filtered_df.to_csv(output_file, index=False)

    metrics = {
        'total_columns': total_cols,
        'columns_removed': len(uniform_cols),
        'columns_remaining': len(cols_to_keep),
        'uniform_columns': {col: df[col].iloc[0] for col in uniform_cols}
    }

    return filtered_df, metrics

def main():
    input_dir = "./results"
    output_dir = "./results/filtered"
    os.makedirs(output_dir, exist_ok=True)

    method_files = ['2dsa-dataset.csv', 'ga-dataset.csv', 'pcsa-dataset.csv']
    all_metrics = []

    for file in method_files:
        input_path = os.path.join(input_dir, file)
        if not os.path.exists(input_path):
            print(f"File not found: {input_path}")
            continue

        method_name = file.split('-')[0].upper()
        output_path = os.path.join(output_dir, f'{method_name.lower()}-filtered.csv')

        _, metrics = analyze_and_filter_columns(input_path, output_path)

        method_metrics = {
            'method': method_name,
            'total_columns': metrics['total_columns'],
            'columns_removed': metrics['columns_removed'],
            'columns_remaining': metrics['columns_remaining']
        }

        for col, value in metrics['uniform_columns'].items():
            method_metrics[f'uniform_{col}'] = value

        all_metrics.append(method_metrics)
        print(f"\nProcessed {method_name}:")
        print(f"Total columns: {metrics['total_columns']}")
        print(f"Columns removed: {metrics['columns_removed']}")
        print(f"Columns remaining: {metrics['columns_remaining']}")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(output_dir, 'uniform_columns_metrics.csv'), index=False)

if __name__ == "__main__":
    main()
