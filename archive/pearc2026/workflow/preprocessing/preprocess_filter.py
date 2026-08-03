#!/usr/bin/env python3
import glob
import argparse
import sys
import os
import logging
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Optional, Tuple, List

class DataPreprocessor:
    def __init__(self, config: Dict[str, Any] = None, log_level: str = 'INFO'):
        """Initialize the DataPreprocessor with optional configuration and logging."""
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Prevent duplicate handlers
        if not self.logger.handlers:
            # Create console handler
            ch = logging.StreamHandler()
            ch.setLevel(getattr(logging, log_level.upper()))
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        # Load configuration
        self.config = config or {}

    @classmethod
    def from_json_config(cls, config_path: str, log_level: str = 'INFO'):
        """Create a DataPreprocessor instance from a JSON configuration file."""
        try:
            with open(config_path, 'r') as config_file:
                config = json.load(config_file)
            return cls(config, log_level)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Error loading configuration: {e}")

    def _determine_model_from_filename(self, filename: str) -> str:
        """Determine the model type based on the input filename."""
        basename = os.path.basename(filename).lower()
        if '2dsa' in basename:
            return '2dsa'
        elif 'ga' in basename:
            return 'ga'
        elif 'pcsa' in basename:
            return 'pcsa'
        else:
            # Default fallback or raise an error
            raise ValueError(f"Cannot determine model type from filename: {filename}")

    def _configure_column_selection(self, df: pd.DataFrame, model_name: str = None) -> pd.DataFrame:
        """Apply column selection and transformation based on configuration."""
        if model_name:
            model_configs = self.config.get('column_configuration', {}).get('model_columns', {})
            model_config = model_configs.get(model_name, {})

            # keep_columns (optional)
            columns_to_keep = model_config.get('keep_columns', [])
            if columns_to_keep:
                self.logger.info(f"Keeping columns for model {model_name}: {columns_to_keep}")
                valid_columns = [col for col in columns_to_keep if col in df.columns]
                df = df[valid_columns]

            # drop_columns (optional)
            columns_to_drop = model_config.get('drop_columns', [])
            if columns_to_drop:
                self.logger.info(f"Dropping columns for model {model_name}: {columns_to_drop}")
                df = df.drop(columns=[c for c in columns_to_drop if c in df.columns], errors='ignore')

            # transformations (optional)
            transforms = model_config.get('transformations', {})
            rename_map = transforms.get('rename_columns', {}) if transforms else {}
            if rename_map:
                df = df.rename(columns=rename_map)

            return df

        # Original column configuration logic for non-model-specific processing
        column_config = self.config.get('column_configuration', {})
        columns_to_keep = column_config.get('keep_columns', [])
        if columns_to_keep:
            valid_columns = [col for col in columns_to_keep if col in df.columns]
            df = df[valid_columns]

        columns_to_drop = column_config.get('drop_columns', [])
        if columns_to_drop:
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

        column_transforms = column_config.get('transformations', {})
        rename_map = column_transforms.get('rename_columns', {})
        if rename_map:
            df = df.rename(columns=rename_map)

        return df

    def _handle_negative_values(self, df: pd.DataFrame, model_name: str = None) -> pd.DataFrame:
        """Handle negative values based on configuration."""
        # Check for model-specific configuration first
        if model_name:
            model_configs = self.config.get('column_configuration', {}).get('model_columns', {})
            if model_name in model_configs:
                model_negative_config = model_configs[model_name].get('negative_value_handling', {})
                if model_negative_config:
                    negative_value_config = model_negative_config
                else:
                    negative_value_config = self.config.get('negative_value_handling', {})
            else:
                negative_value_config = self.config.get('negative_value_handling', {})
        else:
            negative_value_config = self.config.get('negative_value_handling', {})

        negative_filter_mode = negative_value_config.get('mode', 'disabled')

        # Identify numeric columns
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

        if negative_filter_mode == 'all_columns':
            self.logger.info(
                f"Filtering out rows with negative values in ALL numeric columns for {model_name or 'default'}")
            keep_mask = ~(df[numeric_columns] < 0).any(axis=1)
            df = df[keep_mask]
        elif negative_filter_mode == 'specific_columns':
            negative_filter_columns = negative_value_config.get('columns', [])
            valid_columns = [col for col in negative_filter_columns if col in numeric_columns]
            if valid_columns:
                self.logger.info(
                    f"Filtering out rows with negative values in columns: {valid_columns} for {model_name or 'default'}")
                keep_mask = ~df[valid_columns].lt(0).any(axis=1)
                df = df[keep_mask]
        elif negative_filter_mode == 'replace':
            replace_value = negative_value_config.get('replace_value', 0)
            replace_columns = negative_value_config.get('columns', [])
            if not replace_columns:
                replace_columns = numeric_columns
            valid_columns = [col for col in replace_columns if col in numeric_columns]
            if valid_columns:
                self.logger.info(f"Replacing negative values in columns: {valid_columns} for {model_name or 'default'}")
                for col in valid_columns:
                    df[col] = df[col].clip(lower=replace_value)

        return df

    def _apply_value_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply additional value-based filtering."""
        additional_filters = self.config.get('additional_filters', {})

        max_filters = additional_filters.get('max_value_filters', {})
        for col, max_val in max_filters.items():
            if col in df.columns:
                self.logger.info(f"Applying max filter for {col}: {max_val}")
                df = df[df[col] <= max_val]

        min_filters = additional_filters.get('min_value_filters', {})
        for col, min_val in min_filters.items():
            if col in df.columns:
                self.logger.info(f"Applying min filter for {col}: {min_val}")
                df = df[df[col] >= min_val]

        return df

    def process_for_single_model(self, df: pd.DataFrame, model_name: str, output_dir: str, filename: str) -> Dict[
        str, Any]:
        """Process the DataFrame for a single model."""
        self.logger.info(f"Processing data for model: {model_name}")

        # Store initial metrics
        total_rows_initial = len(df)
        total_cols_initial = len(df.columns)

        # Apply model-specific column selection
        df = self._configure_column_selection(df, model_name)

        # Drop rows with missing target values
        target_columns = ['wallTime', 'max_rss']
        valid_targets = [col for col in target_columns if col in df.columns]
        if valid_targets:
            initial_rows = len(df)
            df = df.dropna(subset=valid_targets)
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0:
                self.logger.info(f"{filename}: Dropped {dropped_rows} rows with missing target values")

        # Apply uniform column handling
        uniform_column_config = self.config.get('uniform_columns', {})
        if uniform_column_config.get('remove', True):
            nunique = df.nunique()
            threshold = uniform_column_config.get('threshold', 1)
            uniform_cols = nunique[nunique <= threshold].index
            df = df[nunique[nunique > threshold].index]
            self.logger.info(f"Removed {len(uniform_cols)} uniform columns for {model_name}")

        # Apply common preprocessing steps
        df = self._handle_negative_values(df, model_name)
        df = self._apply_value_filters(df)

        # Get output configuration
        output_configs = self.config.get('output_files', {}).get('models', {})
        output_config = output_configs.get(model_name, {})
        output_filename = output_config.get('filename', f'{model_name}_cleaned.csv')
        metrics_filename = output_config.get('metrics_file', f'{model_name}_preprocessing_metrics.csv')

        # Save the processed DataFrame
        output_path = os.path.join(output_dir, output_filename)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved processed data for {model_name} to {output_path}")

        # Compile metrics
        metrics = {
            'model': model_name,
            'input_rows': total_rows_initial,
            'output_rows': len(df),
            'input_columns': total_cols_initial,
            'output_columns': len(df.columns),
            'columns_kept': list(df.columns),
            'output_file': output_filename
        }

        # Save model-specific metrics
        metrics_path = os.path.join(output_dir, metrics_filename)
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

        return metrics

    def process_file(self, input_file: str, output_dir: str = './results/cleaned') -> Tuple[
        Optional[pd.DataFrame], List[Dict[str, Any]]]:
        """Process a single CSV file for its corresponding model."""
        self.logger.info(f"Processing file: {input_file}")

        try:
            # Read the CSV file
            df = pd.read_csv(input_file)

            # Replace "<unset>" with NaN
            df = df.replace('<unset>', np.nan)

            # Determine model type from filename
            model_name = self._determine_model_from_filename(input_file)

            # Process for the single determined model
            metrics = self.process_for_single_model(df, model_name, output_dir, os.path.basename(input_file))

            return df, [metrics]
        except Exception as e:
            self.logger.error(f"Error processing {input_file}: {e}")
            return None, [{
                'input_file': os.path.basename(input_file),
                'error': str(e)
            }]

    def batch_process(self, input_files: List[str], output_dir: str = './results/cleaned') -> List[Dict[str, Any]]:
        """Process multiple CSV files."""
        all_metrics = []

        for input_file in input_files:
            if not os.path.exists(input_file):
                self.logger.warning(f"File not found: {input_file}")
                continue

            _, file_metrics = self.process_file(input_file, output_dir)
            all_metrics.extend(file_metrics)

        # Save aggregated metrics
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            metrics_path = os.path.join(output_dir, 'preprocessing_metrics.csv')
            metrics_df.to_csv(metrics_path, index=False)
            self.logger.info(f"Aggregate metrics saved to {metrics_path}")

        return all_metrics

    @classmethod
    def cli_main(cls):
        """Command-line interface for the DataPreprocessor."""
        parser = argparse.ArgumentParser(
            description='Advanced CSV Data Preprocessing and Filtering Tool',
            epilog='Example: python preprocess_filter.py -c config.json'
        )

        parser.add_argument(
            '-c', '--config',
            type=str,
            required=True,
            help='Path to JSON configuration file'
        )

        parser.add_argument(
            '-i', '--input',
            nargs='+',
            default=[],
            help='Input CSV file(s) to process (optional, can be specified in config)'
        )

        parser.add_argument(
            '-o', '--output',
            default=None,
            help='Output directory for cleaned files (overrides config if provided)'
        )

        parser.add_argument(
            '--log-level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            default='INFO',
            help='Set the logging level (default: INFO)'
        )

        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Validate configuration and input files without processing'
        )

        args = parser.parse_args()

        logger = logging.getLogger('DataPreprocessor')

        try:
            with open(args.config, 'r') as config_file:
                config = json.load(config_file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading configuration file: {e}")
            sys.exit(1)

        # Resolve paths relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Resolve input directory path
        input_config = config.get('input_files', {})
        if 'source_directory' in input_config:
            source_dir = input_config['source_directory']
            if not os.path.isabs(source_dir):
                input_config['source_directory'] = os.path.join(script_dir, source_dir)

        # Resolve output directory path
        output_config = config.get('output_files', {})
        if 'directory' in output_config:
            output_dir = output_config['directory']
            if not os.path.isabs(output_dir):
                output_config['directory'] = os.path.join(script_dir, output_dir)

        # Determine input files
        valid_input_files = []

        if args.input:
            valid_input_files = [f for f in args.input if os.path.exists(f) and f.lower().endswith('.csv')]

        if not valid_input_files:
            input_config = config.get('input_files', {})
            source_dir = input_config.get('source_directory', '.')
            file_patterns = input_config.get('file_patterns', [])

            for pattern in file_patterns:
                search_paths = [
                    os.path.join(source_dir, pattern),
                    pattern
                ]

                for search_path in search_paths:
                    matching_files = glob.glob(search_path)
                    valid_input_files.extend([f for f in matching_files if f.lower().endswith('.csv')])

        if not valid_input_files:
            logger.error("No valid input files found in config or command line.")
            sys.exit(1)

        output_dir = args.output or config.get('output_files', {}).get('directory', './results/cleaned')

        if args.dry_run:
            logger.info("Dry run mode: Validating configuration and input files")
            logger.info(f"Valid input files: {valid_input_files}")
            logger.info(f"Output directory: {output_dir}")
            return

        try:
            preprocessor = cls(config, log_level=args.log_level)
        except Exception as e:
            logger.error(f"Error initializing preprocessor: {e}")
            sys.exit(1)

        os.makedirs(output_dir, exist_ok=True)

        try:
            metrics = preprocessor.batch_process(valid_input_files, output_dir)

            if metrics:
                logger.info("\nProcessing Summary:")
                for metric in metrics:
                    logger.info(f"Model: {metric['model']}")
                    logger.info(f"  Input Rows: {metric['input_rows']}")
                    logger.info(f"  Output Rows: {metric['output_rows']}")
                    logger.info(f"  Output Columns: {metric['output_columns']}")
                    logger.info(f"  Output File: {metric['output_file']}")
                    logger.info("---")
            else:
                logger.warning("No files were processed successfully.")

        except Exception as e:
            logger.error(f"Error during batch processing: {e}")
            sys.exit(1)

def main():
    DataPreprocessor.cli_main()

if __name__ == "__main__":
    main()
