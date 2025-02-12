#!/usr/bin/env python3
import glob
import argparse
import sys
import os
import logging
import pandas as pd
import json
from typing import Dict, Any, Optional, Tuple, List

class DataPreprocessor:
    def __init__(self, config: Dict[str, Any] = None, log_level: str = 'INFO'):
        """Initialize the DataPreprocessor with optional configuration and logging."""
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))

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

    def _configure_column_selection(self, df: pd.DataFrame, model_name: str = None) -> pd.DataFrame:
        """Apply column selection and transformation based on configuration."""
        if model_name:
            # Get model-specific column configuration
            model_configs = self.config.get('column_configuration', {}).get('model_columns', {})
            if model_name in model_configs:
                model_config = model_configs[model_name]
                columns_to_keep = model_config.get('keep_columns', [])
                if columns_to_keep:
                    self.logger.info(f"Keeping columns for model {model_name}: {columns_to_keep}")
                    valid_columns = [col for col in columns_to_keep if col in df.columns]
                    df = df[valid_columns]
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

    def _handle_negative_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle negative values based on configuration."""
        negative_value_config = self.config.get('negative_value_handling', {})
        negative_filter_mode = negative_value_config.get('mode', 'disabled')

        # Identify numeric columns
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

        if negative_filter_mode == 'all_columns':
            self.logger.info("Filtering out rows with negative values in ALL numeric columns")
            keep_mask = ~(df[numeric_columns] < 0).any(axis=1)
            df = df[keep_mask]
        elif negative_filter_mode == 'specific_columns':
            negative_filter_columns = negative_value_config.get('columns', [])
            valid_columns = [col for col in negative_filter_columns if col in numeric_columns]
            if valid_columns:
                self.logger.info(f"Filtering out rows with negative values in columns: {valid_columns}")
                keep_mask = ~df[valid_columns].lt(0).any(axis=1)
                df = df[keep_mask]
        elif negative_filter_mode == 'replace':
            replace_value = negative_value_config.get('replace_value', 0)
            replace_columns = negative_value_config.get('columns', [])
            if not replace_columns:
                replace_columns = numeric_columns
            valid_columns = [col for col in replace_columns if col in numeric_columns]
            if valid_columns:
                self.logger.info(f"Replacing negative values in columns: {valid_columns}")
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

    def process_for_models(self, df: pd.DataFrame, output_dir: str) -> List[Dict[str, Any]]:
        """Process the DataFrame for each model configuration."""
        model_metrics = []
        model_configs = self.config.get('column_configuration', {}).get('model_columns', {})
        output_configs = self.config.get('output_files', {}).get('models', {})

        # Store initial metrics
        total_rows_initial = len(df)
        total_cols_initial = len(df.columns)

        for model_name, model_config in model_configs.items():
            self.logger.info(f"Processing data for model: {model_name}")

            # Create a copy of the DataFrame for this model
            model_df = df.copy()

            # Apply model-specific column selection
            model_df = self._configure_column_selection(model_df, model_name)

            # Apply uniform column handling
            uniform_column_config = self.config.get('uniform_columns', {})
            if uniform_column_config.get('remove', True):
                nunique = model_df.nunique()
                threshold = uniform_column_config.get('threshold', 1)
                uniform_cols = nunique[nunique <= threshold].index
                model_df = model_df[nunique[nunique > threshold].index]
                self.logger.info(f"Removed {len(uniform_cols)} uniform columns for {model_name}")

            # Apply common preprocessing steps
            model_df = self._handle_negative_values(model_df)
            model_df = self._apply_value_filters(model_df)

            # Get output configuration for this model
            output_config = output_configs.get(model_name, {})
            output_filename = output_config.get('filename', f'2dsadatasetfiltered{model_name}.csv')
            metrics_filename = output_config.get('metrics_file', f'{model_name}_preprocessing_metrics.csv')

            # Save the processed DataFrame
            output_path = os.path.join(output_dir, output_filename)
            model_df.to_csv(output_path, index=False)
            self.logger.info(f"Saved processed data for {model_name} to {output_path}")

            # Compile metrics
            metrics = {
                'model': model_name,
                'input_rows': total_rows_initial,
                'output_rows': len(model_df),
                'input_columns': total_cols_initial,
                'output_columns': len(model_df.columns),
                'columns_kept': list(model_df.columns),
                'output_file': output_filename
            }
            model_metrics.append(metrics)

            # Save model-specific metrics
            metrics_path = os.path.join(output_dir, metrics_filename)
            pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

        return model_metrics

    def process_file(self, input_file: str, output_dir: str = './results/filtered') -> Tuple[Optional[pd.DataFrame], List[Dict[str, Any]]]:
        """Process a single CSV file for multiple models."""
        self.logger.info(f"Processing file: {input_file}")

        try:
            # Read the CSV file
            df = pd.read_csv(input_file)

            # Process for each model configuration
            metrics = self.process_for_models(df, output_dir)

            return df, metrics
        except Exception as e:
            self.logger.error(f"Error processing {input_file}: {e}")
            return None, [{
                'input_file': os.path.basename(input_file),
                'error': str(e)
            }]

    def batch_process(self, input_files: List[str], output_dir: str = './results/filtered') -> List[Dict[str, Any]]:
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
            help='Output directory for filtered files (overrides config if provided)'
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

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger('DataPreprocessor')

        try:
            with open(args.config, 'r') as config_file:
                config = json.load(config_file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading configuration file: {e}")
            sys.exit(1)

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

        output_dir = args.output or config.get('output_files', {}).get('directory', './results/filtered')

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
