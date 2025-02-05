#!/usr/bin/env python3
import glob  # <-- Add this here
import argparse
import sys
import os
import logging
import pandas as pd
import json
from typing import Dict, Any, Optional, Tuple, List

class DataPreprocessor:
    def __init__(self, config: Dict[str, Any] = None, log_level: str = 'INFO'):
        """
        Initialize the DataPreprocessor with optional configuration and logging.

        Args:
            config (Dict[str, Any], optional): Configuration dictionary for preprocessing.
            log_level (str, optional): Logging level. Defaults to 'INFO'.
        """
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
        """
        Create a DataPreprocessor instance from a JSON configuration file.

        Args:
            config_path (str): Path to the JSON configuration file.
            log_level (str, optional): Logging level. Defaults to 'INFO'.

        Returns:
            DataPreprocessor: Initialized preprocessor with configuration from JSON file.
        """
        try:
            with open(config_path, 'r') as config_file:
                config = json.load(config_file)
            return cls(config, log_level)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Error loading configuration: {e}")

    def _configure_column_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply column selection and transformation based on configuration.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Processed DataFrame with selected and transformed columns.
        """
        column_config = self.config.get('column_configuration', {})

        # Columns to keep (whitelist approach)
        columns_to_keep = column_config.get('keep_columns', [])
        if columns_to_keep:
            self.logger.info(f"Keeping only specified columns: {columns_to_keep}")
            valid_columns = [col for col in columns_to_keep if col in df.columns]
            df = df[valid_columns]

        # Columns to drop (blacklist approach)
        columns_to_drop = column_config.get('drop_columns', [])
        if columns_to_drop:
            self.logger.info(f"Dropping specified columns: {columns_to_drop}")
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

        # Column transformations
        column_transforms = column_config.get('transformations', {})

        # Rename columns
        rename_map = column_transforms.get('rename_columns', {})
        if rename_map:
            self.logger.info(f"Renaming columns: {rename_map}")
            df = df.rename(columns=rename_map)

        return df

    def _handle_negative_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle negative values based on configuration.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Processed DataFrame with negative values handled.
        """
        negative_value_config = self.config.get('negative_value_handling', {})
        negative_filter_mode = negative_value_config.get('mode', 'disabled')

        if negative_filter_mode == 'all_columns':
            # Remove rows with negative values in ANY column
            self.logger.info("Filtering out rows with negative values in ALL columns")
            keep_mask = ~(df < 0).any(axis=1)
            df = df[keep_mask]
        elif negative_filter_mode == 'specific_columns':
            # Remove rows with negative values in SPECIFIED columns
            negative_filter_columns = negative_value_config.get('columns', [])
            if negative_filter_columns:
                self.logger.info(f"Filtering out rows with negative values in columns: {negative_filter_columns}")
                keep_mask = ~df[negative_filter_columns].lt(0).any(axis=1)
                df = df[keep_mask]
        elif negative_filter_mode == 'replace':
            # Replace negative values with a specified value or method
            replace_value = negative_value_config.get('replace_value', 0)
            replace_columns = negative_value_config.get('columns', [])

            if not replace_columns:
                # If no columns specified, replace in all numeric columns
                replace_columns = df.select_dtypes(include=['int64', 'float64']).columns

            self.logger.info(f"Replacing negative values in columns: {replace_columns}")
            for col in replace_columns:
                df[col] = df[col].clip(lower=replace_value)

        return df

    def _apply_value_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply additional value-based filtering.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        additional_filters = self.config.get('additional_filters', {})

        # Maximum value filters
        max_filters = additional_filters.get('max_value_filters', {})
        for col, max_val in max_filters.items():
            if col in df.columns:
                self.logger.info(f"Applying max filter for {col}: {max_val}")
                df = df[df[col] <= max_val]

        # Minimum value filters
        min_filters = additional_filters.get('min_value_filters', {})
        for col, min_val in min_filters.items():
            if col in df.columns:
                self.logger.info(f"Applying min filter for {col}: {min_val}")
                df = df[df[col] >= min_val]

        return df

    def process_file(self,
                     input_file: str,
                     output_dir: str = './results/filtered') -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Process a single CSV file based on the configured preprocessing steps.

        Args:
            input_file (str): Path to the input CSV file.
            output_dir (str, optional): Directory to save filtered files. Defaults to './results/filtered'.

        Returns:
            Tuple containing:
            - Filtered DataFrame (or None if processing fails)
            - Metrics dictionary
        """
        self.logger.info(f"Processing file: {input_file}")

        try:
            # Read the CSV file
            df = pd.read_csv(input_file)
        except Exception as e:
            self.logger.error(f"Error reading {input_file}: {e}")
            return None, {
                'input_file': os.path.basename(input_file),
                'error': str(e)
            }

        # Store initial metrics
        original_df = df.copy()
        total_rows_initial = len(df)
        total_cols_initial = len(df.columns)

        try:
            # Handle uniform columns
            uniform_column_config = self.config.get('uniform_columns', {})
            remove_uniform_columns = uniform_column_config.get('remove', True)
            uniform_threshold = uniform_column_config.get('threshold', 1)

            # Apply preprocessing steps
            df = self._configure_column_selection(df)

            if remove_uniform_columns:
                # Identify uniform columns
                nunique = df.nunique()
                uniform_cols = nunique[nunique <= uniform_threshold].index
                cols_to_keep = nunique[nunique > uniform_threshold].index

                self.logger.info(f"Removing uniform columns: {list(uniform_cols)}")

                # Filter DataFrame to keep only non-uniform columns
                df = df[cols_to_keep]
            else:
                self.logger.info("Skipping uniform column removal")
                uniform_cols = pd.Index([])

            df = self._handle_negative_values(df)
            df = self._apply_value_filters(df)

            # Prepare output
            os.makedirs(output_dir, exist_ok=True)
            input_filename = os.path.basename(input_file)
            method_name = os.path.splitext(input_filename)[0]
            output_path = os.path.join(output_dir, f'{method_name}-filtered.csv')

            # Save filtered DataFrame
            df.to_csv(output_path, index=False)
            self.logger.info(f"Filtered data saved to {output_path}")

            # Compile comprehensive metrics
            metrics = {
                'input_file': input_filename,
                'total_rows_initial': total_rows_initial,
                'total_rows_after_filtering': len(df),
                'rows_removed': total_rows_initial - len(df),
                'total_columns_initial': total_cols_initial,
                'columns_removed': len(uniform_cols),
                'columns_remaining': len(df.columns),
                'uniform_columns': {col: original_df[col].iloc[0] for col in uniform_cols},
                'column_configuration': {
                    'keep_columns': self.config.get('column_configuration', {}).get('keep_columns', []),
                    'drop_columns': self.config.get('column_configuration', {}).get('drop_columns', []),
                    'renamed_columns': self.config.get('column_configuration', {})
                    .get('transformations', {})
                    .get('rename_columns', {})
                },
                'negative_value_handling': self.config.get('negative_value_handling', {}),
                'additional_filters': self.config.get('additional_filters', {}),
                'uniform_columns_config': uniform_column_config
            }

            return df, metrics

        except Exception as e:
            self.logger.error(f"Error processing {input_file}: {e}")
            return None, {
                'input_file': os.path.basename(input_file),
                'error': str(e)
            }

    def batch_process(self,
                      input_files: List[str],
                      output_dir: str = './results/filtered') -> List[Dict[str, Any]]:
        """
        Process multiple CSV files.

        Args:
            input_files (List[str]): List of input file paths.
            output_dir (str, optional): Directory to save filtered files. Defaults to './results/filtered'.

        Returns:
            List of metrics for each processed file.
        """
        all_metrics = []

        for input_file in input_files:
            if not os.path.exists(input_file):
                self.logger.warning(f"File not found: {input_file}")
                continue

            _, file_metrics = self.process_file(input_file, output_dir)
            all_metrics.append(file_metrics)

        # Save aggregated metrics
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            metrics_path = os.path.join(output_dir, 'preprocessing_metrics.csv')
            metrics_df.to_csv(metrics_path, index=False)
            self.logger.info(f"Aggregate metrics saved to {metrics_path}")

        return all_metrics

    def _handle_negative_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle negative values based on configuration.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Processed DataFrame with negative values handled.
        """
        negative_value_config = self.config.get('negative_value_handling', {})
        negative_filter_mode = negative_value_config.get('mode', 'disabled')

        # Identify numeric columns
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

        if negative_filter_mode == 'all_columns':
            # Remove rows with negative values in ANY numeric column
            self.logger.info("Filtering out rows with negative values in ALL numeric columns")
            keep_mask = ~(df[numeric_columns] < 0).any(axis=1)
            df = df[keep_mask]
        elif negative_filter_mode == 'specific_columns':
            # Remove rows with negative values in SPECIFIED numeric columns
            negative_filter_columns = negative_value_config.get('columns', [])
            # Intersect specified columns with actual numeric columns
            valid_columns = [col for col in negative_filter_columns if col in numeric_columns]

            if valid_columns:
                self.logger.info(f"Filtering out rows with negative values in columns: {valid_columns}")
                keep_mask = ~df[valid_columns].lt(0).any(axis=1)
                df = df[keep_mask]
        elif negative_filter_mode == 'replace':
            # Replace negative values with a specified value or method
            replace_value = negative_value_config.get('replace_value', 0)
            replace_columns = negative_value_config.get('columns', [])

            # If no columns specified, use all numeric columns
            if not replace_columns:
                replace_columns = numeric_columns

            # Find valid columns to replace
            valid_columns = [col for col in replace_columns if col in numeric_columns]

            if valid_columns:
                self.logger.info(f"Replacing negative values in columns: {valid_columns}")
                for col in valid_columns:
                    df[col] = df[col].clip(lower=replace_value)

        return df

    @classmethod
    def cli_main(cls):
        """
        Command-line interface for the DataPreprocessor.
        """
        # Set up argument parser
        parser = argparse.ArgumentParser(
            description='Advanced CSV Data Preprocessing and Filtering Tool',
            epilog='Example: python preprocess_filter.py -c config.json'
        )

        # Configuration file argument (now the primary argument)
        parser.add_argument(
            '-c', '--config',
            type=str,
            required=True,
            help='Path to JSON configuration file'
        )

        # Input files argument (now optional)
        parser.add_argument(
            '-i', '--input',
            nargs='+',
            default=[],
            help='Input CSV file(s) to process (optional, can be specified in config)'
        )

        # Output directory argument
        parser.add_argument(
            '-o', '--output',
            default=None,
            help='Output directory for filtered files (overrides config if provided)'
        )

        # Logging level argument
        parser.add_argument(
            '--log-level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            default='INFO',
            help='Set the logging level (default: INFO)'
        )

        # Dry run option
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Validate configuration and input files without processing'
        )

        # Parse arguments
        args = parser.parse_args()

        # Set up logging
        # Remove all existing handlers
        logging.getLogger().handlers.clear()

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger('DataPreprocessor')

        # Load configuration
        try:
            with open(args.config, 'r') as config_file:
                config = json.load(config_file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading configuration file: {e}")
            sys.exit(1)

        # Determine input files
        valid_input_files = []

        # First, check command-line input files
        if args.input:
            valid_input_files = [f for f in args.input if os.path.exists(f) and f.lower().endswith('.csv')]

        # If no input files from command line, check config
        if not valid_input_files:
            # Check for input files in config
            input_config = config.get('input_files', {})
            source_dir = input_config.get('source_directory', '.')
            file_patterns = input_config.get('file_patterns', [])

            # Collect input files based on patterns
            for pattern in file_patterns:
                # Support both absolute and relative paths
                search_paths = [
                    os.path.join(source_dir, pattern),
                    pattern
                ]

                for search_path in search_paths:
                    matching_files = glob.glob(search_path)
                    valid_input_files.extend([f for f in matching_files if f.lower().endswith('.csv')])

        # Exit if no valid input files
        if not valid_input_files:
            logger.error("No valid input files found in config or command line.")
            sys.exit(1)

        # Determine output directory
        output_dir = args.output or config.get('output_files', {}).get('directory', './results/filtered')

        # Dry run mode
        if args.dry_run:
            logger.info("Dry run mode: Validating configuration and input files")
            logger.info(f"Valid input files: {valid_input_files}")
            logger.info(f"Output directory: {output_dir}")
            return

        # Create preprocessor
        try:
            preprocessor = cls(config, log_level=args.log_level)
        except Exception as e:
            logger.error(f"Error initializing preprocessor: {e}")
            sys.exit(1)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Process files
        try:
            metrics = preprocessor.batch_process(valid_input_files, output_dir)

            # Print summary
            if metrics:
                logger.info("\nProcessing Summary:")
                unique_metrics = {}
                for metric in metrics:
                    # Use input_file as a key to avoid duplicates
                    if metric['input_file'] not in unique_metrics:
                        logger.info(f"File: {metric['input_file']}")
                        logger.info(f"  Initial Rows: {metric['total_rows_initial']}")
                        logger.info(f"  Filtered Rows: {metric['total_rows_after_filtering']}")
                        logger.info(f"  Columns Remaining: {metric['columns_remaining']}")
                        logger.info("---")
                        unique_metrics[metric['input_file']] = metric
            else:
                logger.warning("No files were processed successfully.")

        except Exception as e:
            logger.error(f"Error during batch processing: {e}")
            sys.exit(1)

def main():
    DataPreprocessor.cli_main()

if __name__ == "__main__":
    main()
