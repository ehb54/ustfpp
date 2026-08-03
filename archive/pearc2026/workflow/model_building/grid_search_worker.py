import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # optional: reduces TF info spam
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # ok: prevents TF grabbing all VRAM

import tensorflow as tf

tf.config.optimizer.set_jit(False)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pandas as pd
import numpy as np
import json
import pickle
import sys
import re
from datetime import datetime


def load_column_file(columns_file):
    """
    Load column names from newline-delimited text file.
    Returns: list of column names, preserving order
    """
    if not os.path.exists(columns_file):
        print(f"ERROR: Columns file not found: {columns_file}", file=sys.stderr)
        sys.exit(1)

    columns = []
    with open(columns_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                columns.append(line)

    if len(columns) == 0:
        print(f"ERROR: Columns file contains no valid column names: {columns_file}", file=sys.stderr)
        sys.exit(1)

    return columns


class EnhancedPredictionFramework:
    def __init__(self, output_dir='results/prediction', checkpoint_file=None):
        self.output_dir = output_dir
        self.checkpoint_file = checkpoint_file or os.path.join(output_dir, 'checkpoint.json')
        self.models_dir = os.path.join(output_dir, 'models')
        self.error_analysis_dir = os.path.join(output_dir, 'error_analysis')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.error_analysis_dir, exist_ok=True)

        self.scalers = ['standard', 'minmax', 'robust']
        self.optimizers = ['adam', 'rmsprop', 'sgd']
        self.batch_sizes = [16, 32, 64]
        self.activations = ['relu', 'elu']
        self.dropout_rates = [0.2, 0.3]
        self.epochs = 50
        self.patience = 5
        self.training_fraction = 0.7
        self.validation_fraction = 0.15
        self.random_selection = False

        # Target variable configuration
        self.target_variable = 'CPUTime'  # Default for backwards compatibility
        self.target_variables = None  # For multi-output models

        # Temporal split configuration
        self.temporal_split = False
        self.time_column = 'submitTime'

        # NEW: Column selection configuration
        self.method = None
        self.feature_space = None
        self.columns_file = None
        self.data_file = None
        self.feature_columns = None
        self.dry_run = False

    def load_config(self, config_file, config_json_data):
        """Load configuration from JSON, including new required fields"""

        # Required fields for new config-driven approach
        required_fields = ['method', 'feature_space', 'columns_file', 'data_file']
        missing_fields = [f for f in required_fields if f not in config_json_data]

        if missing_fields:
            print(f"ERROR: Config missing required fields: {', '.join(missing_fields)}", file=sys.stderr)
            sys.exit(1)

        # Validate method
        self.method = config_json_data['method']
        valid_methods = ['2dsa', 'ga', 'pcsa']
        if self.method not in valid_methods:
            print(f"ERROR: Invalid method '{self.method}', must be one of: {', '.join(valid_methods)}", file=sys.stderr)
            sys.exit(1)

        # Validate feature_space
        self.feature_space = config_json_data['feature_space']
        valid_spaces = ['raw']
        if self.feature_space not in valid_spaces:
            print(f"ERROR: Invalid feature_space '{self.feature_space}', must be one of: {', '.join(valid_spaces)}",
                  file=sys.stderr)
            sys.exit(1)

        # Store paths
        self.columns_file = config_json_data['columns_file']
        self.data_file = config_json_data['data_file']

        # Check if dry run
        self.dry_run = config_json_data.get('dry_run', False)

        # Load feature columns from file
        self.feature_columns = load_column_file(self.columns_file)

        # Log configuration at startup
        print("=" * 80)
        print("GRID SEARCH WORKER CONFIGURATION")
        print("=" * 80)
        print(f"Method: {self.method}")
        print(f"Feature space: {self.feature_space}")
        print(f"Data file: {self.data_file}")
        print(f"Columns file: {self.columns_file}")
        print(f"Feature columns loaded: {len(self.feature_columns)}")

        # Log GPU configuration
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
        print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")

        # Handle output directory from config if not overridden by CLI
        if "output_dir" in config_json_data and self.output_dir == 'results/prediction':
            self.output_dir = config_json_data["output_dir"]
            self.checkpoint_file = os.path.join(self.output_dir, 'checkpoint.json')
            self.models_dir = os.path.join(self.output_dir, 'models')
            self.error_analysis_dir = os.path.join(self.output_dir, 'error_analysis')
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.models_dir, exist_ok=True)
            os.makedirs(self.error_analysis_dir, exist_ok=True)

        # Load target variable configuration
        if "target_variable" in config_json_data:
            self.target_variable = config_json_data["target_variable"]
            print(f"Target variable: {self.target_variable}")

        if "target_variables" in config_json_data:
            self.target_variables = config_json_data["target_variables"]
            print(f"Multi-output targets: {self.target_variables}")

        # Load temporal split configuration
        if "temporal_split" in config_json_data:
            self.temporal_split = config_json_data["temporal_split"]
            print(f"Temporal split: {self.temporal_split}")
        else:
            # Default based on legacy behavior
            self.temporal_split = False
            print(f"Temporal split: {self.temporal_split} (default)")

        if "time_column" in config_json_data:
            self.time_column = config_json_data["time_column"]
            print(f"Time column: {self.time_column}")

        print("=" * 80)

        key = "scalers"
        if key in config_json_data:
            self.scalers = config_json_data[key]
        else:
            print("File %s missing required key %s" % (config_file, key), file=sys.stderr)
            sys.exit(-4)

        key = "optimizers"
        if key in config_json_data:
            self.optimizers = config_json_data[key]
        else:
            print("File %s missing required key %s" % (config_file, key), file=sys.stderr)
            sys.exit(-4)

        key = "batch_sizes"
        if key in config_json_data:
            self.batch_sizes = config_json_data[key]
        else:
            print("File %s missing required key %s" % (config_file, key), file=sys.stderr)
            sys.exit(-4)

        key = "activations"
        if key in config_json_data:
            self.activations = config_json_data[key]
        else:
            print("File %s missing required key %s" % (config_file, key), file=sys.stderr)
            sys.exit(-4)

        key = "dropout_rates"
        if key in config_json_data:
            self.dropout_rates = config_json_data[key]
        else:
            print("File %s missing required key %s" % (config_file, key), file=sys.stderr)
            sys.exit(-4)

        key = "epochs"
        if key in config_json_data:
            self.epochs = config_json_data[key]
        else:
            print("File %s missing required key %s" % (config_file, key), file=sys.stderr)
            sys.exit(-4)

        key = "patience"
        if key in config_json_data:
            self.patience = config_json_data[key]
        else:
            print("File %s missing required key %s" % (config_file, key), file=sys.stderr)
            sys.exit(-4)

        key = "training_fraction"
        if key in config_json_data:
            self.training_fraction = config_json_data[key]
        else:
            print("File %s missing required key %s" % (config_file, key), file=sys.stderr)
            sys.exit(-4)

        key = "validation_fraction"
        if key in config_json_data:
            self.validation_fraction = config_json_data[key]
        else:
            print("File %s missing required key %s" % (config_file, key), file=sys.stderr)
            sys.exit(-4)

        if "usegpu" in config_json_data:
            print("Using GPU %d" % (config_json_data["usegpu"]))
            tf.config.experimental.set_visible_devices(
                tf.config.list_physical_devices('GPU')[config_json_data["usegpu"]], 'GPU')

        if "mirroredstrategy" in config_json_data:
            if config_json_data['mirroredstrategy']:
                if "usegpu" in config_json_data:
                    print("config json options 'usegpu' and 'mirroredstrategy' are mutually exclusive", file=sys.stderr)
                    sys.exit(-4)
                tf.distribute.MirroredStrategy()
                print("Mirror active")

        if "gpu_mem_grow" in config_json_data:
            if config_json_data['gpu_mem_grow']:
                print("Setting gpu_options.allow_growth=True")
                gpus = tf.config.experimental.list_physical_devices('GPU')
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

    def load_and_validate_data(self):
        """
        Load data from configured data_file and validate that all requested columns exist.
        Returns: DataFrame with only the requested feature columns plus target columns
        """
        # Check if data file exists
        if not os.path.exists(self.data_file):
            print(f"ERROR: Data file not found: {self.data_file}", file=sys.stderr)
            sys.exit(1)

        # Load the CSV
        print(f"\nLoading data from: {self.data_file}")
        df = pd.read_csv(self.data_file)
        print(f"Rows loaded: {len(df)}")

        # Get CSV columns
        csv_columns = set(df.columns)

        # Check all requested feature columns exist
        missing_columns = [col for col in self.feature_columns if col not in csv_columns]
        if missing_columns:
            print(f"ERROR: Columns missing from CSV: {', '.join(missing_columns)}", file=sys.stderr)
            sys.exit(1)

        # Determine target columns to keep
        target_cols = []
        if self.target_variables:
            target_cols = self.target_variables
        else:
            target_cols = [self.target_variable]

        # Check target columns exist
        missing_targets = [col for col in target_cols if col not in csv_columns]
        if missing_targets:
            print(f"ERROR: Target columns missing from CSV: {', '.join(missing_targets)}", file=sys.stderr)
            sys.exit(1)

        # Add time column if using temporal split
        columns_to_keep = self.feature_columns + target_cols
        if self.temporal_split and self.time_column not in columns_to_keep:
            if self.time_column not in csv_columns:
                print(f"ERROR: Time column missing from CSV: {self.time_column}", file=sys.stderr)
                sys.exit(1)
            columns_to_keep.append(self.time_column)

        # Select only the columns we need
        df = df[columns_to_keep]

        print(f"Feature columns selected: {len(self.feature_columns)}")
        print(f"Target column(s): {', '.join(target_cols)}")
        print(f"Split strategy: {'temporal' if self.temporal_split else 'record-order'}")
        if self.temporal_split:
            print(f"Time column: {self.time_column}")

        return df

    def preprocess_data(self, data):
        """
        Preprocess the input data.
        In the new config-driven approach, column selection is handled by load_and_validate_data,
        so this method only needs to handle any remaining preprocessing.
        """
        df = data.copy()
        print("\nPreprocessing data...")
        print(f"Columns in dataset: {df.columns.tolist()}")

        # Data is already filtered to only include feature columns + targets + time column
        # No additional column dropping needed

        return df

    def create_model(self, input_dim, architecture, activation='relu', dropout_rate=0.2, num_outputs=1):
        """
        Create a neural network model with the specified architecture
        num_outputs: 1 for single target, >1 for multi-output
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(shape=(input_dim,)))

        for units in architecture:
            model.add(tf.keras.layers.Dense(units, activation=activation))
            model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Dense(num_outputs))

        return model

    def get_scaler(self, scaler_name):
        """
        Get the scaler object based on name
        """
        if scaler_name == 'standard':
            return StandardScaler()
        elif scaler_name == 'minmax':
            return MinMaxScaler()
        elif scaler_name == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler: {scaler_name}")

    def generate_config_name(self, experiment_type, arch, scaler_name, optimizer, batch_size, activation, dropout_rate):
        """
        Generate a stable, unique configuration name for a hyperparameter combination.
        This name is used for model files, checkpoints, and result rows.
        """
        arch_str = '_'.join(map(str, arch))
        config_name = (f"{experiment_type}_{scaler_name}_{optimizer}_"
                       f"{batch_size}_{activation}_{dropout_rate}_{arch_str}")
        return config_name

    def load_checkpoint(self):
        """
        Load checkpoint file to resume training.
        Returns dict with completed_config_names, last_completed_config_name, etc.
        """
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load checkpoint file: {e}", file=sys.stderr)
                return None
        return None

    def save_checkpoint(self, config_name, config_num):
        """
        Save checkpoint atomically using temporary file and os.replace.
        Stores completed_config_names list, last_completed info, and timestamp.
        """
        # Load existing checkpoint to preserve completed list
        existing_checkpoint = self.load_checkpoint()
        if existing_checkpoint and 'completed_config_names' in existing_checkpoint:
            completed_configs = existing_checkpoint['completed_config_names']
        else:
            completed_configs = []

        # Add current config if not already in list
        if config_name not in completed_configs:
            completed_configs.append(config_name)

        checkpoint = {
            'completed_config_names': completed_configs,
            'last_completed_config_name': config_name,
            'last_completed_config_num': config_num,
            'updated_at': datetime.utcnow().isoformat() + 'Z'
        }

        # Write atomically using temporary file
        tmp_file = self.checkpoint_file + '.tmp'
        try:
            with open(tmp_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            os.replace(tmp_file, self.checkpoint_file)
        except Exception as e:
            print(f"Warning: Could not save checkpoint: {e}", file=sys.stderr)
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except:
                    pass

    def get_completed_configs(self, experiment_type, resume):
        """
        Determine which configurations have already been completed.
        Checks:
        1. Checkpoint file for completed_config_names
        2. Model files in models directory
        3. Existing results CSV

        Returns: set of completed config_name strings
        """
        completed = set()

        if not resume:
            return completed

        print("\n" + "=" * 80)
        print("RESUME MODE: Checking for completed configurations")
        print("=" * 80)

        # 1. Load checkpoint file
        checkpoint = self.load_checkpoint()
        if checkpoint and 'completed_config_names' in checkpoint:
            completed.update(checkpoint['completed_config_names'])
            print(f"Checkpoint file: {len(checkpoint['completed_config_names'])} completed configs")

        # 2. Scan models directory for .keras files
        if os.path.exists(self.models_dir):
            model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.keras')]
            for model_file in model_files:
                config_name = model_file.replace('.keras', '')
                completed.add(config_name)
            print(f"Model files: {len(model_files)} .keras files found")

        # 3. Check existing results CSV
        results_file = os.path.join(self.output_dir, f'{experiment_type.lower()}_grid_search_results.csv')
        if os.path.exists(results_file):
            try:
                existing_results = pd.read_csv(results_file)
                if 'config_name' in existing_results.columns:
                    csv_completed = set(existing_results['config_name'].dropna())
                    completed.update(csv_completed)
                    print(f"Results CSV: {len(csv_completed)} completed configs")
            except Exception as e:
                print(f"Warning: Could not read results CSV: {e}", file=sys.stderr)

        print(f"Total completed configurations: {len(completed)}")
        print("=" * 80 + "\n")

        return completed

    def append_result_to_csv(self, result, experiment_type):
        """
        Append a single result to the results CSV file, avoiding duplicates.
        Loads existing CSV, checks for duplicates by config_name, then writes.
        """
        results_file = os.path.join(self.output_dir, f'{experiment_type.lower()}_grid_search_results.csv')

        # Load existing results if file exists
        if os.path.exists(results_file):
            try:
                existing_results = pd.read_csv(results_file)

                # Remove duplicate if it exists (based on config_name)
                if 'config_name' in existing_results.columns:
                    existing_results = existing_results[existing_results['config_name'] != result['config_name']]

                # Append new result
                updated_results = pd.concat([existing_results, pd.DataFrame([result])], ignore_index=True)
            except Exception as e:
                print(f"Warning: Could not load existing results, creating new file: {e}", file=sys.stderr)
                updated_results = pd.DataFrame([result])
        else:
            updated_results = pd.DataFrame([result])

        # Write back to CSV
        updated_results.to_csv(results_file, index=False)

    def calculate_error_metrics(self, y_true, y_pred):
        """
        Calculate comprehensive error metrics
        """
        errors = np.abs(y_true - y_pred)
        relative_errors = errors / (y_true + 1e-10)

        metrics = {
            'mean_error': np.mean(errors),
            'median_error': np.median(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors),
            'mean_relative_error': np.mean(relative_errors),
            'median_relative_error': np.median(relative_errors)
        }

        return metrics

    def analyze_worst_predictions(self, y_true, y_pred, X, config_name, dataset_name='test'):
        """
        Analyze worst predictions and save per-target CSV files (no plots)
        """
        # Ensure y_true and y_pred are 2D
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)

        num_targets = y_true.shape[1]

        # Get target names
        if self.target_variables and len(self.target_variables) == num_targets:
            target_names = self.target_variables
        else:
            target_names = [self.target_variable] if num_targets == 1 else [f"target_{i}" for i in range(num_targets)]

        eps = 1e-12
        result_metrics = {}

        # Process each target separately
        for i in range(num_targets):
            target_name = target_names[i]
            true_vals = y_true[:, i]
            pred_vals = y_pred[:, i]

            errors = np.abs(true_vals - pred_vals)
            rel_errors = errors / np.maximum(np.abs(true_vals), eps)

            # Create dataframe for this target
            error_df = pd.DataFrame({
                'true_value': true_vals,
                'predicted_value': pred_vals,
                'absolute_error': errors,
                'relative_error': rel_errors
            })

            # Sort by absolute error and get worst 100 cases
            worst_cases = error_df.nlargest(100, 'absolute_error')

            # Save to CSV (per target)
            filename = f'{config_name}_{dataset_name}_worst_predictions_{target_name}.csv'
            filepath = os.path.join(self.error_analysis_dir, filename)
            worst_cases.to_csv(filepath, index=False)

            # Store metrics for this target
            result_metrics[f'mean_error_{target_name}'] = np.mean(errors)
            result_metrics[f'median_error_{target_name}'] = np.median(errors)
            result_metrics[f'error_std_{target_name}'] = np.std(errors)
            result_metrics[f'max_error_{target_name}'] = np.max(errors)
            result_metrics[f'p95_abs_error_{target_name}'] = np.percentile(errors, 95)

        # For backwards compatibility, also compute pooled metrics if multi-output
        if num_targets > 1:
            all_errors = np.abs(y_true - y_pred).flatten()
            result_metrics['mean_error'] = np.mean(all_errors)
            result_metrics['median_error'] = np.median(all_errors)
            result_metrics['error_std'] = np.std(all_errors)
            result_metrics['max_error'] = np.max(all_errors)
        else:
            # For single output, use the target-specific metrics as the main metrics
            target_name = target_names[0]
            result_metrics['mean_error'] = result_metrics[f'mean_error_{target_name}']
            result_metrics['median_error'] = result_metrics[f'median_error_{target_name}']
            result_metrics['error_std'] = result_metrics[f'error_std_{target_name}']
            result_metrics['max_error'] = result_metrics[f'max_error_{target_name}']

        return result_metrics

    def run_grid_search(self, data, architectures, experiment_type, resume=False, trial_run=False):
        """
        Run grid search over all hyperparameter combinations.
        Supports resume mode to skip already-completed configurations.
        """
        # Preprocess data
        processed_data = self.preprocess_data(data)

        # Determine target columns
        if self.target_variables:
            target_columns = self.target_variables
            num_outputs = len(self.target_variables)
        else:
            target_columns = [self.target_variable]
            num_outputs = 1

        # Separate features and targets
        feature_cols = [col for col in processed_data.columns if col not in target_columns and col != self.time_column]
        X = processed_data[feature_cols].values

        if num_outputs == 1:
            y = processed_data[target_columns[0]].values
        else:
            y = processed_data[target_columns].values

        # Split data based on temporal_split setting
        if self.temporal_split:
            print(f"\nUsing temporal split based on {self.time_column}")
            # Sort by time column
            time_values = processed_data[self.time_column].values
            sort_idx = np.argsort(time_values)
            X = X[sort_idx]
            y = y[sort_idx]

            # Split chronologically
            train_size = int(len(X) * self.training_fraction)
            val_size = int(len(X) * self.validation_fraction)

            X_train = X[:train_size]
            y_train = y[:train_size]
            X_val = X[train_size:train_size + val_size]
            y_val = y[train_size:train_size + val_size]
            X_test = X[train_size + val_size:]
            y_test = y[train_size + val_size:]
        else:
            print("\nUsing record-order split (no temporal ordering)")
            # Use simple sequential split
            train_size = int(len(X) * self.training_fraction)
            val_size = int(len(X) * self.validation_fraction)

            X_train = X[:train_size]
            y_train = y[:train_size]
            X_val = X[train_size:train_size + val_size]
            y_val = y[train_size:train_size + val_size]
            X_test = X[train_size + val_size:]
            y_test = y[train_size + val_size:]

        print(f"\nData split:")
        print(f"  Training set: {len(X_train)} samples")
        print(f"  Validation set: {len(X_val)} samples")
        print(f"  Test set: {len(X_test)} samples")

        # Get completed configurations if in resume mode
        completed_configs = self.get_completed_configs(experiment_type, resume)

        total_configs = (len(architectures) * len(self.scalers) * len(self.optimizers) *
                         len(self.batch_sizes) * len(self.activations) * len(self.dropout_rates))

        print(f"\nTotal configurations to test: {total_configs}")
        if resume:
            remaining = total_configs - len(completed_configs)
            print(f"Configurations already completed: {len(completed_configs)}")
            print(f"Configurations remaining: {remaining}")

        if trial_run:
            print("\nTRIAL RUN MODE - Not executing training")
            return pd.DataFrame()

        config_num = 0
        skipped = 0
        trained = 0

        for arch in architectures:
            for scaler_name in self.scalers:
                for optimizer in self.optimizers:
                    for batch_size in self.batch_sizes:
                        for activation in self.activations:
                            for dropout_rate in self.dropout_rates:
                                config_num += 1

                                # Generate config name BEFORE training
                                config_name = self.generate_config_name(
                                    experiment_type, arch, scaler_name, optimizer,
                                    batch_size, activation, dropout_rate
                                )

                                # Check if already completed
                                if config_name in completed_configs:
                                    skipped += 1
                                    print(
                                        f"\n[{config_num}/{total_configs}] Skipping completed configuration: {config_name}")
                                    continue

                                trained += 1
                                print(f"\n{'=' * 80}")
                                print(f"Configuration {config_num}/{total_configs} (Training #{trained})")
                                print(f"{'=' * 80}")
                                print(f"Config name: {config_name}")
                                print(f"Architecture: {arch}")
                                print(f"Scaler: {scaler_name}")
                                print(f"Optimizer: {optimizer}")
                                print(f"Batch size: {batch_size}")
                                print(f"Activation: {activation}")
                                print(f"Dropout: {dropout_rate}")

                                try:
                                    # Scale the data
                                    scaler = self.get_scaler(scaler_name)
                                    X_train_scaled = scaler.fit_transform(X_train)
                                    X_val_scaled = scaler.transform(X_val)
                                    X_test_scaled = scaler.transform(X_test)

                                    # Create model
                                    model = self.create_model(
                                        X_train_scaled.shape[1],
                                        arch,
                                        activation=activation,
                                        dropout_rate=dropout_rate,
                                        num_outputs=num_outputs
                                    )

                                    # Compile model
                                    model.compile(
                                        optimizer=optimizer,
                                        loss='mse',
                                        metrics=['mae'],
                                        run_eagerly=False,
                                        jit_compile=False
                                    )

                                    # Train model with early stopping
                                    early_stopping = tf.keras.callbacks.EarlyStopping(
                                        monitor='val_loss',
                                        patience=self.patience,
                                        restore_best_weights=True
                                    )

                                    history = model.fit(
                                        X_train_scaled, y_train,
                                        validation_data=(X_val_scaled, y_val),
                                        epochs=self.epochs,
                                        batch_size=batch_size,
                                        callbacks=[early_stopping],
                                        verbose=0
                                    )

                                    # Make predictions
                                    train_pred = model.predict(X_train_scaled, verbose=0)
                                    val_pred = model.predict(X_val_scaled, verbose=0)
                                    test_pred = model.predict(X_test_scaled, verbose=0)

                                    # Keep predictions as (N, K) for multi-output or (N, 1) for single-output
                                    # analyze_worst_predictions will handle the shape

                                    # Analyze predictions
                                    train_analysis = self.analyze_worst_predictions(
                                        y_train, train_pred, X_train_scaled, config_name, 'train'
                                    )
                                    val_analysis = self.analyze_worst_predictions(
                                        y_val, val_pred, X_val_scaled, config_name, 'val'
                                    )
                                    test_analysis = self.analyze_worst_predictions(
                                        y_test, test_pred, X_test_scaled, config_name, 'test'
                                    )

                                    # Save model
                                    model_path = os.path.join(self.models_dir, f'{config_name}.keras')
                                    model.save(model_path)

                                    # Save scaler
                                    scaler_path = os.path.join(self.models_dir, f'{config_name}_scaler.pkl')
                                    with open(scaler_path, 'wb') as f:
                                        pickle.dump(scaler, f)

                                    # Record results with config_name
                                    result = {
                                        'config_name': config_name,
                                        'architecture': str(arch),
                                        'scaler': scaler_name,
                                        'optimizer': optimizer,
                                        'batch_size': batch_size,
                                        'activation': activation,
                                        'dropout_rate': dropout_rate,
                                        'train_mae': train_analysis['mean_error'],
                                        'val_mae': val_analysis['mean_error'],
                                        'test_mae': test_analysis['mean_error'],
                                        'train_error_std': train_analysis['error_std'],
                                        'val_error_std': val_analysis['error_std'],
                                        'test_error_std': test_analysis['error_std'],
                                        'epochs': len(history.history['loss'])
                                    }

                                    # Add per-target metrics
                                    for key in train_analysis.keys():
                                        if key.startswith(('mean_error_', 'median_error_', 'error_std_', 'max_error_',
                                                           'p95_abs_error_')):
                                            result[f'train_{key}'] = train_analysis[key]
                                    for key in val_analysis.keys():
                                        if key.startswith(('mean_error_', 'median_error_', 'error_std_', 'max_error_',
                                                           'p95_abs_error_')):
                                            result[f'val_{key}'] = val_analysis[key]
                                    for key in test_analysis.keys():
                                        if key.startswith(('mean_error_', 'median_error_', 'error_std_', 'max_error_',
                                                           'p95_abs_error_')):
                                            result[f'test_{key}'] = test_analysis[key]

                                    # Append result to CSV (dedupe-safe)
                                    self.append_result_to_csv(result, experiment_type)

                                    # Save checkpoint
                                    self.save_checkpoint(config_name, config_num)

                                    print(f"Configuration completed. Test MAE: {test_analysis['mean_error']:.4f}")
                                    print(f"Checkpoint saved: {config_name}")

                                except Exception as e:
                                    print(f"Error with configuration: {str(e)}")
                                    import traceback
                                    traceback.print_exc()
                                    continue

        print(f"\n{'=' * 80}")
        print(f"Grid search complete!")
        print(f"Total configurations: {total_configs}")
        print(f"Skipped (already completed): {skipped}")
        print(f"Trained in this run: {trained}")
        print(f"{'=' * 80}\n")

        # Return empty dataframe (results are saved incrementally)
        return pd.DataFrame()


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Run hyperparameter search experiments')

    # Add experiment type selection
    parser.add_argument('--experiments', nargs='+', choices=['2DSA', 'GA', 'PCSA'],
                        help='Specify which experiments to run (2DSA, GA, PCSA)')

    # Add architecture selection
    parser.add_argument('--architectures', nargs='+', type=int, action='append',
                        help='Specify neural network architectures (e.g., --architectures 64 32 --architectures 128 64)')

    # Add hyperparameter selection
    parser.add_argument('--optimizers', nargs='+', choices=['adam', 'rmsprop', 'sgd'],
                        help='Specify optimizers to use')
    parser.add_argument('--batch-sizes', nargs='+', type=int,
                        help='Specify batch sizes to use')
    parser.add_argument('--activations', nargs='+', choices=['relu', 'elu'],
                        help='Specify activation functions to use')
    parser.add_argument('--dropout-rates', nargs='+', type=float,
                        help='Specify dropout rates to use')
    parser.add_argument('--scalers', nargs='+', choices=['standard', 'minmax', 'robust'],
                        help='Specify scalers to use')

    # Add output directory and checkpoint file options
    parser.add_argument('--output-dir', type=str, default='results/prediction',
                        help='Specify output directory for results')
    parser.add_argument('--checkpoint-file', type=str,
                        help='Specify custom checkpoint file location')

    # Add resume option
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume from last checkpoint, skipping completed configurations')

    # config file
    parser.add_argument('--config-file', help='read options from JSON formatted configuration file')

    # trial run
    parser.add_argument('--trial-run', action='store_true', help='do not actually run')

    return parser.parse_args()


def main():
    """
    Main function to run the hyperparameter search experiments
    """
    args = parse_args()

    # Set default architectures if not specified
    if args.architectures is None:
        architectures = [
            [64, 32],
            [128, 64],
            [256, 128, 64],
            [512, 256, 128]
        ]
    else:
        architectures = args.architectures

    experiment_files = {
        '2DSA': './preprocess/results/filtered/2dsa-filtered.csv',
        'GA': './preprocess/results/filtered/ga-filtered.csv',
        'PCSA': './preprocess/results/filtered/pcsa-filtered.csv'
    }

    # Filter experiments if specified
    if args.experiments:
        experiment_files = {k: v for k, v in experiment_files.items() if k in args.experiments}

    ## optionally load from config - overrides other parameters!
    if "config_file" in args:
        config_file = args.config_file
        print("Loading from config file %s\n" % (config_file))
        if not os.path.isfile(config_file):
            print("Error: file %s not found\n" % (config_file), file=sys.stderr)
            sys.exit(-2)
        with open(config_file, 'r') as f:
            ## strip out comment lines
            lines = '';
            for line in f:
                line = line.strip()
                if not line.startswith('#'):
                    lines += line

            ## convert json string to dictionary
            try:
                config_json_data = json.loads(lines)
            except json.JSONDecodeError as e:
                print("Error decoding JSON:", e, file=sys.stderr)
                print("json:\n", lines)
                sys.exit(-3)
            # Access the parsed JSON data
            print(config_json_data)

            # Get architectures from config
            key = "architectures"
            if key in config_json_data:
                architectures = config_json_data[key]
            else:
                print("File %s missing required key %s" % (config_file, key), file=sys.stderr)
                sys.exit(-4)

            if "output_dir" in config_json_data:
                args.output_dir = config_json_data['output_dir']
                print(f'Config: output_dir is {args.output_dir}')

            # Initialize framework with specified output directory and checkpoint file
            framework = EnhancedPredictionFramework(
                output_dir=args.output_dir,
                checkpoint_file=args.checkpoint_file
            )

            framework.load_config(config_file, config_json_data)

            # Check if dry run
            if framework.dry_run:
                print("\nDRY RUN MODE - Validation complete, exiting without training")
                sys.exit(0)

            # Load and validate data using new config-driven approach
            data = framework.load_and_validate_data()

            # Use method name as experiment type (uppercase for consistency)
            experiment_type = framework.method.upper()

            # Run grid search with resume support
            print(f"\nProcessing {experiment_type} experiments...")
            results = framework.run_grid_search(data, architectures, experiment_type,
                                                resume=args.resume, trial_run=args.trial_run)
            results['experiment_type'] = experiment_type

    else:
        # Initialize framework with specified output directory and checkpoint file
        framework = EnhancedPredictionFramework(
            output_dir=args.output_dir,
            checkpoint_file=args.checkpoint_file
        )

        # Update framework parameters if specified
        if args.optimizers:
            framework.optimizers = args.optimizers
        if args.batch_sizes:
            framework.batch_sizes = args.batch_sizes
        if args.activations:
            framework.activations = args.activations
        if args.dropout_rates:
            framework.dropout_rates = args.dropout_rates
        if args.scalers:
            framework.scalers = args.scalers

        for exp_type, file_path in experiment_files.items():
            print(f"\nProcessing {exp_type} experiments...")
            data = pd.read_csv(file_path)

            results = framework.run_grid_search(data, architectures, exp_type,
                                                resume=args.resume, trial_run=args.trial_run)
            results['experiment_type'] = exp_type

    print("\nAll experiments completed successfully!")


if __name__ == "__main__":
    main()
    
