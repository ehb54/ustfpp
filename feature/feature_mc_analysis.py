import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, Any, Union, Optional
import warnings
from data_loader import DataLoader
import argparse
from time import time
import logging
import sys
import traceback

class MonteCarloFeatureSelector:
    def __init__(self, data_loader, config: Union[Dict[str, Any], str]):
        """
        Initialize with data loader and config (either as path or dict).

        Args:
            data_loader: The data loader instance
            config: Either a path to config JSON file (str) or config dictionary
        """
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        try:
            self.data_loader = data_loader
            self.config = self._load_config(config)
            self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self._initialize_trackers()

            # Filter out excluded columns
            self._filter_features()

            self.logger.info("\nInitialization complete. Configuration loaded successfully.")
            self.logger.info(f"Number of features: {len(self.filtered_features.columns)}")
            self.logger.info(f"Number of samples: {len(self.filtered_features)}")
            if self.excluded_columns:
                self.logger.info(f"Excluded columns: {', '.join(self.excluded_columns)}\n")

        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def _load_config(self, config: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """Load configuration from either JSON file or dict."""
        self.logger.info("Loading configuration...")
        defaults = {
            'n_iterations': 100,
            'test_size': 0.2,
            'output_dir': 'results/monte_carlo',
            'excluded_columns': [],  # Default for excluded columns
            'lasso_params': {
                'alpha': 1.0,
                'max_iter': 1000,
                'fit_intercept': True,
                'tol': 1e-4,
                'selection': 'cyclic'
            },
            'xgboost_params': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'objective': 'reg:squarederror',
                'verbosity': 0,
                'early_stopping_rounds': None,
                'eval_metric': 'rmse',
                'tree_method': 'hist'
            },
            'rf_params': {
                'n_estimators': 200,
                'max_depth': None,
                'min_samples_split': 5,
                'min_samples_leaf': 3,
                'random_state': 42,
                'n_jobs': -1,
                'max_features': 'sqrt',
                'bootstrap': True,
                'oob_score': True
            }
        }

        try:
            if isinstance(config, str):
                self.logger.info(f"Reading config from file: {config}")
                with open(config, 'r') as f:
                    loaded_config = json.load(f)
            else:
                loaded_config = config

            for key, value in defaults.items():
                if key not in loaded_config:
                    loaded_config[key] = value
                elif isinstance(value, dict):
                    loaded_config[key] = {**value, **loaded_config.get(key, {})}

            return loaded_config

        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'monte_carlo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
    def _initialize_trackers(self):
        """Initialize tracking dictionaries for all methods."""
        try:
            self.results = {
                'lasso': {
                    'importance_scores': [],
                    'selection_frequency': {},
                    'execution_times': []
                },
                'xgboost': {
                    'importance_scores': [],
                    'selection_frequency': {},
                    'execution_times': []
                },
                'random_forest': {
                    'importance_scores': [],
                    'selection_frequency': {},
                    'execution_times': []
                }
            }
            self.logger.debug("Initialized tracking dictionaries")
        except Exception as e:
            self.logger.error(f"Failed to initialize trackers: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def _run_lasso_iteration(self, X_train, y_train, feature_names):
        """Run single LASSO iteration."""
        try:
            start_time = time()

            # Filter valid Lasso parameters
            valid_lasso_params = {
                'alpha', 'fit_intercept', 'precompute', 'copy_X', 'max_iter',
                'tol', 'warm_start', 'positive', 'random_state', 'selection'
            }
            lasso_params = {k: v for k, v in self.config['lasso_params'].items()
                            if k in valid_lasso_params}

            # Fit Lasso
            lasso = Lasso(**lasso_params)
            lasso.fit(X_train, y_train)

            # Store feature importance scores
            self.results['lasso']['importance_scores'].append(np.abs(lasso.coef_))

            # Track selected features (non-zero coefficients)
            selected_features = feature_names[np.abs(lasso.coef_) > 0].tolist()
            for feature in selected_features:
                self.results['lasso']['selection_frequency'][feature] = \
                    self.results['lasso']['selection_frequency'].get(feature, 0) + 1

            self.results['lasso']['execution_times'].append(time() - start_time)

        except Exception as e:
            self.logger.error(f"Lasso iteration failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def _run_rf_iteration(self, X_train, y_train, feature_names):
        """Run single Random Forest iteration."""
        try:
            start_time = time()

            # Fit Random Forest
            model = RandomForestRegressor(**self.config['rf_params'])
            model.fit(X_train, y_train)

            if hasattr(model, 'oob_score_'):
                self.logger.info(f"OOB Score: {model.oob_score_:.4f}")

            # Store feature importance scores
            self.results['random_forest']['importance_scores'].append(model.feature_importances_)

            # Track selected features (importance > 0)
            selected_features = feature_names[model.feature_importances_ > 0].tolist()
            for feature in selected_features:
                self.results['random_forest']['selection_frequency'][feature] = \
                    self.results['random_forest']['selection_frequency'].get(feature, 0) + 1

            self.results['random_forest']['execution_times'].append(time() - start_time)

        except Exception as e:
            self.logger.error(f"Random Forest iteration failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def _run_xgboost_iteration(self, X_train, y_train, X_val, y_val, feature_names):
        """Run single XGBoost iteration."""
        try:
            start_time = time()

            # Fit XGBoost
            model = xgb.XGBRegressor(**self.config['xgboost_params'])
            eval_set = [(X_val, y_val)]
            model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

            # Store feature importance scores
            self.results['xgboost']['importance_scores'].append(model.feature_importances_)

            # Track selected features (importance > 0)
            selected_features = feature_names[model.feature_importances_ > 0].tolist()
            for feature in selected_features:
                self.results['xgboost']['selection_frequency'][feature] = \
                    self.results['xgboost']['selection_frequency'].get(feature, 0) + 1

            self.results['xgboost']['execution_times'].append(time() - start_time)

        except Exception as e:
            self.logger.error(f"XGBoost iteration failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def _validate_data(self) -> bool:
        """Validate input data and configuration."""
        try:
            # Check if target variable exists
            if self.data_loader.target is None:
                self.logger.error("Target variable not found in data")
                return False

            # Check if we have enough samples
            if len(self.filtered_features) < 10:
                self.logger.error("Too few samples for analysis")
                return False

            # Check if we have any features left after filtering
            if len(self.filtered_features.columns) == 0:
                self.logger.error("No features remaining after filtering")
                return False

            return True
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def _filter_features(self):
        """Filter out excluded columns from the feature set."""
        try:
            self.excluded_columns = self.config.get('excluded_columns', [])
            if not isinstance(self.excluded_columns, list):
                raise ValueError("excluded_columns must be a list")

            # Create copy of features
            self.filtered_features = self.data_loader.encoded_features.copy()
            original_columns = set(self.filtered_features.columns)

            # Remove excluded columns if they exist
            existing_excluded = [col for col in self.excluded_columns
                                 if col in self.filtered_features.columns]
            if existing_excluded:
                self.filtered_features = self.filtered_features.drop(columns=existing_excluded)
                self.logger.info(f"\nExcluded {len(existing_excluded)} columns from analysis")

            # Warn about non-existent columns
            non_existent = set(self.excluded_columns) - original_columns
            if non_existent:
                self.logger.warning(f"The following excluded columns were not found in the dataset: {non_existent}")

        except Exception as e:
            self.logger.error(f"Feature filtering failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def _run_model_safely(self, model_func, X_train, y_train, *args, method_name: str) -> Optional[bool]:
        """Safely execute a model training iteration with error handling."""
        try:
            model_func(X_train, y_train, *args)
            return True
        except Exception as e:
            self.logger.error(f"{method_name} failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def _get_analysis_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get features to use in analysis by excluding specified features.
        Does not modify the original data.
        """
        ignore_features = self.config.get('ignore_features', [])
        cols_to_use = [col for col in X.columns if col not in ignore_features]
        return X[cols_to_use]
    def _compile_results(self):
        """Compile results from all iterations into summary DataFrames."""
        try:
            compiled_results = {}

            for method in ['lasso', 'xgboost', 'random_forest']:
                # Convert importance scores to numpy array
                importance_matrix = np.array(self.results[method]['importance_scores'])

                # Calculate statistics
                mean_importance = np.mean(importance_matrix, axis=0)
                std_importance = np.std(importance_matrix, axis=0)

                # Get feature names
                feature_names = self.filtered_features.columns

                # Calculate selection frequency as percentage
                n_iterations = len(self.results[method]['importance_scores'])
                selection_frequency = {
                    feature: (count / n_iterations) * 100
                    for feature, count in self.results[method]['selection_frequency'].items()
                }

                # Create DataFrame with results
                df = pd.DataFrame({
                    'feature': feature_names,
                    'mean_importance': mean_importance,
                    'std_importance': std_importance,
                    'selection_frequency': [selection_frequency.get(feat, 0) for feat in feature_names],
                    'mean_execution_time': np.mean(self.results[method]['execution_times'])
                })

                # Sort by mean importance
                df = df.sort_values('mean_importance', ascending=False).reset_index(drop=True)

                compiled_results[method] = df

            return compiled_results

        except Exception as e:
            self.logger.error(f"Failed to compile results: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def run_analysis(self):
        """Run Monte Carlo feature selection analysis."""
        if not self._validate_data():
            raise ValueError("Data validation failed")

        # Keep original data intact
        X = self.data_loader.encoded_features
        y = self.data_loader.target

        # Get columns for analysis
        analysis_columns = [col for col in X.columns
                            if col not in self.config.get('ignore_features', [])]

        self.logger.info(f"\nStarting Monte Carlo analysis with {self.config['n_iterations']} iterations...")
        self.logger.info(f"Total features: {len(X.columns)}")
        self.logger.info(f"Features used in analysis: {len(analysis_columns)}")
        ignored = set(X.columns) - set(analysis_columns)
        if ignored:
            self.logger.info(f"Ignoring features: {', '.join(ignored)}")
        self.logger.info(f"Test size: {self.config['test_size']}")

        total_start_time = time()
        successful_iterations = 0

        for iteration in range(self.config['n_iterations']):
            try:
                iter_start_time = time()
                self.logger.info(f"\nIteration {iteration + 1}/{self.config['n_iterations']}")

                # Create random train/test split
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=self.config['test_size'], random_state=iteration
                )
                self.logger.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_val.shape[0]}")

                # Run analyses with error handling
                methods_status = {
                    'Lasso': self._run_model_safely(
                        self._run_lasso_iteration, X_train, y_train, X.columns,
                        method_name='Lasso'
                    ),
                    'XGBoost': self._run_model_safely(
                        self._run_xgboost_iteration, X_train, y_train, X_val, y_val, X.columns,
                        method_name='XGBoost'
                    ),
                    'Random Forest': self._run_model_safely(
                        self._run_rf_iteration, X_train, y_train, X.columns,
                        method_name='Random Forest'
                    )
                }

                if all(methods_status.values()):
                    successful_iterations += 1
                else:
                    failed_methods = [method for method, status in methods_status.items() if not status]
                    self.logger.warning(f"Failed methods in iteration {iteration + 1}: {', '.join(failed_methods)}")

                iter_time = time() - iter_start_time
                self.logger.info(f"Iteration completed in {iter_time:.2f} seconds")

            except Exception as e:
                self.logger.error(f"Iteration {iteration + 1} failed: {str(e)}")
                self.logger.error(traceback.format_exc())
                continue

        total_time = time() - total_start_time
        self.logger.info(f"\nAnalysis completed in {total_time:.2f} seconds")
        self.logger.info(f"Successful iterations: {successful_iterations}/{self.config['n_iterations']}")

        if successful_iterations == 0:
            raise RuntimeError("No successful iterations completed")

        return self._compile_results()

    def save_results(self, results_dict):
        """Save analysis results to CSV with error tracking."""
        try:
            base_output_dir = str(self.config['output_dir'])
            output_dir = f"{base_output_dir}_{self.timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"\nSaving results to {output_dir}")

            # Save results summary
            summary_path = os.path.join(output_dir, 'analysis_summary.txt')
            with open(summary_path, 'w') as f:
                f.write(f"Analysis completed at: {datetime.now()}\n")
                f.write(f"Configuration used: {json.dumps(self.config, indent=2)}\n\n")

                for method, df in results_dict.items():
                    f.write(f"\n{method.upper()} Results:\n")
                    f.write(f"Number of features with non-zero importance: {(df['mean_importance'] > 0).sum()}\n")
                    f.write("Top 10 most important features:\n")
                    f.write(df.head(10).to_string())
                    f.write("\n")

            # Save detailed results
            for method, df in results_dict.items():
                output_path = os.path.join(output_dir, f'{method}_feature_importance.csv')
                df.to_csv(output_path, index=False)
                self.logger.info(f"Saved {method} results to {output_path}")

            # Save configuration
            config_path = os.path.join(output_dir, 'analysis_config.json')
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info(f"Saved configuration to {config_path}")

        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise


        """Save analysis results to CSV."""
        try:
            base_output_dir = str(self.config['output_dir'])
            output_dir = f"{base_output_dir}_{self.timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            print(f"\nSaving results to {output_dir}")

            for method, df in results_dict.items():
                output_path = os.path.join(output_dir, f'{method}_feature_importance.csv')
                df.to_csv(output_path, index=False)
                print(f"Saved {method} results to {output_path}")

            config_path = os.path.join(output_dir, 'analysis_config.json')
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"Saved configuration to {config_path}")

        except Exception as e:
            print(f"Error saving results: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Run Monte Carlo Feature Selection Analysis')
    parser.add_argument('--config', '-c', default='monte_carlo_config.json',
                        help='Path to configuration JSON file')
    args = parser.parse_args()

    print(f"Starting Monte Carlo Feature Selection Analysis at {datetime.now()}")

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, 'r') as f:
        config = json.load(f)

    if 'data_file' not in config:
        raise ValueError("data_file must be specified in the config file")

    if not os.path.exists(config['data_file']):
        raise FileNotFoundError(f"Data file not found: {config['data_file']}")

    print(f"\nLoading data from {config['data_file']}")
    data_loader = DataLoader(config['data_file'])
    mc_selector = MonteCarloFeatureSelector(data_loader, config)
    results = mc_selector.run_analysis()
    mc_selector.save_results(results)

    print("\nDetailed Results Summary:")
    for method, df in results.items():
        print(f"\nTop 10 most important features ({method}):")
        print(df.head(10))
        print("\nFeature importance statistics:")
        print(f"Mean importance: {df['mean_importance'].mean():.4f}")
        print(f"Std importance: {df['mean_importance'].std():.4f}")
        print(f"Number of features with non-zero importance: {(df['mean_importance'] > 0).sum()}")

if __name__ == "__main__":
    main()
