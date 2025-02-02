#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime

def load_model(model_path):
    """Load the TensorFlow model from specified path."""
    print(f"\nAttempting to load model from: {model_path}")
    print(f"Path exists: {os.path.exists(model_path)}")
    if os.path.isdir(model_path):
        print(f"Directory contents: {os.listdir(model_path)}")

        # Check if there's a nested 'model' directory
        model_subdir = os.path.join(model_path, 'model')
        if os.path.exists(model_subdir):
            print(f"\nFound nested model directory. Contents: {os.listdir(model_subdir)}")
            model_path = model_subdir

    try:
        # First try loading as TensorFlow SavedModel using TFSMLayer
        print("\nAttempting to load as TensorFlow SavedModel using TFSMLayer...")
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.TFSMLayer(
                    model_path,
                    call_endpoint='serving_default'
                )
            ])
            print("Success!")
            return model
        except Exception as e:
            print(f"Failed TFSMLayer approach with error: {e}")

            # Try alternate endpoints if 'serving_default' failed
            if 'call_endpoint' in str(e):
                try:
                    imported = tf.saved_model.load(model_path)
                    print("\nAvailable endpoints:", list(imported.signatures.keys()))

                    endpoint = list(imported.signatures.keys())[0]
                    print(f"Trying with endpoint: {endpoint}")

                    model = tf.keras.Sequential([
                        tf.keras.layers.TFSMLayer(
                            model_path,
                            call_endpoint=endpoint
                        )
                    ])
                    print("Success!")
                    return model
                except Exception as e2:
                    print(f"Failed alternate endpoint approach with error: {e2}")

        print("\nAttempting to load with tf.keras.models.load_model...")
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print("Success!")
            return model
        except Exception as e:
            print(f"Failed with error: {e}")

        raise Exception("All loading attempts failed")

    except Exception as e:
        print(f"\nError: Could not load model after trying all methods")
        print("\nTroubleshooting tips:")
        print("1. For SavedModel format in Keras 3: The model must be loaded using TFSMLayer")
        print("2. For Keras 3: Native format should use .keras extension")
        print("3. For legacy H5 format: Use .h5 extension")
        print(f"4. Check if model is in subdirectory: {os.path.join(model_path, 'model')}")
        print("\nTensorFlow version:", tf.__version__)
        print("Keras version:", tf.keras.__version__)
        sys.exit(1)

def load_data(data_path, model_dir=None):
    """Load and preprocess data from CSV."""
    print(f"\nAttempting to load data from: {data_path}")
    print(f"File exists: {os.path.exists(data_path)}")

    try:
        data = pd.read_csv(data_path)
        print("\nData loaded successfully!")
        print(f"Columns found: {data.columns.tolist()}")
        print(f"Number of rows: {len(data)}")

        # Try to load scaler if it exists
        if model_dir and os.path.exists(os.path.join(model_dir, 'scaler.pkl')):
            print("\nFound scaler.pkl, loading and applying transformation...")
            import pickle
            with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
                scaler = pickle.load(f)
            return data, scaler

        return data, None
    except Exception as e:
        print(f"\nError loading data: {e}")
        sys.exit(1)

def generate_predictions(model, data, scaler):
    """Generate predictions for input data."""
    try:
        # Get feature columns (all columns available)
        X = data.values

        # Apply scaler if it exists
        if scaler is not None:
            print("\nApplying scaler transformation to input features...")
            X = scaler.transform(X)

        y_pred = model.predict(X)
        # Handle different shapes of predictions
        if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()

        # Prepare results
        results_df = data.copy()
        results_df['predicted_cputime'] = y_pred

        # Prepare summary statistics
        summary_stats = {
            'total_records': len(y_pred),
            'min_predicted': np.min(y_pred),
            'max_predicted': np.max(y_pred),
            'mean_predicted': np.mean(y_pred),
            'median_predicted': np.median(y_pred),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return results_df, summary_stats
    except Exception as e:
        print(f"Error generating predictions: {e}")
        sys.exit(1)

def save_results(results_df, summary_stats, output_dir, run_name):
    """Save detailed results and summary statistics to CSV files."""
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        results_path = output_dir / f"{run_name}_detailed_results.csv"
        results_df.to_csv(results_path, index=False)

        # Save summary statistics
        summary_path = output_dir / f"{run_name}_summary_stats.csv"
        pd.DataFrame([summary_stats]).to_csv(summary_path, index=False)

        print(f"Results saved to:\n{results_path}\n{summary_path}")
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run predictions using a TensorFlow model')
    parser.add_argument('--model', required=True, help='Path to the TensorFlow model')
    parser.add_argument('--data', required=True, help='Path to input CSV data')
    parser.add_argument('--output', default='results/singleruns',
                        help='Output directory (default: results/singleruns)')
    parser.add_argument('--run-name', default=datetime.now().strftime('%Y%m%d_%H%M%S'),
                        help='Name for this prediction run (default: timestamp)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print additional information during execution')
    return parser.parse_args()

def main():
    args = parse_arguments()

    if args.verbose:
        print(f"Loading model from: {args.model}")
    model = load_model(args.model)

    if args.verbose:
        print(f"Loading data from: {args.data}")
    data, scaler = load_data(args.data, args.model)

    if args.verbose:
        print("Generating predictions...")
    results_df, summary_stats = generate_predictions(model, data, scaler)

    if args.verbose:
        print(f"Saving results to: {args.output}")
    save_results(results_df, summary_stats, args.output, args.run_name)

if __name__ == "__main__":
    main()
