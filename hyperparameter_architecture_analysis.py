import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import os
import json
import pickle
import matplotlib.pyplot as plt

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

    def preprocess_data(self, data):
        """
        Preprocess the input data by handling categorical variables and dropping unnecessary columns
        """
        df = data.copy()
        print("Available columns:", df.columns.tolist())

        # Drop non-feature columns
        df = df.drop(['max_rss', 'wallTime'], axis=1)

        print("Final columns:", df.columns.tolist())
        return df

    def create_model(self, input_dim, architecture, activation='relu', dropout_rate=0.2):
        """
        Create a neural network model with the specified architecture
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))

        for units in architecture:
            model.add(tf.keras.layers.Dense(units, activation=activation))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Dense(1))
        return model

    def get_scaler(self, scaler_name):
        """
        Get the appropriate scaler based on the name
        """
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        return scalers.get(scaler_name)

    def save_checkpoint(self, experiment_type, arch_idx, scaler_idx, optimizer_idx, batch_idx, activation_idx, dropout_idx):
        """
        Save the current state of grid search to a checkpoint file
        """
        checkpoint = {
            'experiment_type': experiment_type,
            'arch_idx': arch_idx,
            'scaler_idx': scaler_idx,
            'optimizer_idx': optimizer_idx,
            'batch_idx': batch_idx,
            'activation_idx': activation_idx,
            'dropout_idx': dropout_idx
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)

    def load_checkpoint(self):
        """
        Load the last saved checkpoint if it exists
        """
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return None

    def load_existing_results(self, experiment_type):
        """
        Load existing results from previous runs
        """
        result_file = os.path.join(self.output_dir, f'{experiment_type.lower()}_grid_search_results.csv')
        if os.path.exists(result_file):
            return pd.read_csv(result_file)
        return pd.DataFrame()

    def analyze_predictions(self, y_true, y_pred, original_data, config_name, dataset_type):
        """
        Analyze predictions and save error information for all rows
        """
        errors = np.abs(y_true - y_pred)
        error_df = pd.DataFrame({
            'true_value': y_true,
            'predicted_value': y_pred,
            'absolute_error': errors,
            'relative_error_percent': np.abs(errors / y_true) * 100
        })

        # Add original features
        for col in original_data.columns:
            error_df[f'feature_{col}'] = original_data[col].values

        # Sort by absolute error in descending order
        error_df_sorted = error_df.sort_values('absolute_error', ascending=False)

        # Save complete error analysis
        analysis_file = os.path.join(
            self.error_analysis_dir,
            f'{config_name}_{dataset_type}_complete_error_analysis.csv'
        )
        error_df_sorted.to_csv(analysis_file, index=False)

        # Get top 10 worst predictions for backwards compatibility
        worst_predictions = error_df_sorted.head(10)
        worst_predictions_file = os.path.join(
            self.error_analysis_dir,
            f'{config_name}_{dataset_type}_worst_predictions.csv'
        )
        worst_predictions.to_csv(worst_predictions_file, index=False)

        return {
            'mean_error': errors.mean(),
            'median_error': np.median(errors),
            'max_error': errors.max(),
            'error_std': errors.std(),
            'worst_predictions_file': worst_predictions_file  # Maintain the original key
        }

    def save_model(self, model, scaler, config_name):
        """
        Save model and scaler for a specific configuration
        """
        model_dir = os.path.join(self.models_dir, config_name)
        os.makedirs(model_dir, exist_ok=True)

        # Save model in Keras native format
        model.save(os.path.join(model_dir, 'model.keras'))

        # Save scaler
        with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

    def plot_loss_curves(self, history, config_name):
        """
        Plot and save training and validation loss curves
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Model Loss: {config_name}')
        plt.ylabel('Loss (MAE)')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')

        # Create loss curves directory if it doesn't exist
        loss_curves_dir = os.path.join(self.output_dir, 'loss_curves')
        os.makedirs(loss_curves_dir, exist_ok=True)

        # Save the plot
        plot_filename = os.path.join(loss_curves_dir, f'{config_name}_loss_curve.png')
        plt.savefig(plot_filename)
        plt.close()

        # Print confirmation message
        print(f"Loss curve saved: {plot_filename}")

    def run_grid_search(self, data, architectures, experiment_type, resume=False):
        processed_data = self.preprocess_data(data)
        print(f"Features after preprocessing: {processed_data.columns.shape[0]}")

        # Load existing results if resuming
        results_df = self.load_existing_results(experiment_type)
        results = results_df.to_dict('records') if not results_df.empty else []

        # Setup data splits
        train_size = int(len(processed_data) * 0.7)
        val_size = int(len(processed_data) * 0.15)

        train_data = {
            'features': processed_data.iloc[:train_size].drop(['CPUTime'], axis=1),
            'target': processed_data.iloc[:train_size]['CPUTime']
        }
        val_data = {
            'features': processed_data.iloc[train_size:train_size+val_size].drop(['CPUTime'], axis=1),
            'target': processed_data.iloc[train_size:train_size+val_size]['CPUTime']
        }
        test_data = {
            'features': processed_data.iloc[train_size+val_size:].drop(['CPUTime'], axis=1),
            'target': processed_data.iloc[train_size+val_size:]['CPUTime']
        }

        # Load checkpoint if resuming
        checkpoint = self.load_checkpoint() if resume else None
        start_indices = {
            'arch_idx': 0, 'scaler_idx': 0, 'optimizer_idx': 0,
            'batch_idx': 0, 'activation_idx': 0, 'dropout_idx': 0
        }

        if checkpoint and checkpoint['experiment_type'] == experiment_type:
            start_indices = {
                'arch_idx': checkpoint['arch_idx'],
                'scaler_idx': checkpoint['scaler_idx'],
                'optimizer_idx': checkpoint['optimizer_idx'],
                'batch_idx': checkpoint['batch_idx'],
                'activation_idx': checkpoint['activation_idx'],
                'dropout_idx': checkpoint['dropout_idx']
            }

        for arch_idx in range(start_indices['arch_idx'], len(architectures)):
            arch = architectures[arch_idx]
            for scaler_idx in range(start_indices['scaler_idx'], len(self.scalers)):
                scaler_name = self.scalers[scaler_idx]
                for optimizer_idx in range(start_indices['optimizer_idx'], len(self.optimizers)):
                    optimizer = self.optimizers[optimizer_idx]
                    for batch_idx in range(start_indices['batch_idx'], len(self.batch_sizes)):
                        batch_size = self.batch_sizes[batch_idx]
                        for activation_idx in range(start_indices['activation_idx'], len(self.activations)):
                            activation = self.activations[activation_idx]
                            for dropout_idx in range(start_indices['dropout_idx'], len(self.dropout_rates)):
                                dropout_rate = self.dropout_rates[dropout_idx]

                                # Generate unique configuration name
                                config_name = f"{experiment_type}_arch{arch_idx}_scaler{scaler_name}_opt{optimizer}_batch{batch_size}_act{activation}_drop{dropout_rate}"

                                try:
                                    print(f"\nTrying configuration: {config_name}")

                                    scaler = self.get_scaler(scaler_name)
                                    X_train_scaled = scaler.fit_transform(train_data['features'])
                                    X_val_scaled = scaler.transform(val_data['features'])
                                    X_test_scaled = scaler.transform(test_data['features'])

                                    model = self.create_model(
                                        X_train_scaled.shape[1],
                                        arch,
                                        activation=activation,
                                        dropout_rate=dropout_rate
                                    )

                                    model.compile(
                                        optimizer=optimizer,
                                        loss='mae',
                                        metrics=['mae']
                                    )

                                    early_stopping = tf.keras.callbacks.EarlyStopping(
                                        monitor='val_loss',
                                        patience=5,
                                        restore_best_weights=True
                                    )

                                    history = model.fit(
                                        X_train_scaled, train_data['target'],
                                        validation_data=(X_val_scaled, val_data['target']),
                                        epochs=1500,
                                        batch_size=batch_size,
                                        verbose=0,
                                        callbacks=[early_stopping]
                                    )

                                    # Plot and save loss curves
                                    self.plot_loss_curves(history, config_name)

                                    # Get predictions for error analysis
                                    train_pred = model.predict(X_train_scaled)
                                    val_pred = model.predict(X_val_scaled)
                                    test_pred = model.predict(X_test_scaled)

                                    # Perform error analysis
                                    train_analysis = self.analyze_predictions(
                                        train_data['target'],
                                        train_pred.flatten(),
                                        train_data['features'],
                                        config_name,
                                        'train'
                                    )
                                    val_analysis = self.analyze_predictions(
                                        val_data['target'],
                                        val_pred.flatten(),
                                        val_data['features'],
                                        config_name,
                                        'val'
                                    )
                                    test_analysis = self.analyze_predictions(
                                        test_data['target'],
                                        test_pred.flatten(),
                                        test_data['features'],
                                        config_name,
                                        'test'
                                    )

                                    # Save model and scaler
                                    self.save_model(model, scaler, config_name)

                                    # Add results
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
                                        'train_max_error': train_analysis['max_error'],
                                        'val_max_error': val_analysis['max_error'],
                                        'test_max_error': test_analysis['max_error'],
                                        'train_error_std': train_analysis['error_std'],
                                        'val_error_std': val_analysis['error_std'],
                                        'test_error_std': test_analysis['error_std'],
                                        'epochs': len(history.history['loss'])
                                    }

                                    results.append(result)

                                    # Save results after each successful run
                                    pd.DataFrame(results).to_csv(
                                        os.path.join(self.output_dir, f'{experiment_type.lower()}_grid_search_results.csv'),
                                        index=False
                                    )

                                    print(f"Configuration completed. Test MAE: {test_analysis['mean_error']:.4f}")
                                    print(f"Error analysis saved in: {test_analysis['worst_predictions_file']}")

                                except Exception as e:
                                    print(f"Error with configuration: {str(e)}")
                                    continue

        return pd.DataFrame(results)

def parse_args():
    """
    Parse command line arguments
    """
    import argparse
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
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')

    return parser.parse_args()


def main():
    """
    Main function to run the hyperparameter search experiments
    """
    args = parse_args()

    # Initialize framework with specified output directory and checkpoint file
    framework = EnhancedPredictionFramework(
        output_dir=args.output_dir,
        checkpoint_file=args.checkpoint_file
    )

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

    architectures = [
        [64, 32],
        [128, 64],
        [256, 128, 64],
        [512, 256, 128]
    ]

    experiment_files = {
        '2DSA': './preprocess/results/filtered/2dsa-filtered.csv',
        'GA': './preprocess/results/filtered/ga-filtered.csv',
        'PCSA': './preprocess/results/filtered/pcsa-filtered.csv'
    }

    # Filter experiments if specified
    if args.experiments:
        experiment_files = {k: v for k, v in experiment_files.items() if k in args.experiments}

    # Check for existing checkpoint
    checkpoint = framework.load_checkpoint() if args.resume else None
    start_experiment = None if checkpoint is None else checkpoint['experiment_type']
    resume = checkpoint is not None

    for exp_type, file_path in experiment_files.items():
        # Skip experiments until we reach the checkpoint
        if resume and start_experiment and exp_type != start_experiment:
            continue

        print(f"\nProcessing {exp_type} experiments...")
        data = pd.read_csv(file_path)

        results = framework.run_grid_search(data, architectures, exp_type, resume=resume)
        results['experiment_type'] = exp_type

        # After first experiment, we're caught up to the checkpoint
        resume = False

    print("\nAll experiments completed successfully!")

if __name__ == "__main__":
    main()
