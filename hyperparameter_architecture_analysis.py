import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import os
import json

class EnhancedPredictionFramework:
    def __init__(self, output_dir='results/prediction'):
        self.output_dir = output_dir
        self.checkpoint_file = os.path.join(output_dir, 'checkpoint.json')
        os.makedirs(output_dir, exist_ok=True)

        self.scalers = ['standard', 'minmax', 'robust']
        self.optimizers = ['adam', 'rmsprop', 'sgd']
        self.batch_sizes = [16, 32, 64]
        self.activations = ['relu', 'elu']
        self.dropout_rates = [0.2, 0.3]

    def preprocess_data(self, data):
        df = data.copy()
        print("Available columns:", df.columns.tolist())

        # Drop non-feature columns
        df = df.drop(['max_rss', 'wallTime'], axis=1)

        # Convert cluster name to categorical
        df['job.cluster.@attributes.name'] = df['job.cluster.@attributes.name'].astype(str)
        df = pd.get_dummies(df, columns=['job.cluster.@attributes.name'], prefix=['cluster'])

        print("Final columns:", df.columns.tolist())
        return df

    def create_model(self, input_dim, architecture, activation='relu', dropout_rate=0.2):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))

        for units in architecture:
            model.add(tf.keras.layers.Dense(units, activation=activation))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Dense(1))
        return model

    def get_scaler(self, scaler_name):
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        return scalers.get(scaler_name)

    def save_checkpoint(self, experiment_type, arch_idx, scaler_idx, optimizer_idx, batch_idx, activation_idx, dropout_idx):
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
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return None

    def load_existing_results(self, experiment_type):
        result_file = os.path.join(self.output_dir, f'{experiment_type.lower()}_grid_search_results.csv')
        if os.path.exists(result_file):
            return pd.read_csv(result_file)
        return pd.DataFrame()

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

                                # Save checkpoint
                                self.save_checkpoint(
                                    experiment_type, arch_idx, scaler_idx,
                                    optimizer_idx, batch_idx, activation_idx, dropout_idx
                                )

                                # Check if this configuration has already been processed
                                config_signature = f"{arch}_{scaler_name}_{optimizer}_{batch_size}_{activation}_{dropout_rate}"
                                if results_df.empty is False and not results_df[
                                    (results_df['architecture'] == str(arch)) &
                                    (results_df['scaler'] == scaler_name) &
                                    (results_df['optimizer'] == optimizer) &
                                    (results_df['batch_size'] == batch_size) &
                                    (results_df['activation'] == activation) &
                                    (results_df['dropout_rate'] == dropout_rate)
                                ].empty:
                                    print(f"Skipping already processed configuration: {config_signature}")
                                    continue

                                try:
                                    print(f"\nTrying: {arch}, {scaler_name}, {optimizer}, {batch_size}")

                                    # Your existing training code here
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

                                    history = model.fit(
                                        X_train_scaled, train_data['target'],
                                        validation_data=(X_val_scaled, val_data['target']),
                                        epochs=50,
                                        batch_size=batch_size,
                                        verbose=0
                                    )

                                    train_metrics = model.evaluate(X_train_scaled, train_data['target'], verbose=0)
                                    val_metrics = model.evaluate(X_val_scaled, val_data['target'], verbose=0)
                                    test_metrics = model.evaluate(X_test_scaled, test_data['target'], verbose=0)

                                    result = {
                                        'architecture': str(arch),
                                        'scaler': scaler_name,
                                        'optimizer': optimizer,
                                        'batch_size': batch_size,
                                        'activation': activation,
                                        'dropout_rate': dropout_rate,
                                        'train_loss': train_metrics[0],
                                        'val_loss': val_metrics[0],
                                        'test_loss': test_metrics[0],
                                        'train_mae': train_metrics[1],
                                        'val_mae': val_metrics[1],
                                        'test_mae': test_metrics[1],
                                        'epochs': len(history.history['loss'])
                                    }

                                    results.append(result)

                                    # Save results after each successful run
                                    pd.DataFrame(results).to_csv(
                                        os.path.join(self.output_dir, f'{experiment_type.lower()}_grid_search_results.csv'),
                                        index=False
                                    )

                                    print(f"Test MAE: {test_metrics[1]:.4f}")

                                except Exception as e:
                                    print(f"Error with configuration: {str(e)}")
                                    continue

                            start_indices['dropout_idx'] = 0
                        start_indices['activation_idx'] = 0
                    start_indices['batch_idx'] = 0
                start_indices['optimizer_idx'] = 0
            start_indices['scaler_idx'] = 0

        # Remove checkpoint file after successful completion
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)

        return pd.DataFrame(results)

def main():
    framework = EnhancedPredictionFramework()

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

    # Check for existing checkpoint
    checkpoint = framework.load_checkpoint()
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
