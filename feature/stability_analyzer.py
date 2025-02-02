import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

class FeatureStabilityAnalyzer:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def analyze_stability(self, n_iterations=10):
        """Analyze feature importance stability using random forest."""
        importance_scores = []

        for i in range(n_iterations):
            rf = RandomForestRegressor(n_estimators=100, random_state=i)
            rf.fit(self.data_loader.encoded_features,
                   self.data_loader.target)
            importance_scores.append(rf.feature_importances_)

        return self._create_stability_stats(importance_scores)

    def _create_stability_stats(self, importance_scores):
        """Calculate stability statistics."""
        return pd.DataFrame({
            'feature': self.data_loader.encoded_features.columns,
            'mean_importance': np.mean(importance_scores, axis=0),
            'std_importance': np.std(importance_scores, axis=0)
        }).sort_values('mean_importance', ascending=False)

    def plot_stability(self, stats, output_path):
        """Plot feature importance stability."""
        plt.figure(figsize=(12, 6))
        plt.errorbar(range(len(stats)),
                     stats['mean_importance'],
                     yerr=stats['std_importance'],
                     fmt='o')
        plt.xticks(range(len(stats)),
                   stats['feature'],
                   rotation=45,
                   ha='right')
        plt.title('Feature Importance Stability')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
