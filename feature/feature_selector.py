from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LassoFeatureSelector:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def perform_selection(self):
        """Perform LASSO-based feature selection."""
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.data_loader.encoded_features)

        # Fit LassoCV
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(X_scaled, self.data_loader.target)

        return self._create_importance_df(lasso.coef_)

    def _create_importance_df(self, coefficients):
        """Create DataFrame with feature importance scores."""
        return pd.DataFrame({
            'feature': self.data_loader.encoded_features.columns,
            'coefficient': np.abs(coefficients)
        }).sort_values('coefficient', ascending=False)

    def plot_coefficients(self, importance_df, output_path):
        """Plot LASSO coefficients."""
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(importance_df)), importance_df['coefficient'])
        plt.xticks(range(len(importance_df)),
                   importance_df['feature'],
                   rotation=45,
                   ha='right')
        plt.title('LASSO Feature Importance')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
