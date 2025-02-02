from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from scipy import stats
import numpy as np

class QuantitativeAnalyzer:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def analyze_features(self):
        """Analyze quantitative features using correlation, mutual information, and f-scores."""
        try:
            print("\nAnalyzing quantitative features...")

            # Get quantitative feature data
            quant_features = self.data_loader.get_quantitative_features()
            if not quant_features:
                raise ValueError("No quantitative features found in the dataset")

            X_quant = self.data_loader.encoded_features[quant_features]

            return {
                'correlations': self._calculate_correlations(X_quant),
                'mutual_information': self._calculate_mutual_information(X_quant),
                'f_scores': self._calculate_f_scores(X_quant)
            }
        except Exception as e:
            print(f"Error in quantitative analysis: {str(e)}")
            raise

    def _calculate_correlations(self, X_quant):
        """Calculate correlations with target."""
        try:
            return {
                feature: abs(stats.pearsonr(X_quant[feature], self.data_loader.target)[0])
                for feature in X_quant.columns
            }
        except Exception as e:
            print(f"Error calculating correlations: {str(e)}")
            raise

    def _calculate_mutual_information(self, X_quant):
        """Calculate mutual information scores."""
        try:
            # Convert DataFrame to numpy array for mutual_info_regression
            X_array = X_quant.values
            mi_scores = mutual_info_regression(X_array, self.data_loader.target, random_state=42)
            return dict(zip(X_quant.columns, mi_scores))
        except Exception as e:
            print(f"Error calculating mutual information: {str(e)}")
            raise

    def _calculate_f_scores(self, X_quant):
        """Calculate f-regression scores."""
        try:
            f_selector = SelectKBest(score_func=f_regression, k='all')
            f_selector.fit(X_quant, self.data_loader.target)
            return dict(zip(X_quant.columns, f_selector.scores_))
        except Exception as e:
            print(f"Error calculating f-scores: {str(e)}")
            raise
