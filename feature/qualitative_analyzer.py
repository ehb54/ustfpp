import numpy as np
from sklearn.feature_selection import mutual_info_regression

class QualitativeAnalyzer:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def analyze_features(self):
        """Analyze qualitative features using mutual information and entropy."""
        analysis_results = {}

        for feature in self.data_loader.get_qualitative_features():
            if feature in self.data_loader.features.columns:
                analysis_results[feature] = self._analyze_single_feature(feature)

        return analysis_results

    def _analyze_single_feature(self, feature):
        """Analyze a single qualitative feature."""
        # Calculate entropy
        value_counts = self.data_loader.features[feature].value_counts(normalize=True)
        entropy = -np.sum(value_counts * np.log2(value_counts))

        # Calculate mutual information with target
        encoded_feature = self.data_loader.encoded_features[feature].values.reshape(-1, 1)
        mi_score = mutual_info_regression(encoded_feature,
                                          self.data_loader.target,
                                          random_state=42)[0]

        return {
            'mutual_information': mi_score,
            'entropy': entropy,
            'unique_values': len(value_counts)
        }
