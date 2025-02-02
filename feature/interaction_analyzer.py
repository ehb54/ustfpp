from itertools import combinations
from scipy import stats
import pandas as pd

class FeatureInteractionAnalyzer:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def analyze_interactions(self, top_features):
        """Analyze interactions between specified features."""
        interactions = []

        for f1, f2 in combinations(top_features, 2):
            interaction = (self.data_loader.encoded_features[f1] *
                           self.data_loader.encoded_features[f2])

            interaction_corr = stats.spearmanr(interaction,
                                               self.data_loader.target)[0]

            interactions.append({
                'feature1': f1,
                'feature2': f2,
                'interaction_score': abs(interaction_corr)
            })

        return pd.DataFrame(interactions).sort_values('interaction_score',
                                                      ascending=False)
