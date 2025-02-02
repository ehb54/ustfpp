import os
from datetime import datetime
import pandas as pd

# Import our refactored components
from data_loader import DataLoader
from qualitative_analyzer import QualitativeAnalyzer
from quantitative_analyzer import QuantitativeAnalyzer
from stability_analyzer import FeatureStabilityAnalyzer
from feature_selector import LassoFeatureSelector
from interaction_analyzer import FeatureInteractionAnalyzer

class FeatureAnalysisOrchestrator:
    def __init__(self, data_file):
        """Initialize the orchestrator with data file."""
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = os.path.join('results',
                                        f'feature_analysis_{self.timestamp}')
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize components
        self.data_loader = DataLoader(data_file)
        self.qual_analyzer = QualitativeAnalyzer(self.data_loader)
        self.quant_analyzer = QuantitativeAnalyzer(self.data_loader)
        self.stability_analyzer = FeatureStabilityAnalyzer(self.data_loader)
        self.lasso_selector = LassoFeatureSelector(self.data_loader)
        self.interaction_analyzer = FeatureInteractionAnalyzer(self.data_loader)

    def run_analysis(self, qual_threshold=0.1, quant_threshold=0.1):
        """Run complete feature analysis."""
        # Basic analysis
        qual_results = self.qual_analyzer.analyze_features()
        quant_results = self.quant_analyzer.analyze_features()

        # Advanced analysis
        stability_results = self.stability_analyzer.analyze_stability()
        lasso_results = self.lasso_selector.perform_selection()

        # Get top features for interaction analysis
        top_features = lasso_results.head(10)['feature'].tolist()
        interaction_results = self.interaction_analyzer.analyze_interactions(top_features)

        # Save results
        self._save_results(qual_results, quant_results,
                           stability_results, lasso_results,
                           interaction_results)

        return {
            'qualitative': qual_results,
            'quantitative': quant_results,
            'stability': stability_results,
            'lasso': lasso_results,
            'interactions': interaction_results
        }

    def _save_results(self, qual_results, quant_results,
                      stability_results, lasso_results,
                      interaction_results):
        """Save all analysis results."""
        # Save results to CSV files
        pd.DataFrame(qual_results).to_csv(
            os.path.join(self.results_dir, 'qualitative_features.csv'))
        pd.DataFrame(quant_results).to_csv(
            os.path.join(self.results_dir, 'quantitative_features.csv'))
        stability_results.to_csv(
            os.path.join(self.results_dir, 'feature_stability.csv'))
        lasso_results.to_csv(
            os.path.join(self.results_dir, 'lasso_importance.csv'))
        interaction_results.to_csv(
            os.path.join(self.results_dir, 'feature_interactions.csv'))

        # Generate plots
        self.stability_analyzer.plot_stability(
            stability_results,
            os.path.join(self.results_dir, 'feature_stability.png'))
        self.lasso_selector.plot_coefficients(
            lasso_results,
            os.path.join(self.results_dir, 'lasso_coefficients.png'))
