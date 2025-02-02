import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, Any, List

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

class FeatureAnalysisSystem:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = {
            'qual_threshold': 0.1,
            'quant_threshold': 0.1,
            'n_stability_iterations': 10,
            'top_n_interactions': 10,
            'output_format': 'both',
            'results_dir': None
        }
        if config:
            self.config.update(config)

        self.orchestrators = {}
        self.results = {}
        self._setup_results_directory()

    def _setup_results_directory(self):
        if not self.config['results_dir']:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.config['results_dir'] = os.path.join('results', f'feature_analysis_{timestamp}')
        os.makedirs(self.config['results_dir'], exist_ok=True)

    def _save_json_results(self, method: str):
        output_path = os.path.join(self.config['results_dir'], f'{method}_analysis_results.json')
        with open(output_path, 'w') as f:
            json.dump(self.results[method], f, indent=2, cls=NumpyJSONEncoder)

    def _save_csv_results(self, method: str):
        for key, df in self.results[method].items():
            if isinstance(df, pd.DataFrame):
                output_path = os.path.join(self.config['results_dir'], f'{method}_{key}_results.csv')
                df.to_csv(output_path, index=False)

    def _save_results(self, method: str):
        if self.config['output_format'] in ['csv', 'both']:
            self._save_csv_results(method)
        if self.config['output_format'] in ['json', 'both']:
            self._save_json_results(method)

    def run_analysis(self, data_files: Dict[str, str]):
        from analysis_orchestrator import FeatureAnalysisOrchestrator

        for method, file_path in data_files.items():
            print(f"\nAnalyzing {method} data from: {file_path}")

            try:
                # Initialize orchestrator for each method
                self.orchestrators[method] = FeatureAnalysisOrchestrator(file_path)

                # Run analysis
                self.results[method] = self.orchestrators[method].run_analysis(
                    qual_threshold=self.config['qual_threshold'],
                    quant_threshold=self.config['quant_threshold']
                )

                # Save method-specific results
                self._save_results(method)

                print(f"Completed analysis for {method}")

            except Exception as e:
                print(f"Error analyzing {method} data: {str(e)}")
                continue

        return self.results

    def get_summary_report(self) -> str:
        if not self.results:
            return "No analysis results available."

        report = ["=== Feature Analysis Summary Report ===\n"]

        for method in self.results.keys():
            report.append(f"\n{method} Analysis:")

            if 'qualitative' in self.results[method]:
                report.append("\nTop Qualitative Features:")
                for feature, results in self.results[method]['qualitative'].items():
                    report.append(f"- {feature}: MI={results['mutual_information']:.4f}")

            if 'quantitative' in self.results[method]:
                report.append("\nTop Quantitative Features:")
                correlations = self.results[method]['quantitative']['correlations']
                for feature, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
                    report.append(f"- {feature}: correlation={corr:.4f}")

        return "\n".join(report)

def main():
    config = {
        'qual_threshold': 0.15,
        'quant_threshold': 0.15,
        'n_stability_iterations': 20,
        'top_n_interactions': 15,
        'output_format': 'both',
        'results_dir': 'results/feature_analysis'
    }

    # Initialize system
    analysis_system = FeatureAnalysisSystem(config)

    # Define data files for each method
    data_files = {
        '2DSA': '../preprocess/results/filtered/2dsa-filtered.csv',
        'GA': '../preprocess/results/filtered/ga-filtered.csv',
        'PCSA': '../preprocess/results//filtered/pcsa-filtered.csv'
    }

    try:
        # Run analysis for all methods
        results = analysis_system.run_analysis(data_files)

        # Print summary report
        print(analysis_system.get_summary_report())
        print(f"\nResults saved in: {analysis_system.config['results_dir']}")

    except Exception as e:
        print(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()
