import pandas as pd
import numpy as np
import os
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

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
    DEFAULT_CONFIG = {
        'qual_threshold': 0.1,
        'quant_threshold': 0.1,
        'n_stability_iterations': 10,
        'top_n_interactions': 10,
        'output_format': 'both',
        'results_dir': None
    }

    def __init__(self, config: Dict[str, Any] = None):
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        self.orchestrators = {}
        self.results = {}
        self._setup_results_directory()

    @classmethod
    def from_json(cls, json_path: str) -> 'FeatureAnalysisSystem':
        """Create FeatureAnalysisSystem instance from JSON config file."""
        with open(json_path, 'r') as f:
            config = json.load(f)
        return cls(config)

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
                self.orchestrators[method] = FeatureAnalysisOrchestrator(file_path)
                self.results[method] = self.orchestrators[method].run_analysis(
                    qual_threshold=self.config['qual_threshold'],
                    quant_threshold=self.config['quant_threshold']
                )
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

def parse_args():
    parser = argparse.ArgumentParser(
        description='Feature Analysis System - Analyzes features from multiple data sources',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with config file:
    python analyze_features.py --config config.json

    # Run with specific data directory and output format:
    python analyze_features.py --data-dir ./data --output-format json

    # Override config file settings:
    python analyze_features.py --config config.json --qual-threshold 0.15 --results-dir ./results
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to JSON config file containing analysis parameters'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Directory containing input CSV files (default: ../preprocess/results/filtered/)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        help='Directory for output results (default: results/feature_analysis_<timestamp>)'
    )
    parser.add_argument(
        '--qual-threshold',
        type=float,
        help='Threshold for qualitative feature analysis (default: 0.1)'
    )
    parser.add_argument(
        '--quant-threshold',
        type=float,
        help='Threshold for quantitative feature analysis (default: 0.1)'
    )
    parser.add_argument(
        '--output-format',
        choices=['json', 'csv', 'both'],
        help='Output format for results (default: both)'
    )

    return parser.parse_args()

def load_config(args) -> Dict[str, Any]:
    """Load and merge configuration from JSON file and command line arguments."""
    config = {}

    # Load from JSON file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)

    # Override with command line arguments if provided
    if args.results_dir:
        config['results_dir'] = args.results_dir
    if args.qual_threshold:
        config['qual_threshold'] = args.qual_threshold
    if args.quant_threshold:
        config['quant_threshold'] = args.quant_threshold
    if args.output_format:
        config['output_format'] = args.output_format

    return config

def get_data_files(data_dir: str) -> Dict[str, str]:
    """Get data files from the specified directory."""
    if not data_dir:
        return {
            '2DSA': '../preprocess/results/filtered/2dsa-filtered.csv',
            'GA': '../preprocess/results/filtered/ga-filtered.csv',
            'PCSA': '../preprocess/results/filtered/pcsa-filtered.csv'
        }

    data_files = {}
    data_dir_path = Path(data_dir)

    if not data_dir_path.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    for file_path in data_dir_path.glob('*.csv'):
        method = file_path.stem.split('-')[0].upper()
        data_files[method] = str(file_path)

    return data_files

def main():
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config = load_config(args)

    try:
        # Initialize system
        analysis_system = FeatureAnalysisSystem(config)

        # Get data files
        data_files = get_data_files(args.data_dir)

        if not data_files:
            raise ValueError("No data files found for analysis")

        # Run analysis
        results = analysis_system.run_analysis(data_files)

        # Print summary report
        print(analysis_system.get_summary_report())
        print(f"\nResults saved in: {analysis_system.config['results_dir']}")

    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
