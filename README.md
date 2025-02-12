# **UltraScan TensorFlow Job Performance Prediction (USTFPP)**

**A machine learning-based approach to predict UltraScan job performance using TensorFlow.**  
This repository provides tools for preprocessing data, training models, and evaluating predictions to optimize UltraScan's job scheduling.

---

## **Table of Contents**
1. [Features](#features)
2. [Installation](#installation)
3. [Environment Setup](#environment-setup)
4. [Analysis Workflow](#analysis-workflow)
5. [Configuration](#configuration)
6. [Repository Structure](#repository-structure)
7. [Contributing](#contributing)

---

## **Features**
- **Performance Prediction** – Predicts job execution times for UltraScan using TensorFlow.
- **Hyperparameter Tuning** – Optimizes model architectures for better predictions.
- **Preprocessing Utilities** – Scripts for data cleaning, transformation, and splitting.
- **Monte Carlo Feature Analysis** – Identifies important features for prediction.
- **Configurable Execution** – Supports JSON-based configuration for flexibility.

---

## **Installation**
1. **Clone the Repository**
   ```bash
   git clone https://github.com/ehb54/ustfpp.git
   cd ustfpp
   ```

---

## **Environment Setup**
1. **Create and Activate Virtual Environment**
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate on Linux/Mac
   source venv/bin/activate

   # Activate on Windows
   venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   # Upgrade pip
   pip install --upgrade pip

   # Install requirements
   pip install -r requirements.txt
   ```

3. **Verify Installation**
   ```bash
   # Check TensorFlow installation
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

---

## **Analysis Workflow**

### 1. Data Formatting
Format raw data using the format_data.py script:
```bash
python format_data.py
```
This will process summary_metadata.csv files and create datasets for each experiment type (2DSA, GA, PCSA) in the results directory.

### 2. Data Preprocessing
1. **Review Preprocessing Configuration**
   - See `preprocess/preprocessor_filter.md` for detailed configuration options
   - Copy and modify the sample config:
     ```bash
     cp preprocessor_config.json myconfig.json
     ```

2. **Run Preprocessing**
   ```bash
   python preprocess/preprocess_filter.py -c myconfig.json
   ```
   This will create filtered datasets in the results/filtered directory.

### 3. Monte Carlo Feature Analysis
1. **Configure Analysis**
   Create a Monte Carlo configuration file:
   ```json
   {
       "data_file": "results/filtered/2dsa-dataset.csv",
       "n_iterations": 100,
       "output_dir": "results/monte_carlo",
       "excluded_columns": ["wallTime", "max_rss"]
   }
   ```

2. **Run Analysis**
   ```bash
   python feature/feature_mc_analysis.py --config monte_carlo_config.json
   ```

### 4. Hyperparameter Grid Search
1. **Configure Grid Search**
   Create a hyperparameter search configuration:
   ```json
   {
       "experiment_files": {
           "2DSA": "./results/filtered/2dsa-dataset-filtered-lasso.csv"
       },
       "architectures": [[64, 32], [128, 64]],
       "scalers": ["standard", "minmax"],
       "optimizers": ["adam", "rmsprop"],
       "batch_sizes": [32, 64],
       "activations": ["relu", "elu"],
       "dropout_rates": [0.2, 0.3]
   }
   ```

2. **Run Grid Search**
   ```bash
   python hyperparameter_architecture_analysis.py --config-file grid_config.json
   ```

### 5. Model Evaluation
Test the trained models using run_model.py:
```bash
python run_model.py --model results/prediction/best_model --data test_data.csv --output results/evaluation
```

### 6. Parallel Training with Split Config
The split_config.py utility allows running training jobs in parallel across multiple GPUs:

1. **Create Base Configuration**
   Create a base configuration file with all parameters to test.

2. **Split Configuration**
   ```bash
   python split_config.py --config-file base_config.json --gpus 0,1 --jobs-per-gpu 2
   ```
   This creates:
   - Individual configuration files for each combination
   - GPU-specific config lists
   - Run scripts for each GPU

3. **Execute Training**
   ```bash
   # Run training on GPU 0
   bash gpu_0.run

   # Run training on GPU 1
   bash gpu_1.run
   ```

---

## **Configuration**
- `preprocessor_config.json`: Controls data preprocessing
- `monte_carlo_config.json`: Configures feature analysis
- `grid_config.json`: Defines hyperparameter search space
- See `preprocess/preprocessor_filter.md` for detailed preprocessing configuration options

---

## **Repository Structure**
```
ustfpp
├── attic/                                    # Archived code and utilities
├── feature/                                  # Feature analysis utilities
│   └── feature_mc_analysis.py               # Monte Carlo feature analysis
├── preprocess/                               # Data preprocessing scripts
│   ├── format_data.py                        # Initial data formatting
│   ├── preprocess_filter.py                  # Data preprocessing and filtering
│   └── preprocessor_filter.md                # Preprocessing configuration guide
├── scripts/                                  # Utility scripts
├── visual/                                   # Visualization tools
├── hyperparameter_architecture_analysis.py   # Model architecture tuning
├── run_model.py                              # Model evaluation
├── split_config.py                           # Parallel training utility
├── config.json.sample                        # Sample configuration file
├── requirements.txt                          # Python dependencies
└── README.md                                 # Project documentation
```

---

## **Contributing**
Contributions are welcome! To contribute:
1. **Fork** the repository
2. **Create** a feature branch (`feature-branch`)
3. **Commit** your changes
4. **Open** a Pull Request

---

## **Contact**
For questions or collaboration, reach out via GitHub Issues.

---
