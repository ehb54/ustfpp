# Comprehensive Guide to Data Preprocessing Configuration

## File Structure Breakdown

### 1. Input Files Section

```json
"input_files": {
    "source_directory": "./results",
    "file_patterns": [
    "2dsa-dataset.csv",
    "ga-dataset.csv",
    "pcsa-dataset.csv"
  ],
"file_type": "csv"
}
```

- **Purpose**: Defines the source of input files
- `source_directory`: Where the input files are located
- `file_patterns`: List of specific CSV files to process
- `file_type`: Specifies the file type (in this case, CSV)

### 2. Output Files Section

```json
"output_files": {
    "directory": "./results/filtered",
    "prefix": "filtered_",
    "suffix": ".csv",
    "metrics_file": "preprocessing_metrics.csv"
}
```

- **Purpose**: Configures how output files will be generated
- `directory`: Where processed files will be saved
- `prefix`: Prefix added to output filenames
- `suffix`: File extension for output files
- `metrics_file`: File to store preprocessing metrics

### 3. Column Configuration Section

```json
"column_configuration": {
    "keep_columns": [...],
    "drop_columns": [...],
    "transformations": {
      "rename_columns": {
      ...
      }
    }
  }
```

- **Purpose**: Controls column selection and transformations
- `keep_columns`: Columns to retain in the processed dataset
- `drop_columns`: Columns to remove from the dataset
- `transformations.rename_columns`: Mapping to rename specific columns

### 4. Negative Value Handling

```json
"negative_value_handling": {
  "mode": "replace",
  "replace_value": 0,
  "columns": [...]
}
```

**Mode Options**:

- `"disabled"`: No action on negative values
- `"all_columns"`: Remove rows with negative values in ANY column
- `"specific_columns"`: Remove rows with negative values in SPECIFIED columns
- `"replace"`: Replace negative values with a specified value (in this case, 0)

### 5. Additional Filters

```json
"additional_filters": {
"min_value_filters": {...},
"max_value_filters": {...
}
}
```

- **Purpose**: Apply minimum and maximum value constraints to specific columns
- Helps remove outliers or invalid data points
- Each filter specifies the minimum or maximum allowed value for a column

### 6. Logging Configuration

```json
"logging": {
  "log_level": "INFO",
  "log_file": "2dsa_preprocessing.log"
}
```

- **Purpose**: Configure logging behavior
- `log_level`: Determines verbosity of log messages
    - `DEBUG`: Most detailed
    - `INFO`: Standard informational messages
    - `WARNING`: Only warnings and errors
    - `ERROR`: Only error messages
    - `CRITICAL`: Only critical errors

### 7. Uniform Column Handling

```json
"uniform_columns": {
"remove": true,
"threshold": 1
}
```

- **Purpose**: Control handling of columns with uniform/constant values
- `remove`: Boolean to enable/disable removal of uniform columns
- `threshold`: Number of unique values to consider a column non-uniform (default is 1)

- **Purpose**: Provides context and version information about the configuration


## Potential Configurations

### 1. Strict Filtering

```json
    "negative_value_handling": {
    "mode": "specific_columns",
    "columns": ["cpu_time", "memory_usage"]
    },
    "additional_filters": {
    "min_value_filters": {"cpu_time": 0.1},
    "max_value_filters": {"memory_usage": 1024}
    }
```

### 2. Lenient Preprocessing

```json
    "negative_value_handling": {
        "mode": "replace",
        "replace_value": 0,
        "columns": ["all"]
    },
    "additional_filters": {
    "min_value_filters": {},
    "max_value_filters": {
    }
}
```

## Troubleshooting

- If preprocessing fails, check:
    1. Input file paths
    2. Column names
    3. Value constraints
    4. Logging output
- Use verbose logging (`DEBUG` level) for detailed diagnostics
