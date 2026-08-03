#!/usr/bin/env python3
"""
run_grid_search.py - Directory-based grid search with column file validation

Reset environment on server

module purge
module load python/3.11.11_tensorflow
# make sure you are NOT picking up /usr/local/cuda toolchain
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/usr/local/cuda' | paste -sd:)
unset LD_LIBRARY_PATH   # if you can; or at least remove any cuda paths
# sanity checks
python -c "import tensorflow as tf; print(tf.__version__); print(tf.sysconfig.get_build_info())"
which ptxas || echo "no ptxas in PATH (good)"

"""

import json
import itertools
import os
import sys
import subprocess
import multiprocessing as mp
from queue import Empty
import time
from datetime import datetime
from pathlib import Path
import glob
import csv


def format_duration(seconds):
    """Format seconds as HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_column_file(columns_file):
    """
    Load column names from newline-delimited text file.
    Returns: list of column names, preserving order
    """
    columns = []
    with open(columns_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                columns.append(line)
    return columns


def get_csv_columns(csv_file):
    """Get column names from CSV header"""
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        return header


def validate_config(config_path, config, dry_run=False):
    """
    Validate a single config file.
    Returns: (valid, error_message, validation_info)
    """
    validation_info = {
        'config_file': os.path.basename(config_path),
        'method': None,
        'feature_space': None,
        'data_file': None,
        'columns_file': None,
        'features_requested': 0,
        'features_found': 0,
        'data_file_exists': False,
        'columns_file_exists': False,
        'validation_ok': False,
        'error_message': None
    }

    # Check required fields
    required_fields = ['method', 'feature_space', 'columns_file', 'data_file']
    missing = [f for f in required_fields if f not in config]
    if missing:
        error = f"Missing required fields: {', '.join(missing)}"
        validation_info['error_message'] = error
        return False, error, validation_info

    # Extract info
    method = config['method']
    feature_space = config['feature_space']
    columns_file = config['columns_file']
    data_file = config['data_file']

    validation_info['method'] = method
    validation_info['feature_space'] = feature_space
    validation_info['data_file'] = data_file
    validation_info['columns_file'] = columns_file

    # Validate method
    valid_methods = ['2dsa', 'ga', 'pcsa']
    if method not in valid_methods:
        error = f"Invalid method '{method}', must be one of: {', '.join(valid_methods)}"
        validation_info['error_message'] = error
        return False, error, validation_info

    # Validate feature_space
    valid_spaces = ['raw']
    if feature_space not in valid_spaces:
        error = f"Invalid feature_space '{feature_space}', must be one of: {', '.join(valid_spaces)}"
        validation_info['error_message'] = error
        return False, error, validation_info

    # Check data file exists
    if not os.path.exists(data_file):
        error = f"Data file not found: {data_file}"
        validation_info['error_message'] = error
        return False, error, validation_info
    validation_info['data_file_exists'] = True

    # Check columns file exists
    if not os.path.exists(columns_file):
        error = f"Columns file not found: {columns_file}"
        validation_info['error_message'] = error
        return False, error, validation_info
    validation_info['columns_file_exists'] = True

    # Load requested columns
    try:
        requested_columns = load_column_file(columns_file)
    except Exception as e:
        error = f"Failed to parse columns file: {e}"
        validation_info['error_message'] = error
        return False, error, validation_info

    validation_info['features_requested'] = len(requested_columns)

    if len(requested_columns) == 0:
        error = "Columns file contains no valid column names"
        validation_info['error_message'] = error
        return False, error, validation_info

    # Check CSV header contains all requested columns
    try:
        csv_columns = get_csv_columns(data_file)
    except Exception as e:
        error = f"Failed to read CSV header: {e}"
        validation_info['error_message'] = error
        return False, error, validation_info

    csv_columns_set = set(csv_columns)
    missing_columns = [col for col in requested_columns if col not in csv_columns_set]

    if missing_columns:
        error = f"Columns missing from CSV: {', '.join(missing_columns[:5])}"
        if len(missing_columns) > 5:
            error += f" (and {len(missing_columns) - 5} more)"
        validation_info['error_message'] = error
        return False, error, validation_info

    validation_info['features_found'] = len(requested_columns)
    validation_info['validation_ok'] = True

    return True, None, validation_info


def run_single_job(config_dict, gpu_id, resume=False):
    """
    Run a single training job on specified GPU
    Returns: (config_dict, success, stdout, stderr, duration)
    """
    start_time = time.time()

    # Set environment to use specific GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Create temporary config file for this job
    pid = os.getpid()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_file = f"temp_config_gpu{gpu_id}_pid{pid}_{timestamp}.json"

    try:
        # Write config
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=4)

        # Build command
        cmd = [sys.executable, 'grid_search_worker.py', '--config-file', config_file]

        # Add --resume flag if enabled
        if resume:
            cmd.append('--resume')

        # Print command for debugging
        cmd_str = ' '.join(cmd)
        print(f"    [GPU {gpu_id}] Executing: {cmd_str}")

        # Run the training script
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per job
        )

        duration = time.time() - start_time
        success = (result.returncode == 0)

        return (config_dict, success, result.stdout, result.stderr, duration)

    except subprocess.TimeoutExpired:
        return (config_dict, False, "", "Timeout", time.time() - start_time)
    except Exception as e:
        return (config_dict, False, "", str(e), time.time() - start_time)
    finally:
        # Cleanup temp config file
        if os.path.exists(config_file):
            os.remove(config_file)


def gpu_worker(job_queue, results_queue, gpu_id, worker_id, resume=False):
    """
    Worker process for a specific GPU
    Continuously pulls jobs from queue and executes them
    """
    job_count = 0

    while True:
        try:
            job = job_queue.get(timeout=1)

            if job is None:
                break

            job_count += 1  # ADD THIS LINE
            print(f"[GPU {gpu_id}] job {job_count}", flush=True)

            result = run_single_job(job, gpu_id, resume=resume)
            results_queue.put(result)
            job_queue.task_done()

        except Empty:
            continue
        except KeyboardInterrupt:
            break
        except Exception as e:
            continue


def generate_config_name(method, arch, scaler, opt, batch, act, drop):
    """
    Generate stable, unique configuration name matching worker's naming convention.
    This must EXACTLY match worker's generate_config_name() for resume to work.

    Worker format: "{EXPERIMENT_TYPE}_{scaler}_{optimizer}_{batch_size}_{activation}_{dropout_rate}_{arch_str}"
    Example: "2DSA_standard_rmsprop_64_relu_0.2_64_32"
    """
    arch_str = '_'.join(map(str, arch))
    # Match worker's format: uppercase method, no "arch_" prefix, direct concatenation
    return f"{method.upper()}_{scaler}_{opt}_{batch}_{act}_{drop}_{arch_str}"


def is_config_completed(output_dir, config_params):
    """
    Check if a configuration has been completed by validating checkpoint.json.

    A config is completed if and only if:
    1. checkpoint.json exists in output_dir
    2. checkpoint.json is valid JSON
    3. No parameter validation needed - presence of checkpoint.json is sufficient

    Args:
        output_dir: Path to configuration's output directory
        config_params: Dict with hyperparameters (not used but kept for clarity)

    Returns:
        bool: True if completed, False otherwise
    """
    checkpoint_file = os.path.join(output_dir, 'checkpoint.json')

    # Checkpoint file must exist
    if not os.path.exists(checkpoint_file):
        return False

    # Checkpoint file must be valid JSON
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        # If we can parse it, consider it valid
        return True
    except (json.JSONDecodeError, IOError):
        return False


def generate_configs(base_config, base_output_dir, gpu_id):
    """
    Generate all hyperparameter configurations.

    CRITICAL: Output directory names must match what worker expects.
    Worker does NOT use config_name for directories - it uses the output_dir from config.

    The worker will:
    1. Receive output_dir from this config
    2. Create that directory
    3. Generate its own config_name for checkpoint.json contents
    4. Save checkpoint in output_dir

    So we just need to create unique directories - the naming doesn't have to match config_name.
    """
    configs = []

    # Get required fields from base config
    method = base_config.get('method', '').lower()
    data_file = base_config.get('data_file', '')
    feature_space = base_config.get('feature_space', '')
    columns_file = base_config.get('columns_file', '')

    config_num = 0
    for params in itertools.product(
            base_config['architectures'],
            base_config['scalers'],
            base_config['optimizers'],
            base_config['batch_sizes'],
            base_config['activations'],
            base_config['dropout_rates']
    ):
        arch, scaler, opt, batch, act, drop = params
        config_num += 1

        # Create unique directory name - use simple numbering or original scheme
        # This directory name is just for organization - checkpoint validation is what matters
        dir_name = (
            f"{method}_"
            f"arch_{'_'.join(map(str, arch))}_"
            f"scaler_{scaler}_"
            f"opt_{opt}_"
            f"batch_{batch}_"
            f"act_{act}_"
            f"drop_{drop}"
        )

        output_dir = os.path.join(base_output_dir, dir_name)

        # Build job config
        job_config = {
            # Required fields
            'method': method,
            'feature_space': feature_space,
            'data_file': data_file,
            'columns_file': columns_file,

            # Hyperparameters - single values wrapped in lists for worker compatibility
            'architectures': [arch],
            'scalers': [scaler],
            'optimizers': [opt],
            'batch_sizes': [batch],
            'activations': [act],
            'dropout_rates': [drop],

            # Training configuration
            'epochs': base_config['epochs'],
            'patience': base_config['patience'],
            'training_fraction': base_config['training_fraction'],
            'validation_fraction': base_config['validation_fraction'],

            # System configuration
            'mirroredstrategy': base_config.get('mirroredstrategy', False),
            'gpu_mem_grow': base_config.get('gpu_mem_grow', True),

            # Split strategy
            'temporal_split': base_config.get('temporal_split', False),
            'time_column': base_config.get('time_column', 'submitTime'),

            # Target configuration
            'target_variable': base_config.get('target_variable', 'CPUTime'),

            # Output
            'output_dir': output_dir
        }

        # Handle multi-output if specified
        if 'target_variables' in base_config:
            job_config['target_variables'] = base_config['target_variables']
            if 'target_variable' in job_config:
                del job_config['target_variable']

        configs.append(job_config)

    return configs


def filter_completed_configs(all_configs, resume):
    """
    Filter out completed configurations based on checkpoint.json validation.

    Args:
        all_configs: List of configuration dictionaries
        resume: Boolean flag indicating whether resume mode is enabled

    Returns:
        tuple: (remaining_configs, completed_count, skipped_count)
    """
    if not resume:
        return all_configs, 0, 0

    print("\n" + "=" * 80)
    print("RESUME MODE: Validating checkpoints")
    print("=" * 80)

    remaining = []
    completed = 0

    for config in all_configs:
        output_dir = config['output_dir']
        config_params = {
            'method': config['method'],
            'architecture': config['architectures'][0],
            'scaler': config['scalers'][0],
            'optimizer': config['optimizers'][0],
            'batch_size': config['batch_sizes'][0],
            'activation': config['activations'][0],
            'dropout_rate': config['dropout_rates'][0]
        }

        if is_config_completed(output_dir, config_params):
            completed += 1
        else:
            remaining.append(config)

    print(f"Total configurations: {len(all_configs)}")
    print(f"Completed (valid checkpoint.json): {completed}")
    print(f"Remaining work: {len(remaining)}")
    print("=" * 80 + "\n")

    return remaining, completed, len(all_configs) - len(remaining)


def run_config(config_path, gpus, jobs_per_gpu, run_start_time, config_index, total_configs, log_entries,
               dry_run=False, resume=False):
    """
    Run a single config file through grid search.
    Returns: (status, duration, error_message)
    """
    config_basename = os.path.basename(config_path)
    remaining = total_configs - config_index

    # Load config
    try:
        with open(config_path, 'r') as f:
            lines = ''.join([line for line in f if not line.strip().startswith('#')])
            base_config = json.loads(lines)
    except Exception as e:
        elapsed = format_duration(time.time() - run_start_time)
        print(
            f"[{config_index}/{total_configs}] SKIP {config_basename}: Failed to load JSON - {e} (elapsed={elapsed}, remaining={remaining})")

        log_entry = {
            'config_file': config_basename,
            'method': None,
            'feature_space': None,
            'data_file': None,
            'columns_file': None,
            'features_requested': 0,
            'features_found': 0,
            'validation_ok': False,
            'run_status': 'json_error',
            'start_time': None,
            'end_time': None,
            'duration_sec': 0,
            'error_message': f"JSON parse error: {e}"
        }
        log_entries.append(log_entry)
        return 'json_error', 0, str(e)

    # Validate config
    valid, error_msg, validation_info = validate_config(config_path, base_config, dry_run=dry_run)

    elapsed = format_duration(time.time() - run_start_time)

    if not valid:
        print(
            f"[{config_index}/{total_configs}] SKIP {config_basename}: {error_msg} (elapsed={elapsed}, remaining={remaining})")

        log_entry = validation_info.copy()
        log_entry.update({
            'run_status': 'validation_failed',
            'start_time': None,
            'end_time': None,
            'duration_sec': 0
        })
        log_entries.append(log_entry)
        return 'validation_failed', 0, error_msg

    # Dry-run mode: just report validation success
    if dry_run:
        print(f"[{config_index}/{total_configs}] OK   {config_basename}: "
              f"data={'ok' if validation_info['data_file_exists'] else 'missing'}, "
              f"columns_file={'ok' if validation_info['columns_file_exists'] else 'missing'}, "
              f"features={validation_info['features_found']} "
              f"(elapsed={elapsed}, remaining={remaining})")

        log_entry = validation_info.copy()
        log_entry.update({
            'run_status': 'dry_run_ok',
            'start_time': None,
            'end_time': None,
            'duration_sec': 0
        })
        log_entries.append(log_entry)
        return 'dry_run_ok', 0, None

    # Print run banner
    method = validation_info['method']
    feature_space = validation_info['feature_space']
    data_file = validation_info['data_file']
    columns_file = validation_info['columns_file']
    n_features = validation_info['features_found']

    print(f"[{config_index}/{total_configs}] START {config_basename} "
          f"(method={method}, feature_space={feature_space}) "
          f"(elapsed={elapsed}, remaining={remaining})")
    print(f"    Dataset: {data_file}")
    print(f"    Columns: {columns_file}")
    print(f"    Features: {n_features}")
    print(f"    Resume mode: {resume}")

    # Get base output directory from config or use default
    base_output_dir = base_config.get('output_dir', './results')

    # Generate grid search jobs
    all_configs = generate_configs(base_config, base_output_dir, gpu_id=0)

    # CRITICAL FIX: Filter completed configs based on checkpoint validation
    configs_to_run, completed_count, skipped_count = filter_completed_configs(all_configs, resume)

    total_jobs = len(all_configs)

    # Early exit if all jobs complete
    if resume and len(configs_to_run) == 0:
        elapsed = format_duration(time.time() - run_start_time)
        print(f"[{config_index}/{total_configs}] SKIP {config_basename}: All jobs complete")
        print(f"    Total jobs: {total_jobs}, All completed")
        print(f"    (elapsed={elapsed}, remaining={remaining})")

        log_entry = validation_info.copy()
        log_entry.update({
            'run_status': 'all_complete',
            'start_time': None,
            'end_time': None,
            'duration_sec': 0,
            'jobs_total': total_jobs,
            'jobs_successful': completed_count,
            'jobs_failed': 0,
            'error_message': 'All jobs already completed'
        })
        log_entries.append(log_entry)
        return 'all_complete', 0, None

    print(
        f"    Grid search: {total_jobs} total jobs, {completed_count} completed, {len(configs_to_run)} to run")
    print(
        f"    Workers: {len(gpus)} GPUs × {jobs_per_gpu} workers = {len(gpus) * jobs_per_gpu} parallel processes")

    # Run grid search
    start_time = time.time()
    start_timestamp = datetime.now().isoformat()

    try:
        # Create multiprocessing queues
        manager = mp.Manager()
        job_queue = manager.Queue()
        results_queue = manager.Queue()

        # Fill job queue with ONLY remaining configs
        for config in configs_to_run:
            job_queue.put(config)

        # Add poison pills
        num_workers = len(gpus) * jobs_per_gpu
        for _ in range(num_workers):
            job_queue.put(None)

        # Start worker processes
        processes = []
        for gpu_id in gpus:
            for worker_num in range(jobs_per_gpu):
                p = mp.Process(target=gpu_worker, args=(job_queue, results_queue, gpu_id, worker_num, resume))
                p.start()
                processes.append(p)

        # Wait for completion
        for p in processes:
            p.join()

        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        # Check success
        successful = sum(1 for r in results if r[1])
        failed = len(results) - successful

        # Total successful includes previously completed
        total_successful = completed_count + successful

        duration = time.time() - start_time
        duration_str = format_duration(duration)
        elapsed = format_duration(time.time() - run_start_time)
        end_timestamp = datetime.now().isoformat()

        if failed == 0:
            status = 'success'
            print(f"[{config_index}/{total_configs}] DONE {config_basename}: "
                  f"status=success, duration={duration_str}, elapsed={elapsed}, remaining={remaining}")
            print(f"    Completed: {total_successful}/{total_jobs} (resumed: {completed_count}, trained: {successful})")
            error_message = None
        else:
            status = 'partial_failure'
            print(f"[{config_index}/{total_configs}] DONE {config_basename}: "
                  f"status=partial_failure, duration={duration_str}, elapsed={elapsed}, remaining={remaining}")
            print(
                f"    Completed: {total_successful}/{total_jobs} (resumed: {completed_count}, trained: {successful}, failed: {failed})")
            error_message = f"{failed}/{len(configs_to_run)} jobs failed in this run"

        log_entry = validation_info.copy()
        log_entry.update({
            'run_status': status,
            'start_time': start_timestamp,
            'end_time': end_timestamp,
            'duration_sec': duration,
            'error_message': error_message,
            'jobs_total': total_jobs,
            'jobs_successful': total_successful,
            'jobs_failed': failed,
            'jobs_resumed': completed_count
        })
        log_entries.append(log_entry)

        return status, duration, error_message

    except Exception as e:
        duration = time.time() - start_time
        duration_str = format_duration(duration)
        elapsed = format_duration(time.time() - run_start_time)
        end_timestamp = datetime.now().isoformat()

        print(f"[{config_index}/{total_configs}] DONE {config_basename}: "
              f"status=failed, duration={duration_str}, elapsed={elapsed}, remaining={remaining}")
        print(f"    Error: {e}")

        log_entry = validation_info.copy()
        log_entry.update({
            'run_status': 'failed',
            'start_time': start_timestamp,
            'end_time': end_timestamp,
            'duration_sec': duration,
            'error_message': str(e)
        })
        log_entries.append(log_entry)

        return 'failed', duration, str(e)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run grid search across multiple configs and GPUs')
    parser.add_argument('--config-dir', help='Directory containing config JSON files')
    parser.add_argument('--config-file', help='Single configuration JSON file (legacy mode)')
    parser.add_argument('--pattern', default='*.json', help='Glob pattern for config files (default: *.json)')
    parser.add_argument('--sort', action='store_true', default=True, help='Run configs in sorted order (default: True)')
    parser.add_argument('--gpus', default='0,1,2', help='Comma-separated GPU IDs (default: 0,1,2)')
    parser.add_argument('--jobs-per-gpu', type=int, default=4, help='Jobs per GPU (default: 4)')
    parser.add_argument('--dry-run', action='store_true', help='Validate configs without running')
    parser.add_argument('--output-dir', help='Output directory for run log')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume by validating checkpoint.json in each output directory')

    args = parser.parse_args()

    # Parse GPU list
    gpus = [int(x) for x in args.gpus.split(',')]

    # Determine config files
    if args.config_dir:
        config_pattern = os.path.join(args.config_dir, args.pattern)
        config_files = glob.glob(config_pattern)
        if args.sort:
            config_files = sorted(config_files)
    elif args.config_file:
        # Legacy single-file mode
        config_files = [args.config_file]
    else:
        print("Error: Must specify either --config-dir or --config-file")
        sys.exit(1)

    total_configs = len(config_files)

    if total_configs == 0:
        print(
            f"Error: No config files found matching pattern: {config_pattern if args.config_dir else args.config_file}")
        sys.exit(1)

    # Print startup banner
    print("=" * 80)
    print("GRID SEARCH RUNNER")
    print("=" * 80)
    if args.config_dir:
        print(f"Found {total_configs} config(s) in {args.config_dir} matching {args.pattern}")
    else:
        print(f"Running single config: {args.config_file}")
    print(f"Dry-run: {args.dry_run}")
    print(f"Resume mode: {args.resume}")
    if args.resume:
        print("Resume validation: checkpoint.json existence in output directories")
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    print()

    # Run each config
    run_start_time = time.time()
    log_entries = []
    succeeded = 0
    failed = 0
    skipped = 0

    for i, config_path in enumerate(config_files, 1):
        status, duration, error_msg = run_config(
            config_path, gpus, args.jobs_per_gpu,
            run_start_time, i, total_configs, log_entries,
            dry_run=args.dry_run,
            resume=args.resume
        )

        if status in ['success', 'partial_failure', 'dry_run_ok', 'all_complete']:
            succeeded += 1
        elif status in ['validation_failed', 'json_error']:
            skipped += 1
        else:
            failed += 1

        print()  # Blank line between configs

    # End-of-run summary
    total_elapsed = time.time() - run_start_time
    elapsed_str = format_duration(total_elapsed)

    print("=" * 80)
    print("RUN COMPLETE")
    print("=" * 80)
    print(f"Total: {total_configs} | Succeeded: {succeeded} | Failed: {failed} | Skipped: {skipped}")
    print(f"Total elapsed time: {elapsed_str}")
    print("=" * 80)

    # Write log file
    if args.output_dir or log_entries:
        output_dir = args.output_dir or './results'
        os.makedirs(output_dir, exist_ok=True)

        log_file = os.path.join(output_dir, 'run_log.csv')

        if log_entries:
            # Collect all unique keys from all log entries
            all_keys = set()
            for entry in log_entries:
                all_keys.update(entry.keys())
            keys = sorted(all_keys)  # Sort for consistent column order

            with open(log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(log_entries)
            print(f"\nRun log saved to: {log_file}")

    # Exit code
    if args.dry_run:
        # Exit nonzero if any validation failed
        if skipped > 0:
            sys.exit(1)
    else:
        # Exit nonzero if any config failed to run
        if failed > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
