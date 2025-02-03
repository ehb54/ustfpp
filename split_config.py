import json
import os
import sys
import re

def parse_args():
    """
    Parse command line arguments
    """
    import argparse
    parser = argparse.ArgumentParser(description='Split json config file into multiple files suitable for parallel runs')

    # config file
    parser.add_argument( '--config-file', help='read options from JSON formatted configuration file', required=True )

    # GPUs
    parser.add_argument( '--gpus', help='gpu numbers to run on, separate with commas if more than one', required=True )

    # jobs per gpu
    parser.add_argument( '--jobs-per-gpu', help='the number of jobs to run simultaneously on each GPU', required=True )

    # trial run
    parser.add_argument( '--trial-run', action='store_true', help='do not actually run, just show what would be done' )

    # job unique output base directory
    parser.add_argument( '--job-unique-output-dir', help='create a unique directory for each run based upon the job parameters under the specified directory' )

    return parser.parse_args()

def main():
    """
    Main function to run the hyperparameter search experiments
    """
    args = parse_args()
    print(args)

    config_file = args.config_file
    print("Loading from config file %s\n" % (config_file))
    if not os.path.isfile( config_file ) :
        print("Error: file %s not found\n" % (config_file), file=sys.stderr)
        sys.exit(-2)
    with open(config_file, 'r') as f:
        ## strip out comment lines
        lines = '';
        for line in f:
            line = line.strip()
            if not line.startswith('#'):
                lines += line

        ## convert json string to dictionary
        try:
            config_json_data = json.loads(lines)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e, file=sys.stderr)
            print("json:\n", lines )
            sys.exit(-3)
    
    print(config_json_data)

    training_fraction = config_json_data['training_fraction'];
    validation_fraction = config_json_data['validation_fraction'];
    epochs              = config_json_data['epochs'];
    patience            = config_json_data['patience'];

    mirroredstrategy = False
    if 'mirroredstrategy' in config_json_data :
        mirroredstrategy = config_json_data['mirroredstrategy']
        
    gpu_mem_grow = True
    if 'gpu_mem_grow' in config_json_data :
        gpu_mem_grow = config_json_data['gpu_mem_grow']
    
    gpus       = list( map( int, args.gpus.split( ',' ) ) )
    gpus_count = len( gpus )
    gpu_index  = 0
    cmds_for_gpus = ['' for x in range( gpus_count )];
    total_job_count = 0
    
    for experiment_type in config_json_data['experiment_files']:
        for arch in config_json_data['architectures']:
            for scaler_name in config_json_data['scalers']:
                for optimizer in config_json_data['optimizers']:
                    for batch_size in config_json_data['batch_sizes']:
                        for activation in config_json_data['activations']:
                            for dropout_rate in config_json_data['dropout_rates']:
                                config_name = f'{experiment_type}_arch_{arch}_scaler_{scaler_name}_opt_{optimizer}_batch_{batch_size}_act_{activation}_drop_{dropout_rate}_training_{training_fraction}_validation_{validation_fraction}_epochs_{epochs}'
                                config_name = re.sub(r'[\[\], ]+', '_', config_name )
                                file_name   = f'{config_name}.json'
                                total_job_count += 1
                                output_dir  = config_json_data['output_dir']
                                if 'job_unique_output_dir' in args:
                                    output_dir = os.path.join( args.job_unique_output_dir, config_name )
                                this_json_data = {
                                    'experiment_files'     : {
                                        f'{experiment_type}' : config_json_data['experiment_files'][experiment_type]
                                        }
                                    ,'architectures'        : [ arch ]
                                    ,'scalers'              : [ scaler_name ]
                                    ,'optimizers'           : [ optimizer ]
                                    ,'batch_sizes'          : [ batch_size ]
                                    ,'activations'          : [ activation ]
                                    ,'dropout_rates'        : [ dropout_rate ]
                                    ,'epochs'               : epochs
                                    ,'mirroredstrategy'     : mirroredstrategy
                                    ,'patience'             : patience
                                    ,'gpu_mem_grow'         : gpu_mem_grow
                                    ,'usegpu'               : gpus[gpu_index]
                                    ,'training_fraction'    : training_fraction
                                    ,'validation_fraction'  : validation_fraction
                                    ,'output_dir'           : output_dir
                                    }
                                cmds_for_gpus[ gpu_index ] += f'{file_name}\n'
                                gpu_index = (gpu_index + 1 ) % gpus_count;

                                if 'trial_run' in args and args.trial_run :
                                    print( f'not created, trial run: {file_name} would contain:' )
                                    print ( json.dumps( this_json_data, indent=4 ) )
                                    print ("-----------------------")
                                else:
                                    with open( file_name, 'w' ) as f:
                                        print( json.dumps( this_json_data, indent=4 ), file=f )
                                        print( f'> {file_name}' )


    for gpu_index in range(gpus_count) :
        list_file_name = 'configs_gpu_%s.txt' % ( gpus[gpu_index] )
        run_file_name = 'gpu_%s.run' % ( gpus[gpu_index] )
        cmd_to_run = f'cat {list_file_name} | xargs -P{args.jobs_per_gpu} -n1 python hyperparameter_architecture_analysis.py --config-file';

        if 'trial_run' in args and args.trial_run :
            print( f'not created, trial run: {list_file_name} would contain:' )
            print( cmds_for_gpus[gpu_index] )
            print ("-----------------------")
            print( f'not created, trial run: {run_file_name} would contain:' )
            print( cmd_to_run )
            print ("-----------------------")
        else :
            with open( list_file_name, 'w' ) as f:
                print( cmds_for_gpus[gpu_index], file=f )
                print( f'> {list_file_name}' )
            with open( run_file_name, 'w' ) as f:
                print( cmd_to_run, file=f )
                print( f'> {run_file_name}' )


    print ("-----------------------")
    print(f'total jobs count {total_job_count}')
    
if __name__ == "__main__":
    main()
