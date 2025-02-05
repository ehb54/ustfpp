#!/usr/bin/php
<?php

{};

$self = __FILE__;

require "utility.php";

$notes = <<<__EOD
  usage: $self {options} --config configjson --error-analysis-file filename

Reads the config json and analysis the error analysis file


Required

--config                filename          : config file input in JSON format
--error-analysis-file   filename          : csv file from an error analysis (can be specified multiple times)

Options

--z-score               #                 : recomputed min/max from config file based on updated z-score
--z-score-range         start, end, delta : run a z-score scan 

--out-accept-csv        filename          : create csv output file of accepted records
--out-reject-csv        filename          : create csv output file of accepted records
--only-accepted-max-error                 : just report accepted max error

--help                                    : print this information and exit


__EOD;

$u_argv = $argv;
array_shift( $u_argv ); # first element is program name

$error_analysis_files = [];

while( count( $u_argv ) && substr( $u_argv[ 0 ], 0, 1 ) == "-" ) {
    switch( $arg = $u_argv[ 0 ] ) {
        case "--help": {
            echo $notes;
            exit;
        }
        case "--config" : {
            array_shift( $u_argv );
            if ( !count( $u_argv ) ) {
                error_exit( "ERROR: option '$arg' requires an argument\n$notes" );
            }
            $config = array_shift( $u_argv );
            break;
        }
        case "--error-analysis-file" : {
            array_shift( $u_argv );
            if ( !count( $u_argv ) ) {
                error_exit( "ERROR: option '$arg' requires an argument\n$notes" );
            }
            $error_analysis_files[] = array_shift( $u_argv );
            break;
        }
        case "--out-accept-csv": {
            array_shift( $u_argv );
            if ( !count( $u_argv ) ) {
                error_exit( "ERROR: option '$arg' requires an argument\n$notes" );
            }
            $out_accept_csv = array_shift( $u_argv );
            break;
        }
        case "--out-reject-csv": {
            array_shift( $u_argv );
            if ( !count( $u_argv ) ) {
                error_exit( "ERROR: option '$arg' requires an argument\n$notes" );
            }
            $out_reject_csv = array_shift( $u_argv );
            break;
        }
        case "--only-accepted-max-error" : {
            array_shift( $u_argv );
            $only_accepted_max_error = true;
            break;
        }
        case "--z-score": {
            array_shift( $u_argv );
            if ( !count( $u_argv ) ) {
                error_exit( "ERROR: option '$arg' requires an argument\n$notes" );
            }
            $z_score = array_shift( $u_argv );
            break;
        }
        case "--z-score-range": {
            array_shift( $u_argv );
            if ( count( $u_argv ) < 3 ) {
                error_exit( "ERROR: option '$arg' requires three arguments\n$notes" );
            }
            $z_score_range       = true;
            $z_score_range_start = array_shift( $u_argv );
            $z_score_range_end   = array_shift( $u_argv );
            $z_score_range_delta = array_shift( $u_argv );
            break;
        }
      default:
        error_exit( "\nUnknown option '$u_argv[0]'\n\n$notes" );
    }        
}

if ( count( $u_argv ) ) {
    echo $notes;
    exit;
}

if ( isset( $z_score ) && isset( $z_score_range ) ) {
    error_exit( "--z-score and --z-score-range are mutually exclusive" );
}

if ( isset( $z_score_range )
     && (
         isset( $out_accept_csv )
         || isset( $out_reject_csv )
         )
    ) {
    error_exit( "--z-score-range can not be used with --out-accept-csv nor --out-reject-csv" );
}

if ( !isset( $config ) ) {
    error_exit( "--config must be specified" );
}

if ( !count( $error_analysis_files ) ) {
    error_exit( "--error-analysis-file must be specified" );
}

if ( isset( $out_accept_csv )
     && file_exists( $out_accept_csv ) ) {
    error_exit( "--out-accept_csv file '$out_accept_csv' already exists, rename or remove" );
}

if ( isset( $out_reject_csv )
     && file_exists( $out_reject_csv ) ) {
    error_exit( "--out-reject_csv file '$out_reject_csv' already exists, rename or remove" );
}

if ( !file_exists( $config ) ) {
    error_exit( "$config does not exist" );
}

if ( false === ( $config_contents = file_get_contents( $config ) ) ) {
    error_exit( "error reading $config" );
}

$config_data = json_decode( $config_contents );
if ( $config_data === null && json_last_error() !== JSON_ERROR_NONE ) {
    error_exit( "JSON decode error in $config" );
}

$first_analysis_file = true;
foreach ( $error_analysis_files as $error_analysis_file ) {
    if ( !file_exists( $error_analysis_file ) ) {
        error_exit( "$error_analysis_file does not exist" );
    }

    if ( false === ( $this_error_analysis_file_contents = file_get_contents( $error_analysis_file ) ) ) {
        error_exit( "error reading $error_analysis_file" );
    }

    $these_lines = explode( "\n", $this_error_analysis_file_contents );
    if ( !count( $these_lines ) ) {
        error_exit( "error $error_analysis_file is empty!" );
    }

    if ( $first_analysis_file ) {
        $first_analysis_file          = false;
        $first_analysis_file_header   = $these_lines[ 0 ];
        $error_analysis_file_contents = $this_error_analysis_file_contents;
        
    } else {
        if ( $first_analysis_file_header != $these_lines[ 0 ] ) {
            error_exit( "mismatching file headers $error_analysis_file vs $error_analysis_files[0]" );
        }
        array_shift( $these_lines );
        $error_analysis_file_contents .= implode( "\n", $these_lines ) . "\n";
    }
}

# debug_json( "config_data", $config_data );

$lines = explode( "\n", $error_analysis_file_contents );

$headers = explode( ",", trim( array_shift( $lines ) ) );

$features = $headers;

#debug_json( "features", $features );
#debug_json( "headers", $headers );

## validate all config features in features

$sumobj = (object)[];

foreach ( $config_data as $v ) {
    if (($key = array_search( $v->feature, $features)) === false) {
        error_exit( "key $v->feature not found in headers of $error_analysis_file" );
    }
    $sumobj->{$v->feature}      = $v;
    $sumobj->{$v->feature}->key = $key;
}

# debug_json( "sumobj", $sumobj );

# remove features not in config
foreach ( $features as $k => $v ) {
    if ( !isset( $sumobj->$v ) ) {
        unset( $features[$k] );
    }
}

# find 'absolute_error' column
$absolute_error_col_name = 'absolute_error';
if (($absolute_error_col = array_search( $absolute_error_col_name, $headers)) === false) {
    error_exit( "key $absolute_error_col_name not found in headers of $error_analysis_file" );
}

# optionally recompute min/max
if ( isset( $z_score ) ) {
    foreach ( $features as $feature ) {
        if ( !isset( $sumobj->{$feature} ) ) {
            error_exit( "sumobj does not have feature $feature" );
        }
        $sumobj->{$feature}->accept_min = $sumobj->{$feature}->mean - $z_score * $sumobj->{$feature}->sd;
        $sumobj->{$feature}->accept_max = $sumobj->{$feature}->mean + $z_score * $sumobj->{$feature}->sd;
        # perhaps median for rejection
        # $sumobj->{$feature}->accept_min = $sumobj->{$feature}->median - $z_score * $sumobj->{$feature}->sd;
        # $sumobj->{$feature}->accept_max = $sumobj->{$feature}->median + $z_score * $sumobj->{$feature}->sd;
    }
}

$sumobj_by_z_score = (object)[];
if ( isset( $z_score_range ) ) {
    for ( $z = $z_score_range_start; $z <= $z_score_range_end; $z += $z_score_range_delta ) {
        $sumobj_by_z_score->{$z} = json_decode( json_encode( $sumobj ) );

        foreach ( $features as $feature ) {
            if ( !isset( $sumobj_by_z_score->{$z}->{$feature} ) ) {
                error_exit( "sumobj_by_z_score->{$z} does not have feature $feature" );
            }
            $sumobj_by_z_score->{$z}->{$feature}->accept_min = $sumobj_by_z_score->{$z}->{$feature}->mean - $z * $sumobj_by_z_score->{$z}->{$feature}->sd;
            $sumobj_by_z_score->{$z}->{$feature}->accept_max = $sumobj_by_z_score->{$z}->{$feature}->mean + $z * $sumobj_by_z_score->{$z}->{$feature}->sd;
            # perhaps median for rejection
            # $sumobj_by_z_score->{$z}->{$feature}->accept_min = $sumobj_by_z_score->{$z}->{$feature}->median - $z * $sumobj_by_z_score->{$z}->{$feature}->sd;
            # $sumobj_by_z_score->{$z}->{$feature}->accept_max = $sumobj_by_z_score->{$z}->{$feature}->median + $z * $sumobj_by_z_score->{$z}->{$feature}->sd;
        }
    }
    # debug_json( "sumobj_by_z_score", $sumobj_by_z_score );
    # exit;
} else {
    $sumobj_by_z_score->from_config = $subobj;
}

if ( !isset( $only_accepted_max_error ) ) {
    debug_json( "N.B. features to be checked", $features );
}

function maybe_echo( $msg ) {
    global $only_accepted_max_error;
    if ( isset( $only_accepted_max_error ) ) {
        return;
    }
    echo $msg;
}

if ( isset( $z_score_range ) && isset( $only_accepted_max_error ) ) {
    echo "z_score,accepted_maximum_absolute_error,accepted_maximum_absolute_error_hms,number_of_accepted_records,number_of_rejected_records,percent_records_accepted\n";
}

foreach ( $sumobj_by_z_score as $sumobj_z_score_name => $sumobj ) {

    maybe_echo( "computing accepted/rejected rows for $sumobj_z_score_name ...\n" );

    $count = 0;
    $accepted_rows = [];
    $rejected_rows = [];
    $rejection_col_counts = [];
    foreach ( $features as $feature_col => $feature ) {
        $rejected_col_counts[ $feature ] = 0;
    }
    $accepted_max_abs_error = -1e99;
    $rejected_min_abs_error = 1e99;

    foreach ( $lines as $line ) {
        # echo "line: $line\n";
        
        if ( !strlen( trim( $line ) ) ) {
            ## skip insufficent data lines
            continue;
        }

        $line_array = explode( ',', trim( $line ) );

        if ( count( $line_array ) < count( $headers ) ) {
            echo sprintf( "WARNING : insufficent data %d need %d\n", count( $line_array ), count( $headers ) );
            continue;
        }

        $rejected = false;
        ++$count;
        
        foreach ( $features as $feature_col => $feature ) {
            if ( $line_array[ $feature_col ] < $sumobj->{$feature}->accept_min
                 || $line_array[ $feature_col ] > $sumobj->{$feature}->accept_max ) {
                $rejected = true;
                $rejected_col_counts[ $feature ]++;
                # echo "rejected value for column $feature_col : " . $line_array[$feature_col] . "\n";
                # exit;
            }
        }

        if ( $rejected ) {
            $rejected_rows[] = $line;
            if ( $rejected_min_abs_error > $line_array[ $absolute_error_col ] ) {
                $rejected_min_abs_error = $line_array[ $absolute_error_col ];
            }
        } else {
            $accepted_rows[] = $line;
            if ( $accepted_max_abs_error < $line_array[ $absolute_error_col ] ) {
                $accepted_max_abs_error = $line_array[ $absolute_error_col ];
            }
        }
    }
    maybe_echo( "done computing accepted/rejected rows for $sumobj_z_score_name\n" );

    if ( !isset( $only_accepted_max_error ) ) {
        debug_json( "rejected_col_counts", $rejected_col_counts );

        headerline( "summary $sumobj_z_score_name" );

        echo sprintf(
            "total data rows %d\n"
            . "rows accepted %d (%.2f %%) maximum absolute error %s\n"
            . "rows rejected %d (%.2f %%) minimum absolute error %s\n"
            , $count
            , count( $accepted_rows )
            , 100 * count( $accepted_rows ) / $count
            , dhms_from_minutes( $accepted_max_abs_error / 60 )
            , count( $rejected_rows )
            , 100 * count( $rejected_rows ) / $count
            , dhms_from_minutes( $rejected_min_abs_error / 60 )
            );

    } else {
        if ( isset( $z_score_range ) ) {
            echo sprintf(
                "%s,%.2f,%s,%d,%d,%.2f\n"
                ,$sumobj_z_score_name
                ,$accepted_max_abs_error
                ,dhms_from_minutes( $accepted_max_abs_error / 60 )
                ,count( $accepted_rows )
                ,count( $rejected_rows )
                ,100 * count( $accepted_rows ) / $count
                );
        } else {
            echo "$accepted_max_abs_error\n";
        }
    }

    if ( isset( $out_accept_csv ) ) {
        if ( false ===
             file_put_contents(
                 $out_accept_csv
                 , implode( ',', $headers ) . "\n"
                 . implode( "\n", $accepted_rows ) . "\n"
             ) ) {
            error_exit( "error writing --out-accept-csv file '$out_accept_csv'" );
        }
        maybe_echo( "created $out_accept_csv\n" );
    }

    if ( isset( $out_reject_csv ) ) {
        if ( false ===
             file_put_contents(
                 $out_reject_csv
                 , implode( ',', $headers ) . "\n"
                 . implode( "\n", $rejected_rows ) . "\n"
             ) ) {
            error_exit( "error writing --out-reject-csv file '$out_reject_csv'" );
        }
        maybe_echo( "created $out_reject_csv\n" );
    }
}
