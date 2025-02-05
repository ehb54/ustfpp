#!/usr/bin/php
<?php

{};

$self = __FILE__;

require "utility.php";

$notes = <<<__EOD
  usage: $self {options} --file csvfile -=z-score # 

analyzes each column of csvfile, determines limits for each and outputs csvfile trimmed of outliers to z-score

Required

--file           csvfile      : data csv file
--z-score        #            : z-score for cutoff

Options

--out-config     filename     : config file output
--out-accept-csv filename     : create csv output file of accepted records
--out-reject-csv filename     : create csv output file of accepted records

--help                        : print this information and exit


__EOD;

$u_argv = $argv;
array_shift( $u_argv ); # first element is program name

$files = [];

while( count( $u_argv ) && substr( $u_argv[ 0 ], 0, 1 ) == "-" ) {
    switch( $arg = $u_argv[ 0 ] ) {
        case "--help": {
            echo $notes;
            exit;
        }
        case "--file": {
            array_shift( $u_argv );
            if ( !count( $u_argv ) ) {
                error_exit( "ERROR: option '$arg' requires an argument\n$notes" );
            }
            $file = array_shift( $u_argv );
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

        case "--out-config": {
            array_shift( $u_argv );
            if ( !count( $u_argv ) ) {
                error_exit( "ERROR: option '$arg' requires an argument\n$notes" );
            }
            $out_config = array_shift( $u_argv );
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

      default:
        error_exit( "\nUnknown option '$u_argv[0]'\n\n$notes" );
    }        
}

if ( count( $u_argv ) ) {
    echo $notes;
    exit;
}

if ( !isset( $file ) ) {
    error_exit( "--file must be specified" );
}

if ( !isset( $z_score ) ) {
    error_exit( "--z-score must be specified" );
}

if ( isset( $out_config )
     && file_exists( $out_config ) ) {
    error_exit( "--out-config file '$out_config' already exists, rename or remove" );
}

if ( isset( $out_accept_csv )
     && file_exists( $out_accept_csv ) ) {
    error_exit( "--out-accept_csv file '$out_accept_csv' already exists, rename or remove" );
}

if ( isset( $out_reject_csv )
     && file_exists( $out_reject_csv ) ) {
    error_exit( "--out-reject_csv file '$out_reject_csv' already exists, rename or remove" );
}

if ( !file_exists( $file ) ) {
    error_exit( "$file does not exist" );
}

if ( false === ( $file_contents = file_get_contents( $file ) ) ) {
    error_exit( "error reading $file" );
}

$lines = explode( "\n", $file_contents );

$headers = explode( ",", trim( array_shift( $lines ) ) );

$features = $headers;

$exclude_features = [
    'CPUTime'
    ,'max_rss'
    ,'wallTime'
#    ,"CPUCount"
#    ,"edited_radial_points.0"
#    ,"edited_scans.0"
#    ,"job.cluster.@attributes.name"
#    ,"job.jobParameters.ff0_grid_points.@attributes.value"
#    ,"job.jobParameters.ff0_max.@attributes.value"
#    ,"job.jobParameters.ff0_min.@attributes.value"
#    ,"job.jobParameters.ff0_resolution.@attributes.value"
#    ,"job.jobParameters.max_iterations.@attributes.value"
#    ,"job.jobParameters.mc_iterations.@attributes.value"
#    ,"job.jobParameters.meniscus_points.@attributes.value"
#    ,"job.jobParameters.meniscus_range.@attributes.value"
#    ,"job.jobParameters.req_mgroupcount.@attributes.value"
#    ,"job.jobParameters.rinoise_option.@attributes.value"
#    ,"job.jobParameters.s_grid_points.@attributes.value"
#    ,"job.jobParameters.s_max.@attributes.value"
#    ,"job.jobParameters.s_min.@attributes.value"
#    ,"job.jobParameters.s_resolution.@attributes.value"
#    ,"job.jobParameters.tinoise_option.@attributes.value"
#    ,"job.jobParameters.uniform_grid.@attributes.value"
#    ,"simpoints.0"
    ];

debug_json( "N.B. excluded features", $exclude_features );

foreach ( $exclude_features as $v ) {
    if (($key = array_search( $v, $features)) !== false) {
        unset( $features[$key] );
    }
}

debug_json( "features after exclude", $features );

## initialize object

$sumobj = (object)[];

foreach ( $features as $feature_col => $feature ) {
    $sumobj->{$feature} = (object)[];
    $sumobj->{$feature}->key = $feature_col;
    $sumobj->{$feature}->min   = 1e99;
    $sumobj->{$feature}->max   = -1e99;
    $sumobj->{$feature}->sum   = 0;
    $sumobj->{$feature}->sum2  = 0;
    $sumobj->{$feature}->data  = [];
}

## pass thru data, collect sum, sum2, min, max

# debug_json( "sumobj init", $sumobj );

$count = 0;

echo "computing statistics...\n";

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

    ++$count;
    
    foreach ( $features as $feature_col => $feature ) {

        $sumobj->{$feature}->data[] = $line_array[ $feature_col ];

        $sumobj->{$feature}->sum  += $line_array[ $feature_col ];
        $sumobj->{$feature}->sum2 += $line_array[ $feature_col ] * $line_array[ $feature_col ];

        if ( $sumobj->{$feature}->min > $line_array[ $feature_col ] ) {
            $sumobj->{$feature}->min = $line_array[ $feature_col ];
        }
        
        if ( $sumobj->{$feature}->max < $line_array[ $feature_col ] ) {
            $sumobj->{$feature}->max = $line_array[ $feature_col ];
        }
    }

    # debug_json( "sumobj 1 row", $sumobj );
}

## compute mean, sd, median
        
if ( $count <= 2 ) {
    error_exit( "insufficient data to continue, only $count rows of data found" );
}

foreach ( $features as $feature_col => $feature ) {
    $sumobj->{$feature}->mean       = $sumobj->{$feature}->sum / $count;
    $sumobj->{$feature}->sd         = sqrt( abs( ( ( $count * $sumobj->{$feature}->sum2 ) - ( $sumobj->{$feature}->sum * $sumobj->{$feature}->sum ) ) / ( $count * ( $count - 1 ) ) ) );
    $sumobj->{$feature}->accept_min = $sumobj->{$feature}->mean - $z_score * $sumobj->{$feature}->sd;
    $sumobj->{$feature}->accept_max = $sumobj->{$feature}->mean + $z_score * $sumobj->{$feature}->sd;

    ## median
    sort( $sumobj->{$feature}->data );
    $index = intdiv( $count, 2 );
    $sumobj->{$feature}->median     =
        $count & 1
        ? $sumobj->{$feature}->data[ $index ] ## odd
        : ( $sumobj->{$feature}->data[ $index - 1 ] +  $sumobj->{$feature}->data[ $index ] ) / 2 ## even
        ;
    unset( $sumobj->{$feature}->data );
    # perhaps median for rejection
    # $sumobj->{$feature}->accept_min = $sumobj->{$feature}->median - $z_score * $sumobj->{$feature}->sd;
    # $sumobj->{$feature}->accept_max = $sumobj->{$feature}->median + $z_score * $sumobj->{$feature}->sd;
}

echo "done computing statistics\n";

debug_json( "sumobj", $sumobj );

## compute rejected/accepted based upon z-score, show summaries

$accepted_rows = [];
$rejected_rows = [];
$rejection_col_counts = [];
foreach ( $features as $feature_col => $feature ) {
    $rejected_col_counts[ $feature ] = 0;
}

echo "computing accepted/rejected rows...\n";
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
    } else {
        $accepted_rows[] = $line;
    }
}
echo "done computing accepted/rejected rows\n";

debug_json( "rejected_col_counts", $rejected_col_counts );

headerline( "summary" );

echo sprintf(
    "total data rows %d\n"
    . "rows accepted %d (%.2f %%)\n"
    . "rows rejected %d (%.2f %%)\n"
    , $count
    , count( $accepted_rows )
    , 100 * count( $accepted_rows ) / $count
    , count( $rejected_rows )
    , 100 * count( $rejected_rows ) / $count
    );

if ( isset( $out_config ) ) {
    if ( false ===
         file_put_contents(
             $out_config
             , json_encode( $sumobj, JSON_PRETTY_PRINT ) . "\n"
         ) ) {
        error_exit( "error writing --out-config file '$out_config'" );
    }
    echo "created $out_config\n";
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
    echo "created $out_accept_csv\n";
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
    echo "created $out_reject_csv\n";
}
