#!/usr/bin/php
<?php

{};

$self = __FILE__;

require "utility.php";

$notes = <<<__EOD
  usage: $self {options} --file csvfile

collects results into directory 

Required

--file    csvfile              : summary csv file (required, can be specified multiple times)

Options

--outfile csvoutfile           : create csv output file
--help                         : print this information and exit


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
            $files[] = array_shift( $u_argv );
            break;
        }

        case "--outfile": {
            array_shift( $u_argv );
            if ( !count( $u_argv ) ) {
                error_exit( "ERROR: option '$arg' requires an argument\n$notes" );
            }
            $outfile = array_shift( $u_argv );
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

if ( !count( $files ) ) {
    error_exit( "--file must be specified" );
}

$csvout = '';

$firstfile = true;

foreach ( $files as $file ) {
    if ( !file_exists( $file ) ) {
        error_exit( "$file does not exist" );
    }

    if ( false === ( $file_contents = file_get_contents( $file ) ) ) {
        error_exit( "error reading $file" );
    }

    $lines = explode( "\n", $file_contents );

    $headers = explode( ",", trim( array_shift( $lines ) ) );

    $features = preg_grep( '/^feature_/', $headers );

    # debug_json( "headers", $headers );
    # debug_json( "features", $features );

    $header_ae = 'absolute_error';
    
    if ( false === ( $header_ae_col = array_search( $header_ae, $headers ) ) ) {
        error_exit( "required key '$header_ae' not found in headers of $file" );
    }

    $ranges = [
        [ 0, 60 * 5 ] # 0 to 5m
        ,[ 0, 60 * 15 ] # 0 to 15m
        ,[ 0, 60 * 30 ] # 0 to 30m
        ,[ 0, 60 * 60 ] # 0 to 1h
        ,[ 0, 60 * 60 * 2 ] # 0 to 2h
        ,[ 0, 60 * 60 * 4 ] # 0 to 4h
        ,[ 60 * 60 * 2, 1e99 ] # 2h to max
        ,[ 60 * 60 * 4, 1e99 ] # 4h to max
        ,[ 60 * 60 * 8, 1e99 ] # 8h to max
        ];

    ## get min/max of each feature based upon absolute error

    $csvhdr = [ "start_abs_err", "end_abs_err", "count", "percent_of_total" ];
    foreach ( $features as $hdr ) {
        $csvhdr[] = "min_$hdr";
        $csvhdr[] = "max_$hdr";
    }

    if ( $firstfile ) {
        ## only headers for the first file 
        $csvout .= implode( ",", $csvhdr ) . "\n";
        $firstfile = false;
    }

    $csvout .= "\n$file\n";

    foreach ( $ranges as $range ) {
        $start = $range[0];
        $end   = $range[1];

        headerline( "range $start:$end" );

        $csvrow = [];

        $first  = true;

        foreach ( $features as $col => $hdr ) {
            $minval    = 1e99;
            $maxval    = -1e99;
            $count     = 0;
            $count_tot = 0;
            $max_time  = $maxval;
            
            foreach ( $lines as $l ) {
                # $l = trim( str_replace( ', ', '_', $l ) );
                $l = trim( $l );
                if ( empty( $l ) ) {
                    continue;
                }
                # echo "line is $l\n";
                $linedata = explode( ",", $l );
                if ( count( $linedata ) <= $col ) {
                    error_exit( "line is shorter than $col\n" );
                }
                ++$count_tot;
                if ( $linedata[ $header_ae_col ] < $start
                     || $linedata[ $header_ae_col ] > $end ) {
                    ## outside of range
                    continue;
                }

                if ( $max_time < $linedata[ $header_ae_col ] ) {
                    $max_time = $linedata[ $header_ae_col ];
                }

                if ( $minval > $linedata[$col] ) {
                    $minval = $linedata[$col];
                }
                if ( $maxval < $linedata[$col] ) {
                    $maxval = $linedata[$col];
                }

                ++$count;
            }
            if ( $first ) {
                $csvrow[] = dhms_from_minutes( $start / 60 );
                $csvrow[] = $end == 1e99 ? dhms_from_minutes( $max_time  / 60 ) . " (max found)" : dhms_from_minutes( $end / 60 );
                $csvrow[] = $count;
                $csvrow[] = $count_tot > 0 ? sprintf( "%.2f", 100 * $count / $count_tot ) : "n/a";

                $first    = false;
            }

            $csvrow[] = $minval;
            $csvrow[] = $maxval;

            echo "$start:$end $hdr recs $count $minval $maxval\n";
        }

        $csvout .= implode( ",", $csvrow ) . "\n";
        
    }
}

if ( isset( $outfile ) ) {
    if ( false === file_put_contents( $outfile, $csvout ) ) {
        error_exit( "error creating outfile '$outfile'" );
    }
    echo "$outfile created\n";
}
