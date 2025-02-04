#!/usr/bin/php
<?php

{};

$self = __FILE__;
$selfdir = dirname( $self );

require "utility.php";

$notes = <<<__EOD
  usage: $self {options} --file csvfile --outdir directory

collects results into directory 

Required

--file   csvfile               : summary csv file (required)
--outdir directory             : output directory (required, must not previously exist)

Options

--help                         : print this information and exit
--targz                        : create directory.tar.gz (optional)
--add-range-summary            : add range summary data


__EOD;

$targz = false;

$u_argv = $argv;
array_shift( $u_argv ); # first element is program name

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

        case "--outdir": {
            array_shift( $u_argv );
            if ( !count( $u_argv ) ) {
                error_exit( "ERROR: option '$arg' requires an argument\n$notes" );
            }
            $outdir = array_shift( $u_argv );
            break;
        }

        case "--targz": {
            array_shift( $u_argv );
            $targz = true;
            break;
        }

        case "--add-range-summary": {
            array_shift( $u_argv );
            $add_range_summary = true;
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

if ( !isset( $outdir ) ) {
    error_exit( "--outdir must be specified" );
}

if ( !file_exists( $file ) ) {
    error_exit( "$file does not exist" );
}

if ( false === ( $file_contents = file_get_contents( $file ) ) ) {
    error_exit( "error reading $file" );
}

if ( file_exists( $outdir ) ) {
    error_exit( "$outdir exists, remove or rename" );
}

if ( !mkdir( $outdir ) ) {
    error_exit( "$outdir could not be created" );
}

if ( !is_dir( $outdir ) ) {
    error_exit( "$outdir is not a directory after making this directory" );
}

$lines = explode( "\n", $file_contents );

$headers = explode( ",", trim( array_shift( $lines ) ) );

# debug_json( "headers", $headers );

$keepdata = [ 'train_mae', 'val_mae', 'val_max_error' ];

$used_names_to_dd = [];

foreach ( $keepdata as $v ) {
    if ( false === ( $col = array_search( $v, $headers ) ) ) {
        error_exit( "required key '$v' not found in headers of $file" );
    }

    echo "searching for $v in column $col\n";

    $bestname = '';
    $minval   = 1e99;

    foreach ( $lines as $l ) {
        $l = trim( str_replace( ', ', '_', $l ) );
        if ( empty( $l ) ) {
            continue;
        }
        # echo "line is $l\n";
        $linedata = explode( ",", trim( $l ) );
        if ( count( $linedata ) <= $col ) {
            error_exit( "line is shorter than $col\n" );
        }
        if ( $minval > $linedata[$col] ) {
            $bestname = $linedata[0];
            $minval   = $linedata[$col];
        }
    }

    headerline( "best $v : $bestname" );

    $dd = "$outdir/$v";
    if ( !mkdir( $dd ) ) {
        error_exit( "could not create directory $dd" );
    }
    if ( !is_dir( $dd ) ) {
        error_exit( "$dd is not a directory after making it" );
    }

    if ( isset( $used_names_to_dd[ $bestname ] ) ) {
        $cmd = "ln $used_names_to_dd[$bestname]/* $dd";
    } else {
        $cmd = "find . -type f -name \"$bestname*\"";
        print "$cmd\n";
        $files_array = explode( "\n", `$cmd` );
        $files = implode( ' ', $files_array  );
        $cmd = "ln $files $dd";
        if ( isset( $add_range_summary ) ) {
            $completes = array_values( preg_grep( '/complete_error_analysis.csv$/', $files_array ) );
            $cmd .= " && $selfdir/analyze_results.php --outfile $dd/${bestname}_range_summary.csv --file " . implode( ' --file ', $completes );
        }                                       
    }
    echoline();
    print "$cmd\n";
    echoline();
    print `$cmd`;

    $used_names_to_dd[ $bestname ] = $dd;
}

if ( $targz ) {
    $cmd = "tar zcf $outdir.tar.gz $outdir";
    echoline();
    print "$cmd\n";
    echoline();
    print `$cmd`;
}
