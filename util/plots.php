#!/usr/bin/php
<?php

{};

const PLOTLY_COLORS         = [
    '#1f77b4'  # muted blue
    ,'#ff7f0e'  # safety orange
    ,'#2ca02c'  # cooked asparagus green
    ,'#d62728'  # brick red
    ,'#9467bd'  # muted purple
    ,'#8c564b'  # chestnut brown
    ,'#e377c2'  # raspberry yogurt pink
    ,'#7f7f7f'  # middle gray
    ,'#bcbd22'  # curry yellow-green
    ,'#17becf'  # blue-teal
    ];

$secs_limit = 300;

$self = __FILE__;
$selfdir = dirname( $self );

require "utility.php";

$notes = <<<__EOD
  usage: $self {options} --name modelname

builds max absolute error & z-score plots using plotly

Required

--name   modelname             : modelname

Options

--title desc                   : add text to title
--help                         : print this information and exit

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
        case "--name": {
            array_shift( $u_argv );
            if ( !count( $u_argv ) ) {
                error_exit( "ERROR: option '$arg' requires an argument\n$notes" );
            }
            $name = array_shift( $u_argv );
            break;
        }
        case "--title": {
            array_shift( $u_argv );
            if ( !count( $u_argv ) ) {
                error_exit( "ERROR: option '$arg' requires an argument\n$notes" );
            }
            $title = array_shift( $u_argv );
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

if ( !isset( $name ) ) {
    error_exit( "--name must be specified" );
}

$expected_data = [
    '_val_complete_error_analysis.csv'
    ,'_train_complete_error_analysis.csv'
    ,'_test_complete_error_analysis.csv'
    ];

$files = [];
foreach ( $expected_data as $v ) {
    $file = "$name$v";
    if ( !file_exists( $file ) ) {
        error_exit( "expected data file $file does not exist" );
    }
    $files[] = $file;
}
        
$expected_zscore = "_z_score.csv";
$zscore_file = "$name$expected_zscore";
if ( !file_exists( $zscore_file ) ) {
    error_exit( "expected data file $zscore_file does not exist" );
}

## get max absolute error

$firstfile = true;

$absolute_errors = [];

$total_jobs = 0;

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

    foreach ( $lines as $l ) {
        # $l = trim( str_replace( ', ', '_', $l ) );
        $l = trim( $l );
        if ( empty( $l ) ) {
            continue;
        }
        
        ++$total_jobs;
        
        $linedata = explode( ",", $l );
        if ( count( $linedata ) <= $header_ae_col ) {
            error_exit( "line is shorter than $header_ae_col\n" );
        }
        if ( floatval( $linedata[ $header_ae_col ] ) > $secs_limit ) {
            $absolute_errors[] = floatval( sprintf( "%.2f", floatval( $linedata[ $header_ae_col ] ) / 60 ) );
        }
    }
}

sort( $absolute_errors, SORT_NUMERIC );
$absolute_errors = array_reverse( $absolute_errors );

print "datapoints " . count( $absolute_errors ) . "\n";

# $absolute_errors = array_slice( $absolute_errors, 0, 20000 );

# debug_json( "abserrors", $absolute_errors );

#            ,"line" : {
#                "color"  : "rgb(150,150,222)"
#                ,"width" : 1
#            }


#        ,"annotations" : [ 
#            {
#                "xref"       : "paper"
#                ,"yref"      : "paper"
#                ,"x"         : -0.1
#                ,"xanchor"   : "left"
#                ,"y"         : -0.3
#                ,"yanchor"   : "top"
#                ,"text"      : ""
#                ,"showarrow" : false
#            }
#        }

$plot = json_decode(
'
{
    "data" : [
        {
            "y"       : []
            ,"type" : "histogram"
        }
    ]
    ,"layout" : {
        "title" : {
            "text" : ""
        }
        ,"font" : {
            "color"  : "rgb(0,5,80)"
        }
        ,"margin" : {
            "b" : 100
        }
        ,"paper_bgcolor": "white"
        ,"plot_bgcolor": "white"
        ,"xaxis" : {
            "gridcolor" : "rgba(111,111,111,0.5)"
            ,"type" : "log"
            ,"title" : {
                "text" : "Number of jobs (log scale)"
                ,"font" : {
                    "color"  : "rgb(0,5,80)"
                }
            }
            ,"showticklabels" : true
            ,"showline"       : true
        }
        ,"yaxis" : {
            "gridcolor" : "rgba(111,111,111,0.5)"
            ,"type" : "log"
            ,"title" : {
                "text" : "Absolute prediction error [minutes] (log scale)"
                ,"font" : {
                    "color"  : "rgb(0,5,80)"
                }
            }
            ,"showline"       : true
        }
    }
}
                '
    );

$plot->data[0]->y = $absolute_errors;
if ( isset( $title ) ) {
    $plot->layout->title->text .= "$title<br>";
}
$plot->layout->title->text .=
    "Number of jobs vs. Absolute prediction error<br>"
    . sprintf( "%d (%.1f%%) of jobs with abs. pred. error < %dm not shown"
               ,$total_jobs - count( $absolute_errors )
               ,100 * ($total_jobs - count( $absolute_errors ) ) / $total_jobs
               ,$secs_limit / 60
    );

$plotobj = [
    "plotlydata" => $plot
    ,"_height" => 1000
    ,"_width" => 2000
];

# debug_json( "plot", $plot );

file_put_contents( "plotforpy.json", "\n" . json_encode( $plotobj ) . "\n\n" );
$png = run_cmd( "$selfdir/plotly2img.py plotforpy.json" );
echo "$png\n";

file_put_contents( "plot.json", "\n" . json_encode( $plot ) . "\n\n" );

                   
### plot zscore

echo "$zscore_file\n";

if ( false === ( $zscore_file_contents = file_get_contents( $zscore_file ) ) ) {
    error_exit( "error reading $zscore_file" );
}

$lines = explode( "\n", $zscore_file_contents );

$headers = explode( ",", trim( array_shift( $lines ) ) );

debug_json( 'headers', $headers );

$zsd = (object)[];

foreach ( $lines as $l ) {
    # $l = trim( str_replace( ', ', '_', $l ) );
    $l = trim( $l );
    if ( empty( $l ) ) {
        continue;
    }
    
    ++$total_jobs;
    
    $linedata = explode( ",", $l );
    foreach ( $linedata as $k => $v ) {
        if ( !isset( $zsd->{$headers[$k]} ) ) {
            $zsd->{$headers[$k]} = [];
        }
        $zsd->{$headers[$k]}[] = floatval( $v );
    }
}

#debug_json( "zsd", $zsd );

$plot = json_decode(
'
{
    "data" : [
        {
            "x"       : []
            ,"y"       : []
            ,"type"    : "scatter"
            ,"line"    : {
                "color"  : "#1f77b4"
            }
        }
        ,{
            "x"       : []
            ,"y"       : []
            ,"yaxis"   : "y2"
            ,"type"    : "scatter"
            ,"line"    : {
                "color"  : "#2ca02c"
            }
        }
    ]
    ,"layout" : {
        "title" : {
            "text" : ""
        }
        ,"font" : {
            "color"  : "rgb(0,5,80)"
        }
        ,"margin" : {
            "b" : 100
        }
        ,"showlegend" : false
        ,"paper_bgcolor": "white"
        ,"plot_bgcolor": "white"
        ,"xaxis" : {
            "gridcolor" : "rgba(111,111,111,0.5)"
            ,"type" : "linear"
            ,"title" : {
                "text" : "z-score cutoff"
                ,"font" : {
                    "color"  : "rgb(0,5,80)"
                }
            }
            ,"showticklabels" : true
            ,"showline"       : true
        }
        ,"yaxis" : {
            "gridcolor" : "rgba(111,111,111,0.5)"
            ,"type" : "linear"
            ,"title" : {
                "text" : "percent accepted"
                ,"font" : {
                    "color"  : "#1f77b4"
                }
            }
        }
        ,"yaxis2" : {
            "gridcolor" : "rgba(111,111,111,0.5)"
            ,"type" : "linear"
            ,"title" : {
                "text" : "Maximum absolute prediction error [minutes]"
                ,"font" : {
                    "color"  : "#2ca02c"
                }
            }
            ,"showline"       : true
            ,"overlaying"     : "y"
            ,"side"           : "right"
        }
    }
}
                '
    );

$plot->data[0]->x = $zsd->z_score;
$plot->data[0]->y = $zsd->percent_records_accepted;
$plot->data[1]->x = $zsd->z_score;
$plot->data[1]->y = array_map( function($v) { return $v / 60; }, $zsd->accepted_maximum_absolute_error );

if ( isset( $title ) ) {
    $plot->layout->title->text .= "$title<br>";
}
$plot->layout->title->text .=
    "Percent jobs accepted and Maximum absolute error<br>vs<br>z-score outlier rejection cutoff"
    ;

$plotobj = [
    "plotlydata" => $plot
    ,"_height" => 1000
    ,"_width" => 2000
];

# debug_json( "plot", $plot );

file_put_contents( "plotzscoreforpy.json", "\n" . json_encode( $plotobj ) . "\n\n" );
$png = run_cmd( "$selfdir/plotly2img.py plotzscoreforpy.json" );
echo "$png\n";

file_put_contents( "plotzscore.json", "\n" . json_encode( $plot ) . "\n\n" );
