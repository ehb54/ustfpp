#!/usr/bin/perl

@activations = (
    "relu"
    ,"leaky_relu"
    ,"para_relu"
    ,"sigmoid"
    ,"softmax"
    ,"softplus"
    ,"softsign"
    ,"tanh"
    ,"selu"
    ,"elu"
    ,"exponential"
    )
    ;

@losses = (
    "mse"
    ,"mae"
    ,"kl_divergence"
    );

$epochs = 1000;

@validation_splits = (
#    .3
    .4
#    ,.5
    );

## limited by cuda memory on js2 1/5th GPU

$procs = 1;

for $i ( @activations ) {
    for my $j ( @losses ) {
        for my $k ( @validation_splits ) {
            $cmds{$count++ % $procs} .=
                qq~python tfrun.py summary_metadata.csv config_scan.json '{"activations":["$i"],"losses":["$j"],"epochs":$epochs,"validation_split":$k}' 2>&1 > runs/${i}_${j}_${k}.out\n~;
        }
    }
}

for $i ( keys %cmds ) {
    my $f = "mkrun_$i.sh";
    ">> $f\n";
    open my $fh, ">$f";
    print $fh $cmds{$i};
    close $fh;
    print "$f\n";
}



