import json
import sys
import os
import re        # regular expressions
import functools # for reduce()
import io        # for io.StringIO()

# based upon https://www.tensorflow.org/tutorials/keras/regression retrieved 2023.05.03
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

notes = f'''usage: {sys.argv[0]} csvfile configjson {{overrides}}

run tf tests on csvfile as described in configjson

csvfile    : space separated csv file, first row headers, subsequent rows numeric values
configjson : parameter configurations
overrides  : json string with any overrides to configjson

'''

if len( sys.argv ) < 3:
    sys.exit(notes)
    
datafile = sys.argv[1]
if not os.path.exists(datafile) :
    sys.exit(f'file {datafile} does not exist\n')

config_file_name = sys.argv[2]
if not os.path.exists(config_file_name) :
    sys.exit(f'file {config_file_name} does not exist\n')
    
config_file_lines = open(config_file_name, 'r').readlines();
## remove comments
rx_nc = re.compile('^\s*#')
config = json.loads(''.join([s for s in config_file_lines if not rx_nc.match(s)]))
    
if ( len( sys.argv ) > 3 ) :
    overrides = json.load( io.StringIO( sys.argv[3] ) )
    for key in overrides:
        config[key]=overrides[key]
        
validation_split = config['validation_split']

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

print(f'reading data from {datafile}')
raw_dataset = pd.read_csv( datafile,
#                           names=column_names,
                           na_values='?', comment='\t',
                           sep=' ', skipinitialspace=True, low_memory=False)
dataset = raw_dataset.copy()

## find how many experimental datasets
max_exps = len([ s for s in list(dataset.columns) if re.compile( "^edited_scans.(\\d+)" ).match(s) ])
## compute total datapoints
dataset['total_datapoints'] = functools.reduce( lambda a, b : a + b, list(map( lambda x: dataset["edited_scans." + str(x)] * dataset["edited_radial_points." + str(x)], range( 0, max_exps ) )))

## just 2DSA-MC for now
dataset=dataset.loc[(dataset['@attributes.method'] == 1) & (dataset['job.jobParameters.mc_iterations.@attributes.value'] > 1) ]

## prune columns for now
dataset=dataset[
    [
        '@attributes.method'
        ,'CPUCount'
        ,'job.cluster.@attributes.name'
        ,'job.jobParameters.ff0_grid_points.@attributes.value'
        ,'job.jobParameters.ff0_resolution.@attributes.value'
        ,'job.jobParameters.max_iterations.@attributes.value'
        ,'job.jobParameters.mc_iterations.@attributes.value'
        ,'job.jobParameters.s_grid_points.@attributes.value'
        ,'job.jobParameters.s_resolution.@attributes.value'
        ,'job.jobParameters.solute_type.@attributes.value'
        ,'CPUTime'
        ,'total_datapoints'
]];


# optionally find and drop n/a values
# dataset.isna().sum()
# dataset = dataset.dropna()

train_dataset  = dataset.sample(frac=config['train_fraction'], random_state=0)
test_dataset   = dataset.drop(train_dataset.index)

# split features (the input data) from labels (the target)

train_features = train_dataset.copy()
test_features  = test_dataset.copy()

# target CPUTime

train_labels   = train_features.pop('CPUTime')
test_labels    = test_features.pop('CPUTime')

## normalization is recommended ?

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

# optionally check normalization
'''
first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())
'''

test_results      = {}
histories         = {}
dnn_models        = {}
test_predictions  = {}
train_predictions = {}
test_errors       = {}
train_errors      = {}

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    # limits can be nice if the high epoch range is known
    #    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [CPUTime]')
    plt.legend()
    plt.grid(True)

## DNN
# for some reason we had to split build_and_compile_model() from MPG ref implementation

## model definition is currently a the MPG example... likely needs more info

def build_model(norm, activation, count):
    model = keras.Sequential([
        norm,
        layers.Dense(count, activation=activation),
        layers.Dense(count, activation=activation),
        layers.Dense(1)
    ])
    return model

def compile_model(model, loss):
    model.compile(loss=loss,
                  optimizer=tf.keras.optimizers.Adam(0.001))

for count in config['hidden_size'] :
    for loss in config['losses'] :
        for activation in config['activations'] :
            name = f'dnn_model_{loss}_{activation}_hidden_units{count}_vs{validation_split}'
            print( name )
            dnn_models[name]=build_model(normalizer,activation,count)
            compile_model(dnn_models[name],loss)
            dnn_models[name].summary()
            histories[name] = dnn_models[name].fit(
                train_features,
                train_labels,
                validation_split=validation_split,
                verbose=0, epochs=config['epochs'])
            test_results[name] = dnn_models[name].evaluate(test_features, test_labels, verbose=0)
            test_predictions[name] = dnn_models[name].predict(test_features).flatten()
            train_predictions[name] = dnn_models[name].predict(train_features).flatten()
            test_errors[name] = test_predictions[name] - test_labels
            train_errors[name] = train_predictions[name] - train_labels
            print( f'{name} : max train abs error {max(abs(train_errors[name]))}, max test abs error {max(abs(test_errors[name]))}' )
            dnn_models[name].save(f'runs/{name}.dnn_model')

for count in config['hidden_size'] :
    for loss in config['losses'] :
        for activation in config['activations'] :
            name = f'dnn_model_{loss}_{activation}_hidden_units{count}_vs{validation_split}'
            print( f'{name} : max train abs error {max(abs(train_errors[name]))}, max test abs error {max(abs(test_errors[name]))}' )

