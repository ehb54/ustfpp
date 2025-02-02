# based upon https://www.tensorflow.org/tutorials/keras/regression retrieved 2023.05.03
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re        # regular expressions
import functools # for reduce()

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

datafile     = "summary_metadata.csv"
# column_names = [
# $pppcolnames
#    ]

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


train_dataset  = dataset.sample(frac=0.5, random_state=0)
test_dataset   = dataset.drop(train_dataset.index)

# split features (the input data) from labels (the target)

train_features = train_dataset.copy()
test_features  = test_dataset.copy()

# target CPUTime

train_labels   = train_features.pop('CPUTime')
test_labels    = test_features.pop('CPUTime')

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

test_results = {}
histories    = {}

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
