# based upon https://www.tensorflow.org/tutorials/keras/regression retrieved 2023.05.03
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

datafile     = "summary_metadata.csv"
column_names = [
    '@attributes.method',
    'CPUCount',
    'edited_radial_points.0',
    'edited_radial_points.1',
    'edited_radial_points.2',
    'edited_radial_points.3',
    'edited_radial_points.4',
    'edited_radial_points.5',
    'edited_radial_points.6',
    'edited_radial_points.7',
    'edited_radial_points.8',
    'edited_radial_points.9',
    'edited_scans.0',
    'edited_scans.1',
    'edited_scans.2',
    'edited_scans.3',
    'edited_scans.4',
    'edited_scans.5',
    'edited_scans.6',
    'edited_scans.7',
    'edited_scans.8',
    'edited_scans.9',
    'job.cluster.@attributes.name',
    'job.jobParameters.bucket_fixed.@attributes.fixedtype',
    'job.jobParameters.bucket_fixed.@attributes.value',
    'job.jobParameters.bucket_fixed.@attributes.xtype',
    'job.jobParameters.bucket_fixed.@attributes.ytype',
    'job.jobParameters.conc_threshold.@attributes.value',
    'job.jobParameters.crossover.@attributes.value',
    'job.jobParameters.curve_type.@attributes.value',
    'job.jobParameters.curves_points.@attributes.value',
    'job.jobParameters.demes.@attributes.value',
    'job.jobParameters.elitism.@attributes.value',
    'job.jobParameters.ff0_grid_points.@attributes.value',
    'job.jobParameters.ff0_max.@attributes.value',
    'job.jobParameters.ff0_min.@attributes.value',
    'job.jobParameters.ff0_resolution.@attributes.value',
    'job.jobParameters.generations.@attributes.value',
    'job.jobParameters.gfit_iterations.@attributes.value',
    'job.jobParameters.k_grid.@attributes.value',
    'job.jobParameters.max_iterations.@attributes.value',
    'job.jobParameters.mc_iterations.@attributes.value',
    'job.jobParameters.meniscus_points.@attributes.value',
    'job.jobParameters.meniscus_range.@attributes.value',
    'job.jobParameters.migration.@attributes.value',
    'job.jobParameters.mutate_sigma.@attributes.value',
    'job.jobParameters.mutation.@attributes.value',
    'job.jobParameters.p_mutate_k.@attributes.value',
    'job.jobParameters.p_mutate_s.@attributes.value',
    'job.jobParameters.p_mutate_sk.@attributes.value',
    'job.jobParameters.plague.@attributes.value',
    'job.jobParameters.population.@attributes.value',
    'job.jobParameters.regularization.@attributes.value',
    'job.jobParameters.req_mgroupcount.@attributes.value',
    'job.jobParameters.rinoise_option.@attributes.value',
    'job.jobParameters.s_grid.@attributes.value',
    'job.jobParameters.s_grid_points.@attributes.value',
    'job.jobParameters.s_max.@attributes.value',
    'job.jobParameters.s_min.@attributes.value',
    'job.jobParameters.s_resolution.@attributes.value',
    'job.jobParameters.seed.@attributes.value',
    'job.jobParameters.solute_type.@attributes.value',
    'job.jobParameters.thr_deltr_ratio.@attributes.value',
    'job.jobParameters.tikreg_alpha.@attributes.value',
    'job.jobParameters.tikreg_option.@attributes.value',
    'job.jobParameters.tinoise_option.@attributes.value',
    'job.jobParameters.uniform_grid.@attributes.value',
    'job.jobParameters.vars_count.@attributes.value',
    'job.jobParameters.x_max.@attributes.value',
    'job.jobParameters.x_min.@attributes.value',
    'job.jobParameters.y_max.@attributes.value',
    'job.jobParameters.y_min.@attributes.value',
    'job.jobParameters.z_value.@attributes.value',
    'simpoints.0',
    'simpoints.1',
    'simpoints.2',
    'simpoints.3',
    'simpoints.4',
    'simpoints.5',
    'simpoints.6',
    'simpoints.7',
    'simpoints.8',
    'simpoints.9',
    'CPUTime',
    'max_rss',
    'wallTime'

    ]

raw_dataset = pd.read_csv( datafile, names=column_names,
                           na_values='?', comment='\t',
                           sep=' ', skipinitialspace=True, low_memory=False)
dataset = raw_dataset.copy()

# optionally find and drop n/a values
# dataset.isna().sum()
# dataset = dataset.dropna()

# drop unused targets at this time
dataset.pop( "max_rss" );
dataset.pop( "wallTime" );

# split data into trainig and test sets
train_dataset  = dataset.sample(frac=0.5, random_state=0)
test_dataset   = dataset.drop(train_dataset.index)

# optionally inspect the data
## add more columns to this, all columns takes awhile!
'''
sns.pairplot(train_dataset[['@attributes.method','CPUCount','simpoints.0']], diag_kind='kde' )
plt.show()
'''

# optionally check overall statistics
'''
train_dataset.describe().transpose()
train_dataset.describe().transpose()[['mean', 'std']]
'''

# split features (the input data) from labels (the target)

train_features = train_dataset.copy()
test_features  = test_dataset.copy()

# target CPUTime

train_labels   = train_features.pop('CPUTime')
test_labels    = test_features.pop('CPUTime')

# normalization

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

# check normalization
'''
first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())
'''

## multiple input linear regression

linear_model = tf.keras.Sequential([
        normalizer,
        layers.Dense(units=1)
    ])

linear_model.predict(train_features[:10])
linear_model.layers[1].kernel
linear_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')

print("fitting linear model")

history = linear_model.fit(
        train_features,
        train_labels,
        epochs=100,
        # Suppress logging.
        verbose=0,
        # Calculate validation results on 20% of the training data.
        # could also use validation_data instead of validation_split
        validation_split = 0.2)

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    # limits can be nice if the high epoch range is known
    #    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [CPUTime]')
    plt.legend()
    plt.grid(True)
    
plot_loss(history)
test_results['linear_model'] = linear_model.evaluate(
        test_features, test_labels, verbose=0)

## DNN
# for some reason we had to split build_and_compile_model()

## model definition is currently a the MPG example... likely needs more info

def build_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    return model

def compile_model(model):
    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))

### build dnn_model

dnn_model=build_model(normalizer)
compile_model(dnn_model)
dnn_model.summary()

history = dnn_model.fit(
        train_features,
        train_labels,
        validation_split=0.2,
        verbose=0, epochs=100)

test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)

## test set performance
pd.DataFrame(test_results, index=['Mean absolute error [CPUTime]']).T

## make predictions

'''
test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [CPUTime]')
plt.ylabel('Predictions [CPUTime]')
# turned off limits for now
#lims = [0, 50]
#plt.xlim(lims)
#plt.ylim(lims)
#_ = plt.plot(lims, lims)
plt.plot()

## error distribution

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [CPUTime]')
_ = plt.ylabel('Count')


## save model
dnn_model.save('dnn_model')

## reload
reloaded = tf.keras.models.load_model('dnn_model')

test_results['reloaded'] = reloaded.evaluate(
    test_features, test_labels, verbose=0)

'''
