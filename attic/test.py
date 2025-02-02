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
# column_names = [
# $pppcolnames
#    ]

raw_dataset = pd.read_csv( datafile,
#                           names=column_names,
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

test_results = {}
histories    = {}

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

### build dnn_model
dnn_models        = {};
test_predictions  = {};
train_predictions = {};
test_errors       = {};
train_errors      = {};
hidden_size       = [ 16, 32, 64, 128, 256 ]
losses            = [ 'mse', 'mae', 'kl_divergence' ]             
epochs            = 1000
activations       = [ 'relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential' ]
# testing
losses            = [ 'mae']
hidden_size       = [ 16 ]
epochs            = 100
activations       = [ 'tanh' ]


for count in hidden_size :
    for loss in losses :
        for activation in activations :
            name = f'dnn_model_hidden_units_{loss}_{activation}_{count}'
            print( name )
            dnn_models[name]=build_model(normalizer,activation,count)
            compile_model(dnn_models[name],loss)
            dnn_models[name].summary()
            histories[name] = dnn_models[name].fit(
                train_features,
                train_labels,
                validation_split=0.2,
                verbose=0, epochs=epochs)
            test_results[name] = dnn_models[name].evaluate(test_features, test_labels, verbose=0)
            test_predictions[name] = dnn_models[name].predict(test_features).flatten()
            train_predictions[name] = dnn_models[name].predict(train_features).flatten()
            test_errors[name] = test_predictions[name] - test_labels
            train_errors[name] = train_predictions[name] - train_labels
            print( f'{name} : max train abs error {max(abs(train_errors[name]))}, max test abs error {max(abs(test_errors[name]))}' )

for count in hidden_size :
    for loss in losses :
        for activation in activations :
            name = f'dnn_model_hidden_units_{loss}_{activation}_{count}'
            print( f'{name} : max train abs error {max(abs(train_errors[name]))}, max test abs error {max(abs(test_errors[name]))}' )

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
