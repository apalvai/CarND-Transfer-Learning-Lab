import pickle
import tensorflow as tf
import numpy as np
# TODO: import Keras layers you need here
import keras as K
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Activation, Flatten, Dropout
from sklearn.preprocessing import LabelBinarizer

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    
    # preprocess data
    y_train = y_train.reshape((-1, 1))

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    nb_examples = X_train.shape[0]
    nb_classes = len(np.unique(y_train))
    input_shape = X_train[0].shape
    
    # model
    input = Input(input_shape)
    x = Flatten()(input)
    x = Dense(nb_classes, activation='softmax')(x)
    model = Model(input, x)
    model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
    
    # TODO: train your model here
    history = model.fit(X_train, y_train, nb_epoch=100, validation_split=0.33)
    
    # evaluate
    y_val = y_val.reshape((-1, 1))

    metrics = model.evaluate(X_val, y_val)
    for metric_i in range(len(model.metrics_names)):
        metric_name = model.metrics_names[metric_i]
        metric_value = metrics[metric_i]
        print('{}: {}'.format(metric_name, metric_value))


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
