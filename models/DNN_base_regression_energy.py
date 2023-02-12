import tensorflow as tf
from tensorflow import keras

layers = keras.layers

"""
def loss_relative_energy(y_true, y_pred):

    y_true1, y_true2 = tf.split(y_true, num_or_size_splits=2, axis=-1)
    y_pred1, y_pred2 = tf.split(y_pred, num_or_size_splits=2, axis=-1)

    loss1 = y_true1 - y_pred1
    loss1 = tf.where(loss1 < 0.0, 2 * (loss1 + 1e-7), loss1)
    loss1 = tf.abs(loss1)

    loss2 = y_true2 - y_pred2
    loss2 = tf.where(loss2 < 0.0, 3 * (loss2 + 1e-7), loss2)
    loss2 = tf.abs(loss2)

    return tf.reduce_mean(loss1 + loss2)
"""


def loss_relative_energy(y_true, y_pred):
    loss = y_true - y_pred
    loss = tf.where(loss < 0.0, 4 * (loss + 1e-7), loss)
    loss = tf.abs(loss)
    return tf.reduce_mean(loss)


def return_model(input_dim):
    """
    Dense neural network with basic structure.
    Set to be a baseline model for future neural networks.

    Args:
        input_dim (int): dimension of the features (number of features)

    :return: Tensorflow model
    """

    ####################################################################################################################
    # create model
    model = keras.models.Sequential()
    # add first dense layer with predefined input dimension
    model.add(tf.keras.layers.Dense(128, input_dim=input_dim, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(2, activation="relu"))
    model.compile(loss=loss_relative_energy, optimizer="SGD", metrics=["mean_absolute_error"])
    return model
