import tensorflow as tf
from tensorflow import keras

layers = keras.layers


def loss_energy_asym(y_true, y_pred):
    loss = y_true - y_pred
    loss = tf.where(loss < 0.0, 8 * (loss + 1e-7), loss)
    loss = tf.abs(loss)
    return tf.reduce_mean(loss)


def loss_energy_relative(y_true, y_pred):
    loss = tf.square(y_true - y_pred) / y_true
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
    """   
    # create model
    model = keras.models.Sequential()
    # add first dense layer with predefined input dimension
    model.add(tf.keras.layers.Dense(128, input_dim=input_dim, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    # model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(2, activation="relu"))
    model.compile(loss="mean_absolute_error", optimizer="SGD", metrics=["mean_absolute_error"])
    return model
    """

    # create model
    model = keras.models.Sequential()
    # add first dense layer with predefined input dimension
    model.add(tf.keras.layers.Dense(128, input_dim=input_dim, activation="relu"))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(2, activation="linear"))
    model.compile(loss="mean_absolute_error", optimizer="Adam", metrics=["mean_absolute_error"])
    return model
