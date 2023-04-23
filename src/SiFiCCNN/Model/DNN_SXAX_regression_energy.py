import tensorflow as tf
from tensorflow import keras
layers = keras.layers

def return_model(timesteps, features):
    """
    Dense neural network with basic structure.
    Set to be a baseline model for future neural networks.

    Args:
        input_dim (int): dimension of the features (number of features)

    :return: Tensorflow model
    """

    # create model
    model = keras.models.Sequential()
    # add first dense layer with predefined input dimension
    # model.add(tf.keras.layers.Masking(mask_value=0.0, input_shape=(None, input_shape)))
    model.add(tf.keras.layers.Flatten(input_shape=(timesteps, features)))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(2, activation="linear"))
    model.compile(loss="mean_absolute_error",
                  optimizer="Adam",
                  metrics=["mean_absolute_error"])
    return model
