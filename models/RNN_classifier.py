import tensorflow as tf
from tensorflow import keras

layers = keras.layers


# ----------------------------------------------------------------------------------------------------------------------

def return_model(n_timesteps, n_features):
    """
    Dense neural network with basic structure.
    Set to be a baseline model for future neural networks.

    Args:
        input_dim (int): dimension of the features (number of features)

    :return: Tensorflow model
    """

    model = keras.models.Sequential()
    model.add(keras.layers.Masking(mask_value=0.,
                                   input_shape=(n_timesteps, n_features)))
    model.add(keras.layers.LSTM(32, return_sequence=True))
    model.add(keras.layers.LSTM(16, return_sequence=True))
    model.add(keras.layers.Dense(16, activation="relu"))
    model.add(keras.layers.Dense(units=1, activation="sigmoid"))
    # compile model with loss function, optimizer and accuracy
    model.compile(loss="binary_crossentropy", optimizer="Adam",
                  metrics=[keras.metrics.Precision(), keras.metrics.Recall()])
    # return model
    return model
