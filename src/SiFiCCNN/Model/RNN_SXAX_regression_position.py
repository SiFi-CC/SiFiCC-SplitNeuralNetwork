def return_model(timesteps, features):
    """
    Dense neural network with basic structure.
    Set to be a baseline model for future neural networks.

    Args:
        input_dim (int): dimension of the features (number of features)

    :return: Tensorflow model
    """
    import tensorflow as tf
    from tensorflow import keras

    # create model
    model = keras.models.Sequential()
    # add first dense layer with predefined input dimension
    model.add(tf.keras.layers.Masking(mask_value=0., input_shape=(timesteps, features)))
    model.add(tf.keras.layers.LSTM(32, input_dim=features))
    model.add(tf.keras.layers.Dense(6, activation="linear"))
    model.compile(loss="mean_absolute_error",
                  optimizer="Adam",
                  metrics=["mean_absolute_error"])
    # return model
    return model
