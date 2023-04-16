def return_model(input_shape):
    """
    Dense neural network with basic structure.
    Set to be a baseline model for future neural networks.

    Args:
        input_dim (int): dimension of the features (number of features)

    :return: Tensorflow model
    """
    import tensorflow as tf
    from tensorflow import keras
    layers = keras.layers

    ####################################################################################################################

    # create model
    model = keras.models.Sequential()
    # model.add(tf.keras.layers.Masking(mask_value=0.0, input_shape=(None, None, input_shape)))
    model.add(tf.keras.layers.Dense(64, input_dim=input_shape, activation="relu"))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    # model.add(tf.keras.layers.Dropout(0.05))
    model.add(tf.keras.layers.Dense(6, activation="linear"))
    model.compile(loss="mean_absolute_error", optimizer="Adam", metrics=["mean_absolute_error"])
    # return model
    return model
