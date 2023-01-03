def return_model(input_dim):
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

    model = keras.models.Sequential()
    model.add(tf.keras.layers.Dense(256, input_dim=input_dim, activation="relu"))
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    # model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(6, activation="linear"))
    model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["mean_absolute_error"])

    return model
