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

    # create model
    model = keras.models.Sequential()
    # add first dense layer with predefined input dimension
    # model.add(tf.keras.layers.Masking(mask_value=0.0, input_shape=(None, input_shape)))
    model.add(tf.keras.layers.Dense(64, input_dim=input_shape, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    # compile model with loss function, optimizer and accuracy
    model.compile(loss="binary_crossentropy",
                  optimizer="Adam",
                  metrics=[keras.metrics.Precision(), keras.metrics.Recall()])
    # return model
    return model
