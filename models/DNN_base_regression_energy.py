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
    # create model
    model = keras.models.Sequential()
    # add first dense layer with predefined input dimension
    model.add(tf.keras.layers.Dense(128, input_dim=input_dim, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(2, activation="relu"))
    model.compile(loss=tf.keras.losses.log_cosh, optimizer="SGD", metrics=["mean_absolute_error"])
    return model
