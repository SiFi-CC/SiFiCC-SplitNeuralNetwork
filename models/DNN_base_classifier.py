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

    activation = "relu"
    loss_function = "binary_crossentropy"
    dropout_rate = 0.20
    lr = 1e-3

    ####################################################################################################################

    # create model
    model = keras.models.Sequential()
    # add first dense layer with predefined input dimension
    model.add(tf.keras.layers.Dense(64, input_dim=input_dim, activation=activation))
    model.add(tf.keras.layers.Dense(32, activation=activation))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    # output node with sigmoid as activation function
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    # compile model with loss function, optimizer and accuracy
    model.compile(loss=loss_function, optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=["accuracy"])
    print(model.summary)
    # return model
    return model
