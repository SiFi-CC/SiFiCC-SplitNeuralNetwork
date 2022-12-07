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
    # model settings
    dense_nodes = 128
    dense_layers = 4

    activation = "relu"
    loss_function = "binary_crossentropy"
    optimizer = "adam"
    dropout_rate = 0.2

    ####################################################################################################################

    # create model
    model = keras.models.Sequential()
    # add first dense layer with predefined input dimension
    model.add(tf.keras.layers.Dense(dense_nodes, input_dim=input_dim, activation=activation))
    # add further dense layers (with dropout if needed)
    for i in range(dense_layers - 1):
        model.add(tf.keras.layers.Dense(dense_nodes, activation=activation))
        # model.add(tf.keras.layers.Dropout(dropout_rate / 2))
    # add final dropout layer
    model.add(tf.keras.layers.Dropout(dropout_rate))
    # output node with sigmoid as activation function
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    # compile model with loss function, optimizer and accuracy
    model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])
    # return model
    return model
