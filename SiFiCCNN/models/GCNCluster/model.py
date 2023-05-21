import pickle as pkl
import tensorflow as tf
from spektral.layers import GCNConv, GlobalAttentionPool
from SiFiCCNN.models.GCNCluster.layers import GCNConvResNetBlock


def setupModel(dropout,
               learning_rate,
               nFilter=32,
               activation="relu"):
    # original feature dimensionality
    F = 10
    S = 3
    # Model definition
    xIn = tf.keras.layers.Input(shape=(F,))
    aIn = tf.keras.layers.Input(shape=(None,), sparse=True)
    iIn = tf.keras.layers.Input(shape=(), dtype=tf.int64)

    x = GCNConv(nFilter, activation=activation, use_bias=True)([xIn, aIn])
    x = GCNConvResNetBlock(*[x, aIn], nFilter, activation)
    x = GCNConvResNetBlock(*[x, aIn], nFilter, activation)
    x = GCNConvResNetBlock(*[x, aIn], nFilter, activation)
    x = GCNConvResNetBlock(*[x, aIn], nFilter, activation)
    x = GlobalAttentionPool(32)([x, iIn])
    x = tf.keras.layers.Flatten()(x)

    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Dense(nFilter, activation="relu")(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    # Build model
    model = tf.keras.models.Model(inputs=[xIn, aIn, iIn], outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy",
                  metrics=["Precision", "Recall"])

    return model


def save_model(model,
               model_name):
    print("Saving model at: ", model_name + ".h5")
    model.save_weights(model_name + ".h5")


def loadModel(savedModelFile,
              newModel,
              custom_objects=None):
    r""" Function to load a model and pass its weights on to a newly set up model.
    This is needed if a static tensor is used in tf.layers.Input (Input(tensor = someTensor)) in a saved model.

    Args:
        savedModelFile: string. Path to file of keras model, which can be loaded with tf.keras.models.load_model.
                        This is used as weight source for newModel
        newModel: Keras model. Target model for the parameters from the loaded model.

    Returns:
        newModel: model with newly set weights.
    """

    # Load trained model
    temp_model = tf.keras.models.load_model(savedModelFile,
                                            custom_objects=custom_objects)
    # Get trained weights of loaded model and pass them on to the new model
    newModel.set_weights(temp_model.get_weights())
    # Needed to set optimizer weights properly
    newModel._make_train_function()
    # Get and set optimizer weights
    newModel.optimizer.set_weights(temp_model.optimizer.get_weights())
    lr_temp = tf.keras.backend.get_value(temp_model.optimizer.lr)
    tf.keras.backend.set_value(newModel.optimizer.lr, lr_temp)
    return newModel


def save_history(model_name,
                 history):
    # save history
    if save_history:
        with open(model_name + ".hst", 'wb') as f_hist:
            pkl.dump(history, f_hist)


def lr_scheduler(epoch):
    if epoch < 20:
        return 1e-3
    if epoch < 30:
        return 5e-4
    if epoch < 40:
        return 1e-4
    return 1e-5
