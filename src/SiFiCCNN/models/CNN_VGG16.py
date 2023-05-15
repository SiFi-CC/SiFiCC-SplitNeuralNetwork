import numpy as np
import pickle as pkl

import tensorflow as tf
from tensorflow import keras


def setupModel(nOutput,
               dropout,
               loss,
               output_activation):
    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(input_shape=(32, 16, 2),
                                  filters=64,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  activation="relu"))

    model.add(keras.layers.Conv2D(filters=64,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  activation="relu"))

    model.add(keras.layers.MaxPool2D(pool_size=(2, 2),
                                     strides=(2, 2)))

    model.add(keras.layers.Conv2D(filters=128,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  activation="relu"))

    model.add(keras.layers.Conv2D(filters=128,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  activation="relu"))

    model.add(keras.layers.MaxPool2D(pool_size=(2, 2),
                                     strides=(2, 2)))

    model.add(keras.layers.Conv2D(filters=256,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  activation="relu"))

    model.add(keras.layers.Conv2D(filters=256,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  activation="relu"))

    model.add(keras.layers.Conv2D(filters=256,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  activation="relu"))

    model.add(keras.layers.MaxPool2D(pool_size=(2, 2),
                                     strides=(2, 2)))

    model.add(keras.layers.Flatten(name="flatten"))
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(nOutput, activation=output_activation))

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    if loss == "binary_clas":
        loss = "binary_crossentropy"
        metrics = ["Precision", "Recall"]
    if loss == "regression":
        loss = "mean_absolute_error"
        metrics = ["mean_absolute_error"]
    else:
        loss = "mean_absolute_error"
        metrics = ["mean_absolute_error"]

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    return model
