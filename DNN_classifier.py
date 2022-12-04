import os
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow import keras

layers = keras.layers

########################################################################################################################
# Global DNN settings
########################################################################################################################

model_name = "DNN_base"

epochs = 10

dense_nodes = 64
dense_layers = 4
batch_size = 256
dropout_rate = 0.2
lr = 1e-3

initializer = tf.keras.initializers.random_normal()
activation = "relu"
loss_function = "binary_crossentropy"

########################################################################################################################
# Preprocessing
########################################################################################################################

# load input dataset
dir_main = os.getcwd()
dir_data = dir_main + "/data/"
dir_model = dir_main + "/models/"

npz_data = np.load(dir_data + "DNN_input_OptimizedGeometry_BP0mm_2e10protons.npz")
ary_features = npz_data["features"]
ary_targets = npz_data["targets"]
ary_w = npz_data["weights"]

ary_idx_train = npz_data["idx_train"]
ary_idx_valid = npz_data["idx_valid"]
ary_idx_test = npz_data["idx_test"]

# define neural network sample weighting
# define class weights
_, counts = np.unique(ary_targets, return_counts=True)
class_weights = {0: len(ary_targets) / (2 * counts[0]),
                 1: len(ary_targets) / (2 * counts[1])}

# normalization
for i in range(ary_features.shape[1]):
    ary_features[:, i] = (ary_features[:, i] - np.mean(ary_features[:, i])) / np.std(ary_features[:, i])

# set training and validation data
x_train = ary_features[ary_idx_train]
y_train = ary_targets[ary_idx_train]

x_valid = ary_features[ary_idx_valid]
y_valid = ary_targets[ary_idx_valid]

# define model
model = keras.models.Sequential()
model.add(tf.keras.layers.Dense(dense_nodes, input_dim=x_train.shape[1], activation=activation))
for i in range(dense_layers - 1):
    model.add(tf.keras.layers.Dense(dense_nodes, activation=activation))
    # model.add(tf.keras.layers.Dropout(dropout_rate / 2))
model.add(tf.keras.layers.Dropout(dropout_rate))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])

history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), verbose=1, epochs=epochs,
                    batch_size=batch_size, class_weight=class_weights)
# save model weights
model.save(dir_model + model_name + ".h5")
# save model history
with open(dir_model + model_name + ".hst", 'wb') as f_hist:
    pkl.dump(history.history, f_hist)
