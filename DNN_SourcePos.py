import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

layers = keras.layers

from src import NPZParser
from src import NeuralNetwork
from src import TrainingHandler
from src import EvaluationHandler

# define file settings
# Root files are purely optimal and are left as legacy settings
ROOT_FILE_BP0mm = "OptimisedGeometry_BP0mm_2e10protons.root"
ROOT_FILE_BP5mm = "OptimisedGeometry_BP5mm_4e9protons.root"
# Training file
NPZ_FILE_TRAIN = "OptimisedGeometry_Continuous_2e10protons_DNN_S1AX.npz"
# Evaluation file (can be list)
NPZ_FILE_EVAL_0MM = "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_DNN_S1AX.npz"
NPZ_FILE_EVAL_5MM = "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_DNN_S1AX.npz"

NPZ_LOOKUP_0MM = "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_S1AX_lookup.npz"
NPZ_LOOKUP_5MM = "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_S1AX_lookup.npz"

# define directory paths
dir_main = os.getcwd()
dir_root = dir_main + "/root_files/"
dir_npz = dir_main + "/npz_files/"
dir_results = dir_main + "/results/"

# ----------------------------------------------------------------------------------------------------------------------
# Training Data

DataCluster = NPZParser.wrapper(dir_npz + NPZ_FILE_TRAIN,
                                standardize=True,
                                set_classweights=False)
DataCluster.update_indexing_positives()

print("\n# Training statistics: ")
print("Feature dimension: ({} ,{})".format(DataCluster.features.shape[0], DataCluster.features.shape[1]))
print("")

x_train = DataCluster.x_train()
y_train = DataCluster.meta[DataCluster.idx_train(), 2]
w_train = DataCluster.w_train()
x_valid = DataCluster.x_valid()
y_valid = DataCluster.meta[DataCluster.idx_valid(), 2]
x_test = DataCluster.x_test()
y_test = DataCluster.meta[DataCluster.idx_test(), 2]

# ----------------------------------------------------------------------------------------------------------------------
# Tensorflow Neural Network

model = keras.models.Sequential()
model.add(tf.keras.layers.Dense(300, input_dim=80, activation="relu"))
model.add(tf.keras.layers.Dense(300, activation="relu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(300, activation="relu"))
model.add(tf.keras.layers.Dense(300, activation="relu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(300, activation="relu"))
model.add(tf.keras.layers.Dense(300, activation="relu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1, activation="linear"))
model.compile(loss="mean_absolute_error", optimizer="SGD", metrics="mean_absolute_error")

# train here
history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                    sample_weight=w_train,
                    verbose=1,
                    epochs=100,
                    batch_size=256,
                    callbacks=[])

model.save("DNN_SourcePos_test" + ".h5")

loss = history.history['loss']
val_loss = history.history['val_loss']
mse = history.history["mean_absolute_error"]
val_mse = history.history["val_mean_absolute_error"]

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(mse, label="Training", linestyle='--', color="blue")
ax1.plot(val_mse, label="Validation", linestyle='-', color="red")
ax1.set_ylabel("mean_absolute_error")
ax1.legend()
ax1.grid()

ax2.plot(loss, label="Training", linestyle='--', color="blue")
ax2.plot(val_loss, label="Validation", linestyle='-', color="red")
ax2.set_xlabel("epoch")
ax2.set_ylabel("loss")
ax2.legend()
ax2.grid()
plt.tight_layout()
plt.savefig("DNN_SourcePos_test_training.png")

# ----------------------------------------------------------------------------------------------------------------------
# Evaluation

model.load_weights("DNN_SourcePos_test" + ".h5")

y_pred = model.predict(x_test)
y_true = y_test
y_pred = np.reshape(y_pred, newshape=(len(y_pred),))

bins = np.arange(-100.0, 100.0, 1.0)
bins_err = np.arange(-50, 50, 0.1)

plt.figure()
plt.title("Source Position distribution")
plt.xlabel("Source position z-axis [mm]")
plt.ylabel("Counts")
plt.hist(y_true, bins=bins, histtype=u"step", color="black", label="True")
plt.hist(y_pred, bins=bins, histtype=u"step", color="blue", label="Pred")
plt.legend()
plt.tight_layout()
plt.savefig("DNN_SourcePos_test_reco")

plt.figure()
plt.title("Source Position Error")
plt.xlabel("Source position z-axis [mm]")
plt.ylabel(r"$sp^{Pred}_{y}$ - $sp^{True}_{y}$")
plt.hist(y_pred - y_true, bins=bins_err, histtype=u"step", color="blue", label="Reco")
plt.legend()
plt.tight_layout()
plt.savefig("DNN_SourcePos_test_error")
