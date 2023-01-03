import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import tensorflow as tf
from tensorflow import keras

layers = keras.layers


########################################################################################################################
# Energy Regression
########################################################################################################################

def energy_regression():
    # load training data
    dir_main = os.getcwd()
    dir_npz = dir_main + "/npz_files/"

    npz_data = np.load(dir_npz + "OptimisedGeometry_BP0mm_2e10protons_DNNBase.npz")

    ary_features = npz_data["features"]
    ary_classes = npz_data["targets_clas"]
    ary_targets = npz_data["targets_reg1"]

    # remove last 2 columns from feature list
    ary_features = ary_features[:, :-2]

    # remove all background events from data set
    idx_positives = ary_classes == 1
    ary_features = ary_features[idx_positives, :]
    ary_targets = ary_targets[idx_positives, :]
    ary_classes = ary_classes[idx_positives]

    # standardize features
    for i in range(ary_features.shape[1]):
        ary_features[:, i] = (ary_features[:, i] - np.mean(ary_features[:, i])) / np.std(ary_features[:, i])

    # train test valid split
    ary_idx = np.arange(0, len(ary_classes), 1.0, dtype=int)
    rng = np.random.default_rng(42)
    rng.shuffle(ary_idx)
    p_train = 0.7
    p_test = 0.2
    p_valid = 0.1

    x_train = ary_features[ary_idx[0: int(len(ary_idx) * p_train)], :]
    y_train = ary_targets[ary_idx[0: int(len(ary_idx) * p_train)], :]

    x_valid = ary_features[ary_idx[int(len(ary_idx) * p_train): int(len(ary_idx) * (p_train + p_valid))], :]
    y_valid = ary_targets[ary_idx[int(len(ary_idx) * p_train): int(len(ary_idx) * (p_train + p_valid))], :]

    x_test = ary_features[ary_idx[int(len(ary_idx) * (p_train + p_valid)):], :]
    y_test = ary_targets[ary_idx[int(len(ary_idx) * (p_train + p_valid)):], :]

    # build deep neural network regression model

    # create model
    model = keras.models.Sequential()
    model.add(tf.keras.layers.Dense(256, input_dim=x_train.shape[1], activation="relu"))
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(2, activation="relu"))
    model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["mean_absolute_error"])
    model.summary()

    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                        verbose=2,
                        epochs=50,
                        batch_size=256)

    # plot model performance
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
    plt.show()

    # predict test set
    y_pred = model.predict(x_test)

    bins = np.arange(-2.0, 2.0, 0.05)

    plt.figure()
    plt.title("Error Energy Electron")
    plt.xlabel(r"$E_{Pred}$ - $E_{True}$")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 0] - y_test[:, 0], bins=bins, histtype=u"step", color="blue")
    plt.show()

    plt.figure()
    plt.title("Error Energy Photon")
    plt.xlabel(r"$E_{Pred}$ - $E_{True}$")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 1] - y_test[:, 1], bins=bins, histtype=u"step", color="blue")
    plt.show()

    e_diff = []
    p_diff = []
    for i in range(y_pred.shape[0]):
        e_diff.append(y_pred[i, 0] - y_test[i, 0])
        p_diff.append(y_pred[i, 1] - y_test[i, 1])
    print(min(e_diff), max(e_diff))
    print(min(p_diff), max(p_diff))

    print("Electron Energy Resolution: {:.3f} MeV".format(np.std(e_diff)))
    print("Photon Energy Resolution: {:.3f} MeV".format(np.std(p_diff)))


########################################################################################################################

def position_regression():
    # load training data
    dir_main = os.getcwd()
    dir_npz = dir_main + "/npz_files/"

    npz_data = np.load(dir_npz + "OptimisedGeometry_BP0mm_2e10protons_DNNBase.npz")

    ary_features = npz_data["features"]
    ary_classes = npz_data["targets_clas"]
    ary_targets = npz_data["targets_reg2"]

    # remove last 2 columns from feature list
    ary_features = ary_features[:, :-2]

    # remove all background events from data set
    idx_positives = ary_classes == 1
    ary_features = ary_features[idx_positives, :]
    ary_targets = ary_targets[idx_positives, :]
    ary_classes = ary_classes[idx_positives]

    # standardize features
    for i in range(ary_features.shape[1]):
        ary_features[:, i] = (ary_features[:, i] - np.mean(ary_features[:, i])) / np.std(ary_features[:, i])

    # train test valid split
    ary_idx = np.arange(0, len(ary_classes), 1.0, dtype=int)
    rng = np.random.default_rng(42)
    rng.shuffle(ary_idx)
    p_train = 0.7
    p_test = 0.2
    p_valid = 0.1

    x_train = ary_features[ary_idx[0: int(len(ary_idx) * p_train)], :]
    y_train = ary_targets[ary_idx[0: int(len(ary_idx) * p_train)], :]

    x_valid = ary_features[ary_idx[int(len(ary_idx) * p_train): int(len(ary_idx) * (p_train + p_valid))], :]
    y_valid = ary_targets[ary_idx[int(len(ary_idx) * p_train): int(len(ary_idx) * (p_train + p_valid))], :]

    x_test = ary_features[ary_idx[int(len(ary_idx) * (p_train + p_valid)):], :]
    y_test = ary_targets[ary_idx[int(len(ary_idx) * (p_train + p_valid)):], :]

    # build deep neural network regression model

    # create model
    model = keras.models.Sequential()
    model.add(tf.keras.layers.Dense(256, input_dim=x_train.shape[1], activation="relu"))
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    # model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(6, activation="linear"))
    model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["mean_absolute_error"])
    model.summary()

    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                        verbose=2,
                        epochs=50,
                        batch_size=256)

    # plot model performance
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
    plt.show()

    # predict test set
    y_pred = model.predict(x_test)

    bins_x = np.arange(-20.5, 20.5, 0.5)
    bins_y = np.arange(-60.5, 60.5, 0.5)
    bins_z = np.arange(-20.5, 20.5, 0.5)

    plt.figure()
    plt.title("Error Position x")
    plt.xlabel(r"$x_{Pred}$ - $x_{True}$")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 0] - y_test[:, 0], bins=bins_x, histtype=u"step", color="blue", label=r"e_x")
    plt.hist(y_pred[:, 3] - y_test[:, 3], bins=bins_x, histtype=u"step", color="orange", label=r"p_x")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("Error Position x")
    plt.xlabel(r"$y_{Pred}$ - $y_{True}$")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 1] - y_test[:, 1], bins=bins_y, histtype=u"step", color="blue", label=r"e_y")
    plt.hist(y_pred[:, 4] - y_test[:, 4], bins=bins_y, histtype=u"step", color="orange", label=r"p_y")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("Error Position x")
    plt.xlabel(r"$z_{Pred}$ - $z_{True}$")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 2] - y_test[:, 2], bins=bins_z, histtype=u"step", color="blue", label=r"e_z")
    plt.hist(y_pred[:, 5] - y_test[:, 5], bins=bins_z, histtype=u"step", color="orange", label=r"p_z")
    plt.legend()
    plt.show()


########################################################################################################################

# energy_regression()
position_regression()
