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

def train_model():
    # load input dataset
    dir_main = os.getcwd()
    dir_npz = dir_main + "/npz_files/"
    dir_model = dir_main + "/models/"

    npz_data = np.load(dir_npz + "NNInputDenseBase_OptimisedGeometry_BP0mm_2e10protons.npz")
    ary_features = npz_data["features"]
    ary_targets = npz_data["targets"]
    ary_w = npz_data["weights"]

    p_train = 0.7
    p_test = 0.2
    p_valid = 0.1

    # generate shuffled indices with a random generator
    ary_idx = np.arange(0, len(ary_targets), 1.0, dtype=int)
    rng = np.random.default_rng(42)
    rng.shuffle(ary_idx)

    ary_idx_train = ary_idx[0: int(len(ary_idx) * p_train)]
    ary_idx_valid = ary_idx[int(len(ary_idx) * p_train): int(len(ary_idx) * (p_train + p_valid))]
    ary_idx_test = ary_idx[int(len(ary_idx) * (1 - p_test)):]

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
    w_train = ary_w[ary_idx_train]

    x_valid = ary_features[ary_idx_valid]
    y_valid = ary_targets[ary_idx_valid]

    # define model
    model = keras.models.Sequential()
    model.add(tf.keras.layers.Dense(dense_nodes, input_dim=x_train.shape[1], activation=activation))
    model.add(tf.keras.layers.Dense(dense_nodes, activation=activation))
    model.add(tf.keras.layers.Dense(dense_nodes, activation=activation))
    model.add(tf.keras.layers.Dense(dense_nodes, activation=activation))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])

    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), verbose=1, epochs=epochs,
                        batch_size=batch_size, sample_weight=w_train)
    # save model weights
    model.save(dir_model + model_name + ".h5")
    # save model history
    with open(dir_model + model_name + ".hst", 'wb') as f_hist:
        pkl.dump(history.history, f_hist)


########################################################################################################################

def eval_model():
    import matplotlib.pyplot as plt
    from fastROCAUC import fastROCAUC

    # load input dataset
    dir_main = os.getcwd()
    dir_npz = dir_main + "/npz_files/"
    dir_model = dir_main + "/models/"

    npz_data = np.load(dir_npz + "NNInputDenseBase_OptimisedGeometry_BP0mm_2e10protons.npz")
    ary_features = npz_data["features"]
    ary_targets = npz_data["targets"]
    ary_w = npz_data["weights"]

    p_train = 0.7
    p_test = 0.2
    p_valid = 0.1

    # generate shuffled indices with a random generator
    ary_idx = np.arange(0, len(ary_targets), 1.0, dtype=int)
    rng = np.random.default_rng(42)
    rng.shuffle(ary_idx)

    ary_idx_train = ary_idx[0: int(len(ary_idx) * p_train)]
    ary_idx_valid = ary_idx[int(len(ary_idx) * p_train): int(len(ary_idx) * (p_train + p_valid))]
    ary_idx_test = ary_idx[int(len(ary_idx) * (1 - p_test)):]

    # normalization
    for i in range(ary_features.shape[1]):
        ary_features[:, i] = (ary_features[:, i] - np.mean(ary_features[:, i])) / np.std(ary_features[:, i])

    # set training and validation data
    x_test = ary_features[ary_idx_test]
    y_test = ary_targets[ary_idx_test]

    # load neural network model
    model = keras.models.load_model(dir_model + "DNN_base.h5")
    # load neural network history
    with open(dir_model + "DNN_base.hst", 'rb') as f_hist:
        history = pkl.load(f_hist)

    # plot model performance
    loss = history['loss']
    val_loss = history['val_loss']
    mse = history["accuracy"]
    val_mse = history["val_accuracy"]

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(mse, label="Training", linestyle='--', color="blue")
    ax1.plot(val_mse, label="Validation", linestyle='-', color="red")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid()

    ax2.plot(loss, label="Training", linestyle='--', color="blue")
    ax2.plot(val_loss, label="Validation", linestyle='-', color="red")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.legend()
    ax2.grid()
    plt.tight_layout()
    plt.savefig("DNN_base_history.png")

    # predict test set
    y_pred = model.predict(x_test)

    # run ROC curve and AUC score analysis
    auc, theta = fastROCAUC(y_pred, y_test, results=True)

    # best optimal threshold
    threshold = theta

    # evaluate important metrics
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        # apply prediction threshold
        if y_pred[i] >= threshold:
            y_pred[i] = 1
        else:
            y_pred[i] = 0

        if y_pred[i] == 1 and y_test[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_test[i] == 0:
            FP += 1
        if y_pred[i] == 0 and y_test[i] == 0:
            TN += 1
        if y_pred[i] == 0 and y_test[i] == 1:
            FN += 1

    if (TP + FN) == 0:
        efficiency = 0
    else:
        efficiency = TP / (TP + FN)
    if (TP + FP) == 0:
        purity = 0
    else:
        purity = TP / (TP + FP)

    print("\nClassification results: ")
    print("AUC score: {:.3f}".format(auc))
    print("Threshold: {:.3f}".format(theta))
    print("Accuracy: {:.1f}".format((TP + TN) / (TP + TN + FP + FN) * 100))
    print("Efficiency: {:.1f}%".format(efficiency * 100))
    print("Purity: {:.1f}%".format(purity * 100))
    print("TP: {} | TN: {} | FP: {} | FN: {}".format(TP, TN, FP, FN))
