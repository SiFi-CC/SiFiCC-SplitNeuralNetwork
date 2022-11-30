import os
import numpy as np

from classes import Rootdata
from classes import root_files

import tensorflow as tf
from tensorflow import keras

layers = keras.layers
from sklearn.model_selection import train_test_split

from fastROCAUC import fastROCAUC
import matplotlib.pyplot as plt
import copy

dir_main = os.getcwd()


########################################################################################################################

def network_training():
    # load training data
    data = np.load(dir_main + "/data/" + "model_test.npz")
    ary_features = data["features"]
    ary_targets = data["targets"]
    # load some sort of primary energy array for energy weighting

    # define class weights
    # define energy weights

    # normalize dataset

    # train-test-valid split by indexing


    # subsample balanced training set
    ary_idx0 = [i for i in range(len(ary_targets)) if ary_targets[i] == 0]
    ary_idx1 = [i for i in range(len(ary_targets)) if ary_targets[i] == 1]
    np.random.shuffle(ary_idx0)

    ary_idx = np.concatenate([ary_idx0[:len(ary_idx1)], ary_idx1])
    np.random.shuffle(ary_idx)
    ary_features = ary_features[ary_idx]
    ary_targets = ary_targets[ary_idx]

    # generate index sequence, shuffle sequence and sample by ratio
    idx = np.arange(0, ary_features.shape[0], 1.0, dtype=int)
    np.random.shuffle(idx)
    idx_stop = int(len(idx) * 0.8)
    idx_train = idx[0:idx_stop]
    idx_test = idx[idx_stop + 1:]

    # generate training and test sample
    x_training = ary_features[idx_train]
    y_training = ary_targets[idx_train]

    x_test = ary_features[idx_test]
    y_test = ary_targets[idx_test]

    _, counts = np.unique(y_training, return_counts=True)

    # define class weights
    class_weights = [len(y_training) / (2 * counts[0]), len(y_training) / (2 * counts[1])]

    print("class weights: ", class_weights)

    print("Base accuracy: {:.1f}%".format(np.sum(y_training) / (len(y_training)) * 100))
    print("Number of total events: ", len(y_training))
    print("positive events; ", np.sum(y_training))

    # rescale input
    for i in range(ary_features.shape[1]):
        ary_features[:, i] = (ary_features[:, i] - np.mean(ary_features[:, i])) / np.std(ary_features[:, i])

    # split samples into training and validation pool
    x_train, x_valid, y_train, y_valid = train_test_split(x_training, y_training, test_size=0.2, random_state=42)

    # create model
    epochs = 20
    batch_size = 256
    dropout_rate = 0.2
    lr = 1e-3
    initializer = tf.keras.initializers.random_normal()
    loss_function = "binary_crossentropy"

    # define model
    print("Start training")
    print("shape input: ", x_train.shape[1])

    # build DNN model
    model = keras.models.Sequential()
    model.add(tf.keras.layers.Dense(16, input_dim=x_train.shape[1], activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer='adam',
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', dtype=None, threshold=0.5)])

    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), verbose=1, epochs=epochs,
                        batch_size=batch_size)

    # plot model performance
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mse = history.history["binary_accuracy"]
    val_mse = history.history["val_binary_accuracy"]

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
    plt.show()

    # evaluate model
    y_pred = model.predict(x_test)
    fast_AUC_ROC_curve(y_pred, y_test)
    threshold = 0.83

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

    print("Efficiency: {:.1f}%".format(efficiency * 100))
    print("Purity: {:.1f}%".format(purity * 100))
    print("Accuracy: {:.1f}".format((TP + TN) / (TP + TN + FP + FN) * 100))
    print("TP: {} | TN: {} | FP: {} | FN: {}".format(TP, TN, FP, FN))


########################################################################################################################

# generate_training_data(n=None)
network_training()
