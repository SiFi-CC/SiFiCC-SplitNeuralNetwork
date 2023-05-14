import numpy as np
import pickle as pkl

import tensorflow as tf


def save_model(model,
               model_name):
    print("Saving model at: ", model_name + ".h5")
    model.save(model_name + ".h5",
               save_weights_only=True)


def save_history(model_name,
                 history):
    # save history
    if save_history:
        with open(model_name + ".hst", 'wb') as f_hist:
            pkl.dump(history, f_hist)


def save_norm(model_name,
              norm_mean,
              norm_std):
    # save normalization as npz file
    with open(model_name + "_norm.npz", 'wb') as file:
        np.savez_compressed(file,
                            NORM_MEAN=np.array(norm_mean,
                                               dtype=np.float32),
                            NORM_STD=np.array(norm_std,
                                              dtype=np.float32))


def load_model(model,
               model_name):
    model.load_weights(model_name + ".h5")
    return model


def load_history(model_name):
    # save history
    if load_history:
        with open(model_name + ".hst", 'rb') as f_hist:
            history = pkl.load(f_hist)

    return history


def load_norm(model_name):
    # load normalization
    npz_norm = np.load(model_name + "_norm.npz")
    norm_mean = npz_norm["NORM_MEAN"]
    norm_std = npz_norm["NORM_STD"]
    return norm_mean, norm_std


def lr_scheduler(epoch):
    if epoch < 20:
        return 1e-3
    if epoch < 30:
        return 5e-4
    if epoch < 40:
        return 1e-4
    return 1e-5
