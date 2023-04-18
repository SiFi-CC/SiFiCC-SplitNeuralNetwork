import numpy as np
import pickle as pkl
from tensorflow import keras


# TODO: update this so this is modifiable
def lr_scheduler(epoch):
    if epoch < 50:
        return 1e-3
    if epoch < 75:
        return 5e-4
    if epoch < 100:
        return 1e-4
    return 1e-5


class NeuralNetwork:

    def __init__(self, model, model_name, model_tag="", epochs=10, batch_size=128, verbose=1):
        # Tensorflow model
        self.model_name = model_name
        self.model_tag = model_tag
        self.model = model
        self.history = {}

        # params
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

        self.norm_mean = []
        self.norm_std = []

    def train(self, x_train, y_train, x_weights, x_valid, y_valid):
        # train model
        l_callbacks = [keras.callbacks.LearningRateScheduler(lr_scheduler), ]

        history = self.model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                                 sample_weight=x_weights,
                                 verbose=self.verbose,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 callbacks=[l_callbacks])
        # update history
        self.append_history(history.history)

    def predict(self, x_data):
        return self.model.predict(x_data)

    def save(self, save_history=True):
        # save model
        str_save = self.model_name
        if self.model_tag != "":
            str_save += "_" + self.model_tag

        print("Saving model at: ", str_save + ".h5")
        self.model.save(str_save + ".h5")

        # save history
        if save_history:
            with open(str_save + ".hst", 'wb') as f_hist:
                pkl.dump(self.history, f_hist)

        # save normalization as npz file
        with open(str_save + "_norm.npz", 'wb') as file:
            np.savez_compressed(file,
                                NORM_MEAN=np.array(self.norm_mean, dtype=np.float32),
                                NORM_STD=np.array(self.norm_std, dtype=np.float32))

    def load(self, load_history=True):
        # load model
        str_load = self.model_name
        if self.model_tag != "":
            str_load += "_" + self.model_tag

        self.model.load_weights(str_load + ".h5")

        # save history
        if load_history:
            with open(str_load + ".hst", 'rb') as f_hist:
                self.history = pkl.load(f_hist)

        # load normalization
        npz_norm = np.load(str_load + "_norm.npz")
        self.norm_mean = npz_norm["NORM_MEAN"]
        self.norm_std = npz_norm["NORM_STD"]

    def append_history(self, history):
        if self.history is None or self.history == {}:
            self.history = history
        else:
            for key in self.history.keys():
                if key in history:
                    self.history[key].extend(history[key])
