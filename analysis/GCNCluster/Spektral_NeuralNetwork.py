import numpy as np
import pickle as pkl
from tensorflow import keras
import spektral


# TODO: update this so this is modifiable
def lr_scheduler(epoch):
    if epoch < 25:
        return 1e-3
    if epoch < 40:
        return 5e-4
    if epoch < 50:
        return 1e-4
    return 1e-5


class NeuralNetwork:

    def __init__(self,
                 model,
                 model_name,
                 class_weights,
                 epochs=10,
                 batch_size=32,
                 verbose=1):

        self.model = model
        self.model_name = model_name
        self.history = {}
        self.class_weights = class_weights

        # params
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

        self.norm_mean = []
        self.norm_std = []

        # compile model
        model.compile("adam", "binary_crossentropy",
                      metrics=[keras.metrics.Precision(),
                               keras.metrics.Recall(),
                               keras.metrics.BinaryAccuracy()])

    def train(self, loader):
        # train model
        l_callbacks = [keras.callbacks.LearningRateScheduler(lr_scheduler), ]

        history = self.model.fit(loader.load(),
                                 steps_per_epoch=loader.steps_per_epoch,
                                 class_weight=self.class_weights,
                                 verbose=self.verbose,
                                 epochs=self.epochs,
                                 callbacks=[l_callbacks])
        # update history
        self.append_history(history.history)

    def predict(self, loader):
        output = []
        step = 0

        while step < loader.steps_per_epoch:
            step += 1
            input, target = loader.__next__()
            pred = self.model(input, training=False)
            output.append(pred)

            if step == loader.steps_per_epoch:
                output = np.concatenate(output)
                output = np.reshape(output, newshape=(output.shape[0], ))
                return output

    def y(self, loader):
        output = []
        step = 0

        while step < loader.steps_per_epoch:
            step += 1
            input, target = loader.__next__()
            output.append(target)

            if step == loader.steps_per_epoch:
                output = np.concatenate(output)
                output = np.reshape(output, newshape=(output.shape[0], ))
                return output

    def save(self, save_history=True):
        # save model
        str_save = self.model_name

        print("Saving model at: ", str_save + ".h5")
        self.model.save(str_save + ".h5")

        # save history
        if save_history:
            with open(str_save + ".hst", 'wb') as f_hist:
                pkl.dump(self.history, f_hist)

    def load(self, load_history=True):
        # load model
        str_load = self.model_name
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
