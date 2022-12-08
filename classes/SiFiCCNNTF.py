import os
import pickle as pkl


class SiFiCCNNTF:

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

    def train(self, x_train, y_train, x_weights, x_valid, y_valid):
        # train model
        history = self.model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                                 sample_weight=x_weights,
                                 verbose=self.verbose,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size)
        # update history
        self.append_history(history.history)

    def predict(self, x_data):
        return self.model.predict(x_data)

    def save(self, save_history=True):
        # save model
        str_save = os.getcwd() + "/models/" + self.model_name
        if self.model_tag != "":
            str_save += "_" + self.model_tag

        print("Saving model at: ", str_save + ".h5")
        self.model.save(str_save + ".h5")

        # save history
        if save_history:
            with open(str_save + ".hst", 'wb') as f_hist:
                pkl.dump(self.history, f_hist)

    def load(self, load_history=True):
        # load model
        str_load = os.getcwd() + "/models/" + self.model_name
        if self.model_tag != "":
            str_load += "_" + self.model_tag

        self.model.load(str_load + ".h5")

        # save history
        if load_history:
            with open(str_load + ".hst", 'rb') as f_hist:
                self.history = pkl.load(f_hist)

    def append_history(self, history):
        if self.history is None or self.history == {}:
            self.history = history
        else:
            for key in self.history.keys():
                if key in history:
                    self.history[key].extend(history[key])
