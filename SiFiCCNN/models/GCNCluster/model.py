import pickle as pkl
import tensorflow as tf
from SiFiCCNN.models.GCNCluster.layers import GCNConvResNetBlock

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate
from spektral.layers import GCNConv, ECCConv, GlobalSumPool


class GCNmodel(Model):

    def __init__(self,
                 n_labels,
                 output_activation,
                 dropout=0.0):
        super().__init__()

        self.n_labels = n_labels
        self.output_activation = output_activation
        self.dropout_val = dropout

        self.graph_gcnconv1 = GCNConv(32, activation="relu")
        self.graph_gcnconv2 = GCNConv(64, activation="relu")
        self.graph_eccconv1 = ECCConv(32, activation="relu")
        self.graph_eccconv2 = ECCConv(64, activation="relu")
        self.pool = GlobalSumPool()
        self.dropout = Dropout(dropout)
        self.dense1 = Dense(64, activation="relu")
        self.dense_out = Dense(n_labels, output_activation)
        self.concatenate = Concatenate()

    def call(self, inputs):
        xIn, aIn, eIn, iIn = inputs
        out1 = self.graph_gcnconv1([xIn, aIn])
        out2 = self.graph_gcnconv2([out1, aIn])
        out3 = self.graph_eccconv1([xIn, aIn, eIn])
        out4 = self.graph_eccconv2([out3, aIn, eIn])
        out5 = self.pool([out2, iIn])
        out6 = self.pool([out4, iIn])

        out7 = self.concatenate([out5, out6])
        out8 = self.dense1(out7)
        out9 = self.dropout(out8)
        out_final = self.dense_out(out9)

        return out_final

    def get_config(self):
        return {"n_labels": self.n_labels,
                "output_activation": self.output_activation,
                "dropout": self.dropout_val}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @staticmethod
    def save_history(model_name,
                     history):
        with open(model_name + ".hst", 'wb') as f_hist:
            pkl.dump(history, f_hist)


def lr_scheduler(epoch):
    if epoch < 20:
        return 1e-3
    if epoch < 30:
        return 5e-4
    if epoch < 40:
        return 1e-4
    return 1e-5
