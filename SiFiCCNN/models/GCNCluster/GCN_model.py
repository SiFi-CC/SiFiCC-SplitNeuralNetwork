from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Dropout
from spektral.layers import GCNConv, GlobalSumPool


class GCN(Model):

    def __init__(self, n_hidden, n_labels):
        super().__init__()
        self.graph_conv = GCNConv(n_hidden)
        self.pool = GlobalSumPool()
        self.dropout = Dropout(0.1)
        self.dense = Dense(n_labels, 'sigmoid')

    def call(self, inputs):
        out = self.graph_conv(inputs)
        # out = self.graph_conv(out)
        out = self.dropout(out)
        out = self.pool(out)
        out = self.dense(out)

        return out
