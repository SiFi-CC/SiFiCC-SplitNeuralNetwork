import os
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Dropout
from spektral.layers import GCNConv, GlobalSumPool

dir_main = os.getcwd()
dir_root = dir_main + "/root_files/"
dir_npz = dir_main + "/npz_files/"
dir_results = dir_main + "/results/"
dir_datasets = dir_main + "/datasets/"

# ----------------------------------------------------------------------------------------------------------------------
# generate dataset
"""
root_cont = RootParser.Root(dir_main + RootFiles.OptimisedGeometry_Continuous_2e10protons_withTimestamps_local)
IGGraphConvolutionCluster.gen_SiFiCCCluster(root_cont)
"""
# ----------------------------------------------------------------------------------------------------------------------
# Dataset

dataset = ClusterDataset.SiFiCCCluster(name="OptimisedGeometry_Continuous_2e10protons_GraphConvCluster",
                                       dataset_path=dir_datasets)
print(dataset.n_labels)
"""
from spektral.datasets import TUDataset

dataset = TUDataset('PROTEINS')
print(dataset.n_labels)
print(dataset[0].x)
"""
# ----------------------------------------------------------------------------------------------------------------------
# GNN example with spektral

# load test dataset
from spektral.data import BatchLoader

loader = BatchLoader(dataset, batch_size=16)


# ----------------------------------------------------------------------------------------------------------------------

class MyFirstGNN(Model):

    def __init__(self, n_hidden, n_labels):
        super().__init__()
        self.graph_conv = GCNConv(n_hidden)
        self.pool = GlobalSumPool()
        self.dropout = Dropout(0.2)
        self.dense = Dense(n_labels, 'sigmoid')

    def call(self, inputs):
        out = self.graph_conv(inputs)
        out = self.graph_conv(inputs)
        out = self.dropout(out)
        out = self.pool(out)
        out = self.dense(out)

        return out


model = MyFirstGNN(32, dataset.n_labels)
model.compile("adam", "binary_crossentropy",
              metrics=[keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.BinaryAccuracy()])

model.fit(loader.load(), steps_per_epoch=loader.steps_per_epoch, epochs=50)
