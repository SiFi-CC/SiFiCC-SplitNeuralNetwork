import numpy as np
import os
import pickle as pkl
import json
import tensorflow as tf

import dataset

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate

from spektral.layers import GCNConv, ECCConv, GlobalSumPool
from spektral.data.loaders import DisjointLoader
from spektral.transforms import GCNFilter

from .resNetBlocks import GCNConvResNetBlock, ECCConvResNetBlock

from SiFiCCNN.analysis import fastROCAUC, metrics
from SiFiCCNN.utils.plotter import plot_history_classifier, \
    plot_score_distribution, \
    plot_roc_curve, \
    plot_efficiencymap, \
    plot_sp_distribution, \
    plot_pe_distribution, \
    plot_2dhist_ep_score, \
    plot_2dhist_sp_score


class NNmodel(Model):

    def __init__(self,
                 nOutput,
                 OutputActivation,
                 nNodes,
                 dropout=0.0):
        super().__init__()

        self.nOutput = nOutput
        self.OutputActivation = OutputActivation
        self.dropout_val = dropout
        self.nNodes = nNodes

        self.graph_gcnconv1 = GCNConv(nNodes, activation="relu")
        self.graph_gcnconv2 = GCNConv(int(nNodes * 2), activation="relu")
        self.graph_eccconv1 = ECCConv(nNodes, activation="relu")
        self.graph_eccconv2 = ECCConv(int(nNodes * 2), activation="relu")
        self.pool = GlobalSumPool()
        self.dropout = Dropout(dropout)
        self.dense1 = Dense(nNodes, activation="relu")
        self.dense2 = Dense(int(nNodes / 2), activation="relu")
        self.dense_out = Dense(nOutput, OutputActivation)
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
        out9 = self.dense2(out8)
        out10 = self.dropout(out9)
        out_final = self.dense_out(out10)

        return out_final


class ResNetModel(Model):

    def __init__(self,
                 nOutput,
                 OutputActivation,
                 nNodes,
                 dropout=0.0):
        super().__init__()

        self.nOutput = nOutput
        self.OutputActivation = OutputActivation
        self.dropout_val = dropout
        self.nNodes = nNodes

    def call(self, inputs):
        xIn, aIn, eIn, iIn = inputs
        out1 = GCNConv(channels=self.nNodes)([xIn, aIn])
        out1 = GCNConvResNetBlock(*[out1, aIn], n_filter=self.nNodes)
        out1 = GCNConvResNetBlock(*[out1, aIn], n_filter=self.nNodes * 2)
        out2 = ECCConv(channels=self.nNodes)([xIn, aIn, eIn])
        out2 = ECCConvResNetBlock(*[out2, aIn, eIn], n_filter=self.nNodes)
        out2 = ECCConvResNetBlock(*[out2, aIn, eIn], n_filter=self.nNodes * 2)

        out1 = GlobalSumPool()([out1, iIn])
        out2 = GlobalSumPool()([out2, iIn])

        out = Concatenate()([out1, out2])
        out = Dense(self.nNodes, activation="relu")(out)
        out = Dense(int(self.nNodes / 2), activation="relu")(out)
        out = Dropout(self.dropout)(out)
        out = Dense(self.nOutput, self.OutputActivation)(out)

        return out


def lr_scheduler(epoch):
    if epoch < 20:
        return 1e-3
    if epoch < 30:
        return 5e-4
    if epoch < 40:
        return 1e-4
    return 1e-5


def main():
    # defining hyper parameters
    dropout = 0.2
    nNodes = 64
    batch_size = 64
    nEpochs = 20

    trainsplit = 0.6
    valsplit = 0.2

    RUN_NAME = "TEST_GCNCluster"
    do_training = True
    do_evaluate = True

    # create dictionary for model and training parameter
    modelParameter = {"nOutput": 1,
                      "OutputActivation": "sigmoid",
                      "dropout": dropout,
                      "nNodes": nNodes}

    # Datasets used
    # Training file used for classification and regression training
    # Generated via an input generator, contain one Bragg-peak position
    DATASET_CONT = "GraphCluster_OptimisedGeometry_Continuous_2e10protons_taggingv3"
    DATASET_0MM = "GraphCluster_OptimisedGeometry_BP0mm_2e10protons_taggingv3"
    DATASET_5MM = "GraphCluster_OptimisedGeometry_BP5mm_4e9protons_taggingv3"

    # go backwards in directory tree until the main repo directory is matched
    path = os.getcwd()
    while True:
        path = os.path.abspath(os.path.join(path, os.pardir))
        if os.path.basename(path) == "SiFiCC-SplitNeuralNetwork":
            break
    path_main = path
    path_results = path_main + "/results/" + RUN_NAME + "/"

    # create subdirectory for run output
    if not os.path.isdir(path_results):
        os.mkdir(path_results)
    for file in [DATASET_CONT, DATASET_0MM, DATASET_5MM]:
        if not os.path.isdir(path_results + "/" + file + "/"):
            os.mkdir(path_results + "/" + file + "/")

    if do_training:
        training(dataset_name=DATASET_CONT,
                 RUN_NAME=RUN_NAME,
                 trainsplit=trainsplit,
                 valsplit=valsplit,
                 batch_size=batch_size,
                 nEpochs=nEpochs,
                 path=path_results,
                 modelParameter=modelParameter)

    if do_evaluate:
        for file in [DATASET_CONT, DATASET_0MM, DATASET_5MM]:
            evaluate(dataset_name=file,
                     RUN_NAME=RUN_NAME,
                     path=path_results)


def training(dataset_name,
             RUN_NAME,
             trainsplit,
             valsplit,
             batch_size,
             nEpochs,
             path,
             modelParameter):
    # load graph dataset
    data = dataset.GraphCluster(name=dataset_name,
                                adj_arg="binary")
    # build model from parameter
    tf_model = NNmodel(**modelParameter)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = "binary_crossentropy"
    list_metrics = ["Precision", "Recall"]
    tf_model.compile(optimizer=optimizer,
                     loss=loss,
                     metrics=list_metrics)
    l_callbacks = [tf.keras.callbacks.LearningRateScheduler(lr_scheduler)]

    # set normalization from training dataset
    # dataset is initialized standardized
    class_weights = data.get_classweight_dict()

    # apply GCN filter
    # generate disjoint loader from dataset
    data.apply(GCNFilter())
    idx1 = int(trainsplit * len(data))
    idx2 = int((trainsplit + valsplit) * len(data))
    dataset_tr = data[:idx1]
    dataset_va = data[idx1:idx2]
    dataset_te = data[idx2:]
    loader_train = DisjointLoader(dataset_tr,
                                  batch_size=batch_size,
                                  epochs=nEpochs)
    loader_valid = DisjointLoader(dataset_va,
                                  batch_size=batch_size)

    history = tf_model.fit(loader_train,
                           epochs=nEpochs,
                           steps_per_epoch=loader_train.steps_per_epoch,
                           validation_data=loader_valid,
                           validation_steps=loader_valid.steps_per_epoch,
                           class_weight=class_weights,
                           verbose=1,
                           callbacks=[l_callbacks])

    os.chdir(path)
    # save model
    print("Saving model at: ", RUN_NAME + "_classifier" + ".h5")
    tf_model.save(RUN_NAME + "_classifier")
    # save training history (not needed tbh)
    with open(RUN_NAME + "_classifier_history" + ".hst", 'wb') as f_hist:
        pkl.dump(history.history, f_hist)
    # save norm
    np.save(RUN_NAME + "_classifier" + "_norm_x", data.norm_x)
    np.save(RUN_NAME + "_classifier" + "_norm_e", data.norm_e)
    # save model parameter as json
    with open(RUN_NAME + "_classifier_parameter.json", "w") as json_file:
        json.dump(modelParameter, json_file)

    # plot training history
    plot_history_classifier(history.history, RUN_NAME + "_history_classifier")


def evaluate(dataset_name,
             RUN_NAME,
             path):
    os.chdir(path)
    # load model, model parameter, norm, history
    with open(RUN_NAME + "_classifier_parameter.json", "r") as json_file:
        modelParameter = json.load(json_file)
    tf_model = tf.keras.models.load_model(RUN_NAME + "_classifier")
    norm_x = np.load(RUN_NAME + "_classifier_norm_x.npy")
    norm_e = np.load(RUN_NAME + "_classifier_norm_e.npy")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = "binary_crossentropy"
    list_metrics = ["Precision", "Recall"]
    tf_model.compile(optimizer=optimizer,
                     loss=loss,
                     metrics=list_metrics)

    # predict test dataset
    os.chdir(path + dataset_name + "/")

    # load dataset
    data = dataset.GraphCluster(name=dataset_name,
                                edge_atr=True,
                                adj_arg="binary",
                                norm_x=norm_x,
                                norm_e=norm_e)

    # apply GCN filter, generate disjoint loaders from dataset
    data.apply(GCNFilter())
    loader_test = DisjointLoader(data,
                                 batch_size=64,
                                 epochs=1,
                                 shuffle=False)

    y_true = []
    y_scores = []
    for batch in loader_test:
        inputs, target = batch
        p = tf_model(inputs, training=False)
        y_true.append(target)
        y_scores.append(p.numpy())
    y_true = np.vstack(y_true)
    y_scores = np.vstack(y_scores)
    y_true = np.reshape(y_true, newshape=(y_true.shape[0],)) * 1
    y_scores = np.reshape(y_scores, newshape=(y_scores.shape[0],))

    # evaluate model:
    #   - ROC analysis
    #   - Score distribution#
    #   - Binary classifier metrics

    _, theta_opt, (list_fpr, list_tpr) = fastROCAUC.fastROCAUC(y_scores,
                                                               y_true,
                                                               return_score=True)
    plot_roc_curve(list_fpr, list_tpr, "rocauc_curve")
    plot_score_distribution(y_scores, y_true, "score_dist")
    metrics.write_metrics_classifier(y_scores, y_true)

    plot_efficiencymap(y_pred=y_scores,
                       y_true=y_true,
                       y_sp=data.sp,
                       figure_name="efficiencymap")
    plot_sp_distribution(ary_sp=data.sp,
                         ary_score=y_scores,
                         ary_true=y_true,
                         figure_name="sp_distribution")
    plot_pe_distribution(ary_pe=data.pe,
                         ary_score=y_scores,
                         ary_true=y_true,
                         figure_name="pe_distribution")
    plot_2dhist_sp_score(sp=data.sp,
                         y_score=y_scores,
                         y_true=y_true,
                         figure_name="2dhist_sp_score")
    plot_2dhist_ep_score(pe=data.pe,
                         y_score=y_scores,
                         y_true=y_true,
                         figure_name="2dhist_pe_score")


if __name__ == "__main__":
    main()
