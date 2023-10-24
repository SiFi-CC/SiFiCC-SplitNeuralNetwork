import numpy as np
import os
import pickle as pkl
import json
import tensorflow as tf

from analysis.EdgeConvResNetSiPM import dataset

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Input

from spektral.layers import EdgeConv, GlobalMaxPool
from spektral.data.loaders import DisjointLoader

from SiFiCCNN.utils.layers import EdgeConvResNetBlock, ReZero

from SiFiCCNN.analysis import fastROCAUC, metrics
from SiFiCCNN.utils.plotter import plot_history_classifier, \
    plot_score_distribution, \
    plot_roc_curve, \
    plot_efficiencymap, \
    plot_sp_distribution, \
    plot_pe_distribution, \
    plot_2dhist_ep_score, \
    plot_2dhist_sp_score


def setupModel(F=5,
               nFilter=32,
               activation="relu",
               n_out=1,
               activation_out="sigmoid",
               dropout=0.0):
    X_in = Input(shape=(F,))
    A_in = Input(shape=(None,), sparse=True)
    I_in = Input(shape=(), dtype=tf.int64)

    x = EdgeConv(channels=nFilter)([X_in, A_in])

    # additional layer with skip connections
    x1 = EdgeConvResNetBlock(*[x, A_in], nFilter)
    x2 = EdgeConvResNetBlock(*[x1, A_in], nFilter)
    x3 = EdgeConvResNetBlock(*[x2, A_in], nFilter)
    x_concat = Concatenate()([x1, x2, x3])

    x = GlobalMaxPool()([x_concat, I_in]) 

    if dropout > 0:
        x = Dropout(dropout)(x)

    x = Dense(nFilter * 8, activation=activation)(x)

    out = Dense(n_out, activation=activation_out)(x)

    model = Model(inputs=[X_in, A_in, I_in], outputs=out)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    if n_out == 1:
        loss = "binary_crossentropy"
        list_metrics = ["Precision", "Recall"]
    else:
        loss = "mean_absolute_error"
        list_metrics = ["mean_absolute_error"]
    model.compile(optimizer=optimizer, loss=loss, metrics=list_metrics)

    return model


def lr_scheduler(epoch):
    if epoch < 70:
        return 1e-3
    if epoch < 80:
        return 5e-4
    if epoch < 90:
        return 1e-4
    return 1e-5


def main():
    # defining hyper parameters
    nFilter = 64
    activation = "relu"
    n_out = 1
    activation_out = "sigmoid"
    dropout = 0.0

    batch_size = 64
    nEpochs = 10

    trainsplit = 0.6
    valsplit = 0.2

    RUN_NAME = "EdgeConvResNetSiPM"
    do_training = True
    do_evaluate = True

    # create dictionary for model and training parameter
    modelParameter = {"nFilter": nFilter,
                      "activation": activation,
                      "n_out": n_out,
                      "activation_out": activation_out,
                      "dropout": dropout}

    # Datasets used
    # Training file used for classification and regression training
    # Generated via an input generator, contain one Bragg-peak position
    DATASET_CONT = "GraphSiPM_OptimisedGeometry_4to1_Continuous_2e10protons_simv4"
    DATASET_0MM = "GraphSiPM_OptimisedGeometry_4to1_BP0mm_4e9protons_simv4"
    DATASET_5MM = "GraphSiPM_OptimisedGeometry_4to1_BP5mm_4e9protons_simv4"
    DATASET_m5MM = "GraphSiPM_OptimisedGeometry_4to1_BPm5mm_4e9protons_simv4"
    DATASET_10MM = "GraphSiPM_OptimisedGeometry_4to1_BP10mm_4e9protons_simv4"

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
    for file in [DATASET_CONT, DATASET_0MM, DATASET_5MM, DATASET_m5MM, DATASET_10MM]:
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
        for file in [DATASET_CONT, DATASET_0MM, DATASET_5MM, DATASET_m5MM, DATASET_10MM]:
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
    data = dataset.GraphSiPM(name=dataset_name, adj_arg="binary")

    # build tensorflow model
    tf_model = setupModel(**modelParameter)
    # print(tf_model.summary())

    # callbacks
    l_callbacks = [tf.keras.callbacks.LearningRateScheduler(lr_scheduler)]

    # set class-weights
    class_weights = data.get_classweight_dict()

    # generate disjoint loader from dataset
    idx1 = int(trainsplit * len(data))
    idx2 = int((trainsplit + valsplit) * len(data))
    dataset_tr = data[:idx1]
    dataset_va = data[idx1:idx2]
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
    print("Saving model at: ", RUN_NAME + "_classifier.tf")
    tf_model.save(RUN_NAME + "_classifier.tf")
    # save training history (not needed tbh)
    with open(RUN_NAME + "_classifier_history" + ".hst", 'wb') as f_hist:
        pkl.dump(history.history, f_hist)
    # save norm
    np.save(RUN_NAME + "_classifier" + "_norm_x", data.norm_x)
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

    # load tensorflow model
    # Custom layers have to be stated to load accordingly
    tf_model = tf.keras.models.load_model(RUN_NAME + "_classifier.tf",
                                          custom_objects={"EdgeConv": EdgeConv,
                                                          "GlobalMaxPool": GlobalMaxPool,
                                                          "ReZero": ReZero})

    norm_x = np.load(RUN_NAME + "_classifier_norm_x.npy")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = "binary_crossentropy"
    list_metrics = ["Precision", "Recall"]
    tf_model.compile(optimizer=optimizer,
                     loss=loss,
                     metrics=list_metrics)

    # predict test dataset
    os.chdir(path + dataset_name + "/")

    # load dataset
    data = dataset.GraphSiPM(name=dataset_name,
                                edge_atr=True,
                                adj_arg="binary",
                                norm_x=norm_x)

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