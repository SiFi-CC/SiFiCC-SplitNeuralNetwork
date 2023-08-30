import numpy as np
import os
import pickle as pkl
import json
import tensorflow as tf

import dataset

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv3D, MaxPool3D, Flatten

from SiFiCCNN.analysis import fastROCAUC, metrics
from SiFiCCNN.utils.plotter import plot_history_classifier, \
    plot_score_distribution, \
    plot_roc_curve, \
    plot_efficiencymap, \
    plot_sp_distribution, \
    plot_pe_distribution, \
    plot_2dhist_ep_score, \
    plot_2dhist_sp_score


def setupModel(nFilter,
               activation="relu",
               n_out=1,
               activation_out="sigmoid",
               dropout=0.0):
    X_in = Input(shape=(20, 6, 36, 2))

    x = Conv3D(filters=nFilter,
               kernel_size=(3, 2, 3),
               padding="same",
               activation="relu")(X_in)

    x = Conv3D(filters=nFilter * 2,
               kernel_size=(3, 2, 3),
               padding="same",
               activation="relu")(x)

    x = Conv3D(filters=nFilter * 4,
               kernel_size=(3, 2, 3),
               padding="same",
               activation="relu")(x)

    x = MaxPool3D(pool_size=(2, 1, 2), padding="same")(x)
    x = Flatten()(x)
    x = Dense(nFilter * 2, activation=activation)(x)
    x = Dense(nFilter, activation=activation)(x)

    if dropout > 0:
        x = Dropout(dropout)(x)

    out = Dense(n_out, activation=activation_out)(x)

    model = Model(inputs=[X_in], outputs=out)

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
    if epoch < 120:
        return 1e-3
    if epoch < 130:
        return 5e-4
    if epoch < 140:
        return 1e-4
    return 1e-3


def main():
    # defining hyper parameters
    nFilter = 32
    activation = "relu"
    n_out = 1
    activation_out = "sigmoid"
    dropout = 0.0

    batch_size = 64
    nEpochs = 5

    trainsplit = 0.6
    valsplit = 0.2

    RUN_NAME = "CNN3DConvSiPM"
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
    #DATASET_0MM = "GraphSiPM_OptimisedGeometry_BP0mm_2e10protons_taggingv3"
    #DATASET_5MM = "GraphSiPM_OptimisedGeometry_BP5mm_4e9protons_taggingv3"

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
    for file in [DATASET_CONT]:
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
        for file in [DATASET_CONT]:
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
    # setup generator for training
    loader_train = dataset.DenseSiPM(name=dataset_name,
                                     batch_size=batch_size,
                                     slicing="train",
                                     shuffle=True)

    loader_valid = dataset.DenseSiPM(name=dataset_name,
                                     batch_size=batch_size,
                                     slicing="valid",
                                     shuffle=True)

    # build tensorflow model
    tf_model = setupModel(**modelParameter)
    print(tf_model.summary())

    # callbacks
    l_callbacks = [tf.keras.callbacks.LearningRateScheduler(lr_scheduler)]

    # set class-weights
    # Class-weights are set by the dataset itself

    history = tf_model.fit(loader_train,
                           epochs=nEpochs,
                           steps_per_epoch=loader_train.steps_per_epoch,
                           validation_data=loader_valid,
                           validation_steps=loader_valid.steps_per_epoch,
                           verbose=1,
                           callbacks=[l_callbacks])

    os.chdir(path)
    # save model
    print("Saving model at: ", RUN_NAME + "_classifier.keras")
    tf_model.save(RUN_NAME + "_classifier.tf")
    # save training history (not needed tbh)
    with open(RUN_NAME + "_classifier_history" + ".hst", 'wb') as f_hist:
        pkl.dump(history.history, f_hist)
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
    tf_model = tf.keras.models.load_model(RUN_NAME + "_classifier.keras")

    # predict test dataset
    os.chdir(path + dataset_name + "/")

    # load dataset
    loader_eval = dataset.DenseSiPM(name=dataset_name,
                                    batch_size=32,
                                    slicing="",
                                    shuffle=False)

    y_scores = tf_model.predict(loader_eval)
    y_true = loader_eval.y()
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
                       y_sp=loader_eval.sp,
                       figure_name="efficiencymap")
    plot_sp_distribution(ary_sp=loader_eval.sp,
                         ary_score=y_scores,
                         ary_true=y_true,
                         figure_name="sp_distribution")
    plot_pe_distribution(ary_pe=loader_eval.pe,
                         ary_score=y_scores,
                         ary_true=y_true,
                         figure_name="pe_distribution")
    plot_2dhist_sp_score(sp=loader_eval.sp,
                         y_score=y_scores,
                         y_true=y_true,
                         figure_name="2dhist_sp_score")
    plot_2dhist_ep_score(pe=loader_eval.pe,
                         y_score=y_scores,
                         y_true=y_true,
                         figure_name="2dhist_pe_score")


if __name__ == "__main__":
    main()
