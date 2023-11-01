import numpy as np
import os
import pickle as pkl
import json
import tensorflow as tf

from analysis.EdgeConvResNetSiPM import dataset

from spektral.layers import EdgeConv, GlobalMaxPool
from spektral.data.loaders import DisjointLoader

from analysis.EdgeConvResNetSiPM.ClassificationEdgeConvResNetSiPM import setupModel
from SiFiCCNN.utils.layers import EdgeConvResNetBlock, ReZero

from SiFiCCNN.utils.plotter import plot_history_regression, plot_energy_error,\
    plot_energy_resolution


def lr_scheduler(epoch):
    if epoch < 50:
        return 1e-3
    if epoch < 60:
        return 5e-4
    if epoch < 70:
        return 1e-4
    return 1e-3


def main():
    # defining hyper parameters
    nFilter = 32
    activation = "relu"
    n_out = 2
    activation_out = "relu"
    dropout = 0.0

    batch_size = 64
    nEpochs = 50

    trainsplit = 0.6
    valsplit = 0.2

    RUN_NAME = "EdgeConvResNetSiPM"
    do_training = False
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
    DATASET_0MM = "GraphSiPM_OptimisedGeometry_4to1_0mm_4e9protons_simv4"
    DATASET_5MM = "GraphSiPM_OptimisedGeometry_4to1_5mm_4e9protons_simv4"
    DATASET_m5MM = "GraphSiPM_OptimisedGeometry_4to1_minus5mm_4e9protons_simv4"
    DATASET_10MM = "GraphSiPM_OptimisedGeometry_4to1_10mm_4e9protons_simv4"

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
    data = dataset.GraphSiPM(name=dataset_name,
                             adj_arg="binary",
                             p_only=True,
                             reg_type="Energy")
    print("DATASET LENGTH: ", len(data))

    # build tensorflow model
    tf_model = setupModel(**modelParameter)
    print(tf_model.summary())

    # callbacks
    l_callbacks = [tf.keras.callbacks.LearningRateScheduler(lr_scheduler)]

    # set class-weights
    # DISABLED: REGRESSION DOES NOT NEED CLASS WEIGHTS
    # class_weights = data.get_classweight_dict()

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
                           verbose=1,
                           callbacks=[l_callbacks])

    os.chdir(path)
    # save model
    print("Saving model at: ", RUN_NAME + "_regressionEnergy.tf")
    tf_model.save(RUN_NAME + "_regressionEnergy.tf")
    # save training history (not needed tbh)
    with open(RUN_NAME + "_regressionEnergy_history" + ".hst", 'wb') as f_hist:
        pkl.dump(history.history, f_hist)
    # save norm
    np.save(RUN_NAME + "_regressionEnergy" + "_norm_x", data.norm_x)
    # save model parameter as json
    with open(RUN_NAME + "_regressionEnergy_parameter.json", "w") as json_file:
        json.dump(modelParameter, json_file)

    # plot training history
    plot_history_regression(history.history, RUN_NAME + "_history_regressionEnergy")


def evaluate(dataset_name,
             RUN_NAME,
             path):
    os.chdir(path)
    # load model, model parameter, norm, history
    with open(RUN_NAME + "_regressionEnergy_parameter.json", "r") as json_file:
        modelParameter = json.load(json_file)

    # load tensorflow model
    # Custom layers have to be stated to load accordingly
    tf_model = tf.keras.models.load_model(RUN_NAME + "_regressionEnergy.tf",
                                          custom_objects={"EdgeConv": EdgeConv,
                                                          "GlobalMaxPool": GlobalMaxPool,
                                                          "ReZero": ReZero})

    norm_x = np.load(RUN_NAME + "_regressionEnergy_norm_x.npy")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = "mean_absolute_error"
    list_metrics = ["mean_absolute_error"]
    tf_model.compile(optimizer=optimizer,
                     loss=loss,
                     metrics=list_metrics)

    # load model history and plot
    with open(RUN_NAME + "_regressionEnergy_history" + ".hst", 'rb') as f_hist:
        history = pkl.load(f_hist)
    plot_history_regression(history, RUN_NAME + "_history_regressionEnergy")

    # predict test dataset
    os.chdir(path + dataset_name + "/")

    # load dataset
    data = dataset.GraphSiPM(name=dataset_name,
                             edge_atr=True,
                             adj_arg="binary",
                             norm_x=norm_x,
                             p_only=True,
                             reg_type="Energy")

    loader_test = DisjointLoader(data,
                                 batch_size=64,
                                 epochs=1,
                                 shuffle=False)

    y_true = []
    y_pred = []
    for batch in loader_test:
        inputs, target = batch
        p = tf_model(inputs, training=False)
        y_true.append(target)
        y_pred.append(p.numpy())
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    y_true = np.reshape(y_true, newshape=(y_true.shape[0], 2))
    y_pred = np.reshape(y_pred, newshape=(y_pred.shape[0], 2))

    # evaluate model:
    plot_energy_error(y_pred, y_true, "error_regression_energy")
    plot_energy_resolution(y_pred, y_true, "resolution_regression_energy")

    np.save("energy_pred", y_pred)
    np.save("energy_true", y_true)


if __name__ == "__main__":
    main()
