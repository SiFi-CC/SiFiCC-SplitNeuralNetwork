from src import NPZParser
from src import Plotter


def train_clas(NeuralNetwork, npz_file, verbose=0):
    """

    Args:
        NeuralNetwork: NeuralNetwork object
        npz_file: path to npz file containing training data

    Returns:

    """

    # load npz file into DataCluster object
    data_cluster = NPZParser.parse(npz_file)

    # update training-valid ration to increase validation set
    data_cluster.p_train = 0.7
    data_cluster.p_valid = 0.1
    data_cluster.p_test = 0.2

    # set class weights as sample weights
    data_cluster.weights *= data_cluster.get_classweights()

    # standardize input
    data_cluster.standardize()

    # update run settings
    NeuralNetwork.epochs = 500
    NeuralNetwork.batch_size = 256

    if verbose == 1:
        print("\n# Training statistics: ")
        print("NPZ_FILE: ", npz_file)
        print("Feature dimension: ({} ,{})".format(data_cluster.features.shape[0], data_cluster.features.shape[1]))
        print("")
        # print(NeuralNetwork.model.summary)

    NeuralNetwork.train(data_cluster.x_train(),
                        data_cluster.y_train(),
                        data_cluster.w_train(),
                        data_cluster.x_valid(),
                        data_cluster.y_valid())

    # get evaluation of training performance
    Plotter.plot_history_classifier(NeuralNetwork,
                                    NeuralNetwork.model_name + "_" + NeuralNetwork.model_tag + "_history_training_classifier")

    # save model
    NeuralNetwork.save()


def train_regEnergy(NeuralNetwork, npz_file, verbose=0):
    # load npz file into DataCluster object
    data_cluster = NPZParser.parse(npz_file)

    # update training-valid ration to increase validation set
    data_cluster.p_train = 0.7
    data_cluster.p_valid = 0.1
    data_cluster.p_test = 0.2

    # set class weights as sample weights
    data_cluster.weights *= data_cluster.get_classweights()

    # set regression
    data_cluster.update_targets_energy()
    data_cluster.update_indexing_positives()

    # standardize input
    data_cluster.standardize()

    # update run settings
    NeuralNetwork.epochs = 30
    NeuralNetwork.batch_size = 256

    if verbose == 1:
        print("\n# Training statistics: ")
        print("NPZ_FILE: ", npz_file)
        print("Feature dimension: ({} ,{})".format(data_cluster.features.shape[0], data_cluster.features.shape[1]))
        print("")
        # print(NeuralNetwork.model.summary)

    NeuralNetwork.train(data_cluster.x_train(),
                        data_cluster.y_train(),
                        data_cluster.w_train(),
                        data_cluster.x_valid(),
                        data_cluster.y_valid())

    # get evaluation of training performance
    Plotter.plot_history_regression(NeuralNetwork,
                                    NeuralNetwork.model_name + "_" + NeuralNetwork.model_tag + "_history_training")

    # save model
    NeuralNetwork.save()


def train_regPosition(NeuralNetwork, npz_file, verbose=0):
    # load npz file into DataCluster object
    data_cluster = NPZParser.parse(npz_file)

    # update training-valid ration to increase validation set
    data_cluster.p_train = 0.7
    data_cluster.p_valid = 0.1
    data_cluster.p_test = 0.2

    # set class weights as sample weights
    data_cluster.weights *= data_cluster.get_classweights()

    # set regression
    data_cluster.update_targets_position()
    data_cluster.update_indexing_positives()

    # standardize input
    data_cluster.standardize()

    # update run settings
    NeuralNetwork.epochs = 30
    NeuralNetwork.batch_size = 256

    if verbose == 1:
        print("\n# Training statistics: ")
        print("NPZ_FILE: ", npz_file)
        print("Feature dimension: ({} ,{})".format(data_cluster.features.shape[0], data_cluster.features.shape[1]))
        print("")
        # print(NeuralNetwork.model.summary)

    NeuralNetwork.train(data_cluster.x_train(),
                        data_cluster.y_train(),
                        data_cluster.w_train(),
                        data_cluster.x_valid(),
                        data_cluster.y_valid())

    # get evaluation of training performance
    Plotter.plot_history_regression(NeuralNetwork,
                                    NeuralNetwork.model_name + "_" + NeuralNetwork.model_tag + "_history_training")

    # save model
    NeuralNetwork.save()