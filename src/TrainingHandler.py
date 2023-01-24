from src import NPZParser
from src import Plotter


def train_clas(NeuralNetwork,
               DataCluster,
               verbose=0,
               epochs=50,
               batch_size=256):
    # update run settings
    NeuralNetwork.epochs = epochs
    NeuralNetwork.batch_size = batch_size

    if verbose == 1:
        print("\n# Training statistics: ")
        print("Feature dimension: ({} ,{})".format(DataCluster.features.shape[0], DataCluster.features.shape[1]))
        print("")
        # print(NeuralNetwork.model.summary)

    NeuralNetwork.train(DataCluster.x_train(),
                        DataCluster.y_train(),
                        DataCluster.w_train(),
                        DataCluster.x_valid(),
                        DataCluster.y_valid())

    # get evaluation of training performance
    Plotter.plot_history_classifier(NeuralNetwork,
                                    NeuralNetwork.model_name + "_" +
                                    NeuralNetwork.model_tag + "_history_training_classifier")

    # save model
    NeuralNetwork.save()


def train_regE(NeuralNetwork,
               DataCluster,
               verbose=0,
               epochs=50,
               batch_size=256):

    # set regression
    DataCluster.update_targets_energy()
    DataCluster.update_indexing_positives()

    # update run settings
    NeuralNetwork.epochs = epochs
    NeuralNetwork.batch_size = batch_size

    if verbose == 1:
        print("\n# Training statistics: ")
        print("Feature dimension: ({} ,{})".format(DataCluster.features.shape[0], DataCluster.features.shape[1]))
        print("")
        # print(NeuralNetwork.model.summary)

    NeuralNetwork.train(DataCluster.x_train(),
                        DataCluster.y_train(),
                        DataCluster.w_train(),
                        DataCluster.x_valid(),
                        DataCluster.y_valid())

    # get evaluation of training performance
    Plotter.plot_history_regression(NeuralNetwork,
                                    NeuralNetwork.model_name + "_" + NeuralNetwork.model_tag + "_history_training")

    # save model
    NeuralNetwork.save()


def train_regP(NeuralNetwork,
               DataCluster,
               verbose=0,
               epochs=50,
               batch_size=256):

    # set regression
    DataCluster.update_targets_position()
    DataCluster.update_indexing_positives()

    # update run settings
    NeuralNetwork.epochs = epochs
    NeuralNetwork.batch_size = batch_size

    if verbose == 1:
        print("\n# Training statistics: ")
        print("Feature dimension: ({} ,{})".format(DataCluster.features.shape[0], DataCluster.features.shape[1]))
        print("")
        # print(NeuralNetwork.model.summary)

    NeuralNetwork.train(DataCluster.x_train(),
                        DataCluster.y_train(),
                        DataCluster.w_train(),
                        DataCluster.x_valid(),
                        DataCluster.y_valid())

    # get evaluation of training performance
    Plotter.plot_history_regression(NeuralNetwork,
                                    NeuralNetwork.model_name + "_" + NeuralNetwork.model_tag + "_history_training")

    # save model
    NeuralNetwork.save()
