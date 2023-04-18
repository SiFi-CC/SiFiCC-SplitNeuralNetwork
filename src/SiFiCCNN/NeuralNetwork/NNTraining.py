def train_classifier(NeuralNetwork,
                     DataCluster,
                     verbose=0,
                     epochs=50,
                     batch_size=256):
    # update run settings
    NeuralNetwork.epochs = epochs
    NeuralNetwork.batch_size = batch_size

    # load normalization into Neural Network Class
    ary_mean, ary_std = DataCluster.get_standardization()
    DataCluster.standardize(ary_mean, ary_std)
    NeuralNetwork.norm_mean = ary_mean
    NeuralNetwork.norm_std = ary_std

    if verbose == 1:
        print("\n# Training statistics: ")
        print("Feature dimension: ({} ,{})".format(DataCluster.features.shape[0], DataCluster.features.shape[1]))
        print("")
        # print(NeuralNetwork.model.summary)

    # calling training method
    NeuralNetwork.train(DataCluster.x_train(),
                        DataCluster.y_train(),
                        DataCluster.w_train(),
                        DataCluster.x_valid(),
                        DataCluster.y_valid())

    # save model
    NeuralNetwork.save()


def train_regression_energy(NeuralNetwork,
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

    # load normalization into Neural Network Class
    ary_mean, ary_std = DataCluster.get_standardization()
    DataCluster.standardize(ary_mean, ary_std)
    NeuralNetwork.norm_mean = ary_mean
    NeuralNetwork.norm_std = ary_std

    if verbose == 1:
        print("\n# Training statistics: ")
        print("Feature dimension: ({} ,{})".format(DataCluster.features.shape[0], DataCluster.features.shape[1]))
        print("")
        # print(NeuralNetwork.model.summary)

    # calling training method
    NeuralNetwork.train(DataCluster.x_train(),
                        DataCluster.y_train(),
                        DataCluster.w_train(),
                        DataCluster.x_valid(),
                        DataCluster.y_valid())

    # save model
    NeuralNetwork.save()


def train_regression_position(NeuralNetwork,
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

    # load normalization into Neural Network Class
    ary_mean, ary_std = DataCluster.get_standardization()
    DataCluster.standardize(ary_mean, ary_std)
    NeuralNetwork.norm_mean = ary_mean
    NeuralNetwork.norm_std = ary_std

    if verbose == 1:
        print("\n# Training statistics: ")
        print("Feature dimension: ({} ,{})".format(DataCluster.features.shape[0], DataCluster.features.shape[1]))
        print("")
        # print(NeuralNetwork.model.summary)

    # calling training method
    NeuralNetwork.train(DataCluster.x_train(),
                        DataCluster.y_train(),
                        DataCluster.w_train(),
                        DataCluster.x_valid(),
                        DataCluster.y_valid())

    # save model
    NeuralNetwork.save()


def train_regression_theta(NeuralNetwork,
                           DataCluster,
                           verbose=0,
                           epochs=50,
                           batch_size=256):
    # set regression
    DataCluster.update_targets_theta()
    DataCluster.update_indexing_positives()

    # update run settings
    NeuralNetwork.epochs = epochs
    NeuralNetwork.batch_size = batch_size

    # load normalization into Neural Network Class
    ary_mean, ary_std = DataCluster.get_standardization()
    DataCluster.standardize(ary_mean, ary_std)
    NeuralNetwork.norm_mean = ary_mean
    NeuralNetwork.norm_std = ary_std

    if verbose == 1:
        print("\n# Training statistics: ")
        print("Feature dimension: ({} ,{})".format(DataCluster.features.shape[0], DataCluster.features.shape[1]))
        print("")
        # print(NeuralNetwork.model.summary)

    # calling training method
    NeuralNetwork.train(DataCluster.x_train(),
                        DataCluster.y_train(),
                        DataCluster.w_train(),
                        DataCluster.x_valid(),
                        DataCluster.y_valid())

    # save model
    NeuralNetwork.save()
