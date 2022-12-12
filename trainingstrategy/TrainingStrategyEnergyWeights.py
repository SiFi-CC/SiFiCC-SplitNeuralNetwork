def train_strategy(SiFiCCNN, DataCluster):
    # update training-valid ration to increase validation set
    DataCluster.p_train = 0.6
    DataCluster.p_valid = 0.2

    # set weigts
    DataCluster.weights *= DataCluster.get_classweights()
    DataCluster.weights *= DataCluster.get_energyweights()

    # standardize input
    DataCluster.standardize()

    # train model
    SiFiCCNN.train(DataCluster.x_train(),
                   DataCluster.y_train(),
                   DataCluster.w_train(),
                   DataCluster.x_valid(),
                   DataCluster.y_valid())

    # save model
    SiFiCCNN.save()
