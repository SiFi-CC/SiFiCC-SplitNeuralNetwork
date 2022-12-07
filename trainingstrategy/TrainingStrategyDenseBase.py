def train_strategy(SiFiCCNN, DataCluster):
    # set class weights as sample weights
    DataCluster.set_classweights()

    # TODO: remove
    print(DataCluster.weights[:10])

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
