def train_strategy(SiFiCCNN, DataCluster, load_model=False):
    # update training-valid ration to increase validation set
    DataCluster.p_train = 0.6
    DataCluster.p_valid = 0.2

    # set class weights as sample weights
    DataCluster.weights *= DataCluster.get_classweights()

    # standardize input
    DataCluster.standardize()

    SiFiCCNN.epochs = 20
    SiFiCCNN.batch_size = 256

    if not load_model:
        # train model
        SiFiCCNN.train(DataCluster.x_train(),
                       DataCluster.y_train(),
                       DataCluster.w_train(),
                       DataCluster.x_valid(),
                       DataCluster.y_valid())

        # save model
        SiFiCCNN.save()

    else:
        SiFiCCNN.load()
