def train_schedule(data_cluster,
                   meta_data,
                   nn_classifier,
                   nn_regression1,
                   nn_regression2,
                   load_classifier=False,
                   load_regression1=False,
                   load_regression2=False):

    # update training-valid ration to increase validation set
    data_cluster.p_train = 0.7
    data_cluster.p_valid = 0.1
    data_cluster.p_test = 0.2

    ######################
    # Classifier Training
    ######################

    # set class weights as sample weights
    data_cluster.weights *= data_cluster.get_classweights()

    # standardize input
    data_cluster.standardize()

    nn_classifier.epochs = 30
    nn_classifier.batch_size = 256

    if not load_classifier:
        # train model
        nn_classifier.train(data_cluster.x_train(),
                            data_cluster.y_train(),
                            data_cluster.w_train(),
                            data_cluster.x_valid(),
                            data_cluster.y_valid())
        # save model
        nn_classifier.save()

    else:
        nn_classifier.load()
    """
    #############################
    # Regression Energy Training
    #############################

    nn_regression1.epochs = 30
    nn_regression1.batch_size = 256

    if not load_regression1:
        # train model
        nn_regression1.train(data_cluster.x_train_reg(),
                             data_cluster.y_train_reg1(),
                             None,
                             data_cluster.x_valid_reg(),
                             data_cluster.y_valid_reg1())
        # save model
        nn_regression1.save()

    else:
        nn_regression1.load()

    ###############################
    # Regression Position Training
    ###############################

    nn_regression2.epochs = 30
    nn_regression2.batch_size = 256

    if not load_regression2:
        # train model
        nn_regression2.train(data_cluster.x_train_reg(),
                             data_cluster.y_train_reg2(),
                             None,
                             data_cluster.x_valid_reg(),
                             data_cluster.y_valid_reg2())
        # save model
        nn_regression2.save()

    else:
        nn_regression2.load()
    """
    ##################################
    # Full Neural Network Predictions
    ##################################

    from src import NetworkEvaluation
    NetworkEvaluation.classifier_evaluation(data_cluster, meta_data, nn_classifier)
    #NetworkEvaluation.regression_evaluation(data_cluster, nn_regression1, nn_regression2)

    NetworkEvaluation.export_mlem_cutbased(nn_classifier, data_cluster)




