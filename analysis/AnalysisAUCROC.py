def analysis(SiFiCCNN, DataCluster, MetaData=None):
    import os
    from fastROCAUC import fastROCAUC

    # save results in txt file
    dir_main = os.getcwd()
    dir_results = dir_main + "/results/" + SiFiCCNN.model_name + SiFiCCNN.model_tag
    if not os.path.isdir(dir_results):
        os.mkdir(dir_results)

    # predict test set
    y_pred = SiFiCCNN.predict(DataCluster.x_test())
    y_true = DataCluster.y_test()

    fastROCAUC(y_pred, y_true, save_fig=dir_results + "/ROCAUC.png")
