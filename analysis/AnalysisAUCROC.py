def analysis(SiFiCCNN, DataCluster, MetaData=None):
    import os
    from fastROCAUC import fastROCAUC

    # predict test set
    y_pred = SiFiCCNN.predict(DataCluster.x_test())
    y_true = DataCluster.y_test()

    fastROCAUC(y_pred, y_true, save_fig="ROCAUC.png")
