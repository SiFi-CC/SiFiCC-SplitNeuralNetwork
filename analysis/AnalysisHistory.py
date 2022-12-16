def analysis(SiFiCCNN, DataCluster, MetaData=None):
    import os
    import matplotlib.pyplot as plt

    # save results in txt file
    dir_main = os.getcwd()
    dir_results = dir_main + "/results/" + SiFiCCNN.model_name + SiFiCCNN.model_tag
    if not os.path.isdir(dir_results):
        os.mkdir(dir_results)

    # plot model performance
    loss = SiFiCCNN.history['loss']
    val_loss = SiFiCCNN.history['val_loss']
    mse = SiFiCCNN.history["accuracy"]
    val_mse = SiFiCCNN.history["val_accuracy"]

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(mse, label="Training", linestyle='--', color="blue")
    ax1.plot(val_mse, label="Validation", linestyle='-', color="red")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid()

    ax2.plot(loss, label="Training", linestyle='--', color="blue")
    ax2.plot(val_loss, label="Validation", linestyle='-', color="red")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.legend()
    ax2.grid()
    plt.tight_layout()
    plt.savefig(dir_results + "/history.png")
