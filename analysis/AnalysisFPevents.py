def analysis(SiFiCCNN, DataCluster, MetaData=None):
    import numpy as np
    import os
    import matplotlib.pyplot as plt

    # predict test set
    y_scores = SiFiCCNN.predict(DataCluster.x_test())
    y_scores = np.reshape(y_scores, newshape=(len(y_scores),))
    y_true = DataCluster.y_test()
    y_pred = np.zeros(shape=(len(y_true, )))

    # set threshold
    threshold = 0.6

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(y_pred)):
        # apply prediction threshold
        if y_scores[i] >= threshold:
            y_pred[i] = 1
        else:
            y_pred[i] = 0

        if y_pred[i] == 1 and y_true[i] == 1:
            tp += 1
        if y_pred[i] == 1 and y_true[i] == 0:
            fp += 1
        if y_pred[i] == 0 and y_true[i] == 0:
            tn += 1
        if y_pred[i] == 0 and y_true[i] == 1:
            fn += 1

    # event type histogram
    ary_eventtype = MetaData.simulated_event_type()[DataCluster.idx_test()]
    ary_fp_eventtype = [ary_eventtype[i] for i in range(len(ary_eventtype)) if (y_pred[i] == 1 and y_true[i] == 0)]
    bins = np.arange(0.5, 7.5, 1.0)
    plt.figure()
    plt.xlabel("MCSimulatedEventType")
    plt.ylabel("counts")
    plt.hist(ary_fp_eventtype, bins=bins, histtype=u"step", color="black")
    plt.tight_layout()
    plt.savefig("eventtype_dist.png")