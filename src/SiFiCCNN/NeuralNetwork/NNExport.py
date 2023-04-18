import numpy as np

from src.SiFiCCNN.ImageReconstruction import IRExport


def export_prediction_npz(ary_nn_pred,
                          file_name):
    with open(file_name + ".npz", 'wb') as f_output:
        np.savez_compressed(f_output, NN_PRED=ary_nn_pred)


def export_prediction_cc6(ary_nn_pred,
                          file_name,
                          use_theta="DOTVEC",
                          veto=True,
                          verbose=1):
    IRExport.export_CC6(ary_e=ary_nn_pred[1],
                        ary_p=ary_nn_pred[2],
                        ary_ex=ary_nn_pred[3],
                        ary_ey=ary_nn_pred[4],
                        ary_ez=ary_nn_pred[5],
                        ary_px=ary_nn_pred[6],
                        ary_py=ary_nn_pred[7],
                        ary_pz=ary_nn_pred[8],
                        ary_theta=ary_nn_pred[9],
                        filename=file_name,
                        use_theta=use_theta,
                        veto=veto,
                        verbose=verbose)
