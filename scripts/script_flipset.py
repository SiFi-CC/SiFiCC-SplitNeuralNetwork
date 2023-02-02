import numpy as np

def random_flip(npz_file):

    npz_data = np.load(npz_file)
    ary_features = npz_data["features"]
    ary_targets_clas = npz_data["targets_clas"]
    ary_targets_reg1 = npz_data["targets_reg1"]
    ary_targets_reg2 = npz_data["targets_reg2"]
    ary_w = npz_data["weights"]
    ary_meta = npz_data["META"]

    for i in range(ary_features.shape[1]):
        if not np.random.choice([True, False]):
            continue
        else:
            ary_features[i, [4, 13, 22, 31, 40, 49, 58, 67]] *= -1.0
            ary_targets_reg2[i, [2, 5]] *= -1.0
            ary_meta[i, 2] *= -1

    # save final output file
    with open("Flipped_dataset" + ".npz", 'wb') as f_output:
        np.savez_compressed(f_output,
                            features=ary_features,
                            targets_clas=ary_targets_clas,
                            targets_reg1=ary_targets_reg1,
                            targets_reg2=ary_targets_reg2,
                            weights=ary_w,
                            META=ary_meta)

