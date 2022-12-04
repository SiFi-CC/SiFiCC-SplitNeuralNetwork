import os
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow import keras

from fastROCAUC import fastROCAUC
import matplotlib.pyplot as plt

dir_main = os.getcwd()
dir_data = dir_main + "/data/"
dir_plots = dir_main + "/plots/"
dir_models = dir_main + "/models/"

########################################################################################################################

input_file = "DNN_input_OptimizedGeometry_BP0mm_2e10protons.npz"
model_file = "DNN_base.h5"
history_file = "DNN_base.hst"

########################################################################################################################

# load test data
npz_data = np.load(dir_data + input_file)
ary_features = npz_data["features"]
ary_targets = npz_data["targets"]
ary_w = npz_data["weights"]

ary_idx_test = npz_data["idx_test"]

# normalization
for i in range(ary_features.shape[1]):
    ary_features[:, i] = (ary_features[:, i] - np.mean(ary_features[:, i])) / np.std(ary_features[:, i])

# set test data
x_test = ary_features[ary_idx_test]
y_test = ary_targets[ary_idx_test]

# load neural network model
model = keras.models.load_model(dir_models + model_file)
# load neural network history
with open(dir_models + history_file, 'rb') as f_hist:
    history = pkl.load(f_hist)

# plot model performance
loss = history['loss']
val_loss = history['val_loss']
mse = history["accuracy"]
val_mse = history["val_accuracy"]

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
plt.show()

# predict test set
y_pred = model.predict(x_test)

# run ROC curve and AUC score analysis
fastROCAUC(y_pred, y_test)

# best optimal threshold
threshold = 0.84

# evaluate important metrics
TP = 0
FP = 0
TN = 0
FN = 0

for i in range(len(y_pred)):
    # apply prediction threshold
    if y_pred[i] >= threshold:
        y_pred[i] = 1
    else:
        y_pred[i] = 0

    if y_pred[i] == 1 and y_test[i] == 1:
        TP += 1
    if y_pred[i] == 1 and y_test[i] == 0:
        FP += 1
    if y_pred[i] == 0 and y_test[i] == 0:
        TN += 1
    if y_pred[i] == 0 and y_test[i] == 1:
        FN += 1

if (TP + FN) == 0:
    efficiency = 0
else:
    efficiency = TP / (TP + FN)
if (TP + FP) == 0:
    purity = 0
else:
    purity = TP / (TP + FP)

print("\nAccuracy: {:.1f}".format((TP + TN) / (TP + TN + FP + FN) * 100))
print("Efficiency: {:.1f}%".format(efficiency * 100))
print("Purity: {:.1f}%".format(purity * 100))
print("TP: {} | TN: {} | FP: {} | FN: {}".format(TP, TN, FP, FN))


