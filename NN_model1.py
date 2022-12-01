import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

layers = keras.layers
from sklearn.model_selection import train_test_split

from fastROCAUC import fastROCAUC
import matplotlib.pyplot as plt

dir_main = os.getcwd()
dir_data = dir_main + "/data/"

########################################################################################################################

# load training data
data_train = np.load(dir_data + "test_train.npz")
data_valid = np.load(dir_data + "test_valid.npz")
data_test = np.load(dir_data + "test_test.npz")

x_train = data_train["features"]
y_train = data_train["targets"]
w_train = data_train["weights"]

x_valid = data_valid["features"]
y_valid = data_valid["targets"]
w_valid = data_valid["weights"]

x_test = data_test["features"]
y_test = data_test["targets"]
w_test = data_test["weights"]

_, counts = np.unique(y_train, return_counts=True)
class_weights = {0: len(y_train) / (2 * counts[0]),
                 1: len(y_train) / (2 * counts[1])}

# create model
epochs = 10
batch_size = 256
dropout_rate = 0.2
lr = 1e-3
initializer = tf.keras.initializers.random_normal()
loss_function = "binary_crossentropy"

# define model
model = keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, input_dim=x_train.shape[1], activation="relu"))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(32, activation="relu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer='adam',
              metrics=["accuracy"])

history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), verbose=1, epochs=epochs,
                    batch_size=batch_size, class_weight=class_weights)

# plot model performance
loss = history.history['loss']
val_loss = history.history['val_loss']
mse = history.history["accuracy"]
val_mse = history.history["val_accuracy"]

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

# evaluate model
y_pred = model.predict(x_test)
fastROCAUC(y_pred, y_test)
threshold = 0.84

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
