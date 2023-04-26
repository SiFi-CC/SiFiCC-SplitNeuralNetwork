import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from mpl_toolkits.axes_grid1 import make_axes_locatable

# update matplotlib parameter for bigger font size
plt.rcParams.update({'font.size': 14})


########################################################################################################################

def get_smap(model, image):
    """
    Take tensorflow and one event, ready as neural network input, calculates saliency map
    from : https://stackoverflow.com/questions/63107141/how-to-compute-saliency-map-using-keras-backend .

    Args:
        model: Tensorflow Neural Network model
        image: neural network input

    Return:
          smap: saliency map, same dimension as <input>
    """

    # conversion from numpy array to tensorflow tensor
    image = tf.convert_to_tensor(image)
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(prediction, image)

    # convert to numpy
    gradient = gradient.numpy()

    # normalize between 0 and 1
    min_val, max_val = np.min(gradient), np.max(gradient)
    smap = (gradient - min_val) / (max_val - min_val + keras.backend.epsilon())

    return smap


def smap_plot(smap, image, title, file_name):
    """
    Generation of matplotlib plot of the saliency map

    Args:
         smap: (numpy array, dim=(10,10): Saliency map
         title: (string): custom string for title
         file_name: (string): file name for the saved .png file
    """
    # TODO: update to variable cluster number

    fig, axs = plt.subplots(1, 2)
    axs[0].set_title(title)
    axs[0].set_xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      labels=["No. fibers", "Timestamp", "Energy", "Pos X", "Pos Y", "Pos Z", "Unc. Energy",
                              "Unc. Pos X", "Unc. Pos Y", "Unc. Pos Z"],
                      rotation=90)
    axs[0].set_yticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      labels=["S1", "S2", "S3", "S4", "A1", "A2", "A2", "A4", "A5", "A6"])
    im1 = axs[0].imshow(smap, vmin=0.0, vmax=1.0, cmap="Reds")
    axs[1].set_xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      labels=["No. fibers", "Timestamp", "Energy", "Pos X", "Pos Y", "Pos Z", "Unc. Energy",
                              "Unc. Pos X", "Unc. Pos Y", "Unc. Pos Z"],
                      rotation=90)
    axs[1].set_yticks([])
    im2 = axs[1].imshow(image, cmap="viridis")

    # color bars
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')

    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')

    plt.tight_layout()
    plt.savefig(file_name + ".png")
