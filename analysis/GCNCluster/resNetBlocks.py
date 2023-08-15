import tensorflow as tf
import spektral as sp

from .rezero import ReZero


def adjustChannelSize(xInput, fxInput):
    r"""
    Expand channels of shortcut to match residual.
    If number of filters do not match an additional Conv1D
    with kernel size 1 is applied to the shortcut.

    Args:
        xInput: input layer to ResNetBlock with certain number of filters
        fxInput: output layer of convolution block of ResNetBlock with potentially different number
                 of filters
    Returns:
        A keras layer.
    """

    # get shapes of input layers
    inputShape = tf.keras.backend.int_shape(xInput)
    convBlockShape = tf.keras.backend.int_shape(fxInput)

    # check if number of filters are the same
    equalChannels = inputShape[-1] == convBlockShape[-1]

    # 1 X 1 conv if shape is different. Else identity.
    if not equalChannels:
        x = tf.keras.layers.Conv1D(convBlockShape[-1],
                                   kernel_size=1,
                                   strides=1,
                                   padding="same",
                                   use_bias=False)(xInput)
        return x
    else:
        return xInput


def GCNConvResNetBlock(x,
                       A,
                       n_filter=64,
                       activation="relu",
                       conv_activation="relu",
                       kernel_initializer="glorot_uniform"):
    r"""
    ResNetBlock implementation used in the master thesis
    'Vertex Reconstruction for Neutrino Events in the Double Chooz Experiment using Graph Neural
    Networks'
    Utilizes graph convolutions (Kipf & Welling) instead of conventional layers.Conv...
    Instead of the standard short cut the ReZero approach is used.

    Args:
        x: A keras layers object which is compatible with
           sp.layers.GCNConv with output shape [batch, nodes, features]
        A: Adjacency matrix (Modified Laplacian of shape [batch, nodes, nodes])
           e.g. in form of an Input layer:
           A = tf.keras.layers.Input(tensor = sp.layers.ops.sp_matrix_to_sp_tensor(sp.utils.gcn_filter(Adj)))
        n_filter: number of convolution filters
        conv_activation: activation function of the first convolution layer in the block
        kernel_initializer: initializer for the weights
        activation: activation function which is applied after adding identity to conv block
    Returns:
        A keras layer. Graph after GCNConvResNetBlock with shape [batch, n_nodes, n_filter]
    """

    # two convolution layers in ResNetBlock
    fx = sp.layers.GCNConv(n_filter,
                           activation=conv_activation,
                           use_bias=True,
                           kernel_initializer=kernel_initializer)([x, A])
    fx = sp.layers.GCNConv(n_filter,
                           use_bias=False,
                           kernel_initializer=kernel_initializer)([fx, A])

    # adjust the number of filters in the short cut path by eventually applying a Conv1D with kernel size 1
    x = adjustChannelSize(x, fx)

    # apply rezero layer
    x = ReZero()([x, fx])

    # apply activation function at the end
    x = tf.keras.layers.Activation(activation)(x)

    return x


def ECCConvResNetBlock(x,
                       A,
                       e,
                       n_filter=64,
                       activation="relu",
                       conv_activation="relu",
                       kernel_initializer="glorot_uniform"):
    r"""
    ResNetBlock implementation used in the master thesis
    'Vertex Reconstruction for Neutrino Events in the Double Chooz Experiment using Graph Neural
    Networks'
    Utilizes graph convolutions (Kipf & Welling) instead of conventional layers.Conv...
    Instead of the standard short cut the ReZero approach is used.

    Args:
        x: A keras layers object which is compatible with
           sp.layers.GCNConv with output shape [batch, nodes, features]
        A: Adjacency matrix (Modified Laplacian of shape [batch, nodes, nodes])
           e.g. in form of an Input layer:
           A = tf.keras.layers.Input(tensor = sp.layers.ops.sp_matrix_to_sp_tensor(sp.utils.gcn_filter(Adj)))
        n_filter: number of convolution filters
        conv_activation: activation function of the first convolution layer in the block
        kernel_initializer: initializer for the weights
        activation: activation function which is applied after adding identity to conv block
    Returns:
        A keras layer. Graph after GCNConvResNetBlock with shape [batch, n_nodes, n_filter]
    """

    # two convolution layers in ResNetBlock
    fx = sp.layers.ECCConv(n_filter,
                           activation=conv_activation,
                           use_bias=True,
                           kernel_initializer=kernel_initializer)([x, A, e])
    fx = sp.layers.ECCConv(n_filter,
                           use_bias=False,
                           kernel_initializer=kernel_initializer)([fx, A, e])

    # adjust the number of filters in the short cut path by eventually applying a Conv1D with kernel size 1
    x = adjustChannelSize(x, fx)

    # apply rezero layer
    x = ReZero()([x, fx])

    # apply activation function at the end
    x = tf.keras.layers.Activation(activation)(x)

    return x



def resNetBlocks(implementation, **kwargs):
    if implementation == "GCNResNet":
        return GCNConvResNetBlock(**kwargs)
