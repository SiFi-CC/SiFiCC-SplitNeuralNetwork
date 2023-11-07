import tensorflow as tf
import spektral as sp
import numpy as np

from sklearn.neighbors import kneighbors_graph
from SiFiCCNN.utils.tf_utils import split_disjoint


class ReZero(tf.keras.layers.Layer):
    r"""ReZero layer based on the paper
    'ReZero is All You Need: Fast Convergence at Large Depth' (arXiv:2003.04887v2)

    This layer computes:
        x_(i+1) = x_i + alpha_i*F(x_i)
    where alpha_i is the trainable residual weight, initialized with zero

    Inputs:
      List of layers with same shape.

    Returns:
      Output of ReZero layer with same shape as individual inputs

    Example:

      ReZero layer can by used in ResNet block
      #convolution layer with some prior layer. x serves as short cut path
      x = Conv2D(32, activation="relu", use_bias=True)(someOtherLayer)

      #two convolution layers in ResNetBlock
      fx = Conv2D(32, activation="relu", use_bias=True)(x)
      fx = Conv2D(32, use_bias=False)(fx)

      #addition like in normal ResNet block, but weighted with trainable residual weight
      x = ReZero()([x, fx])
      #final activation function of this ReZero/ResNet block
      x = Activation('relu')(x)
    """

    def __init__(self, **kwargs):
        super(ReZero, self).__init__(**kwargs)

    def build(self, input_shape):
        # create residual weight
        assert isinstance(input_shape, list)
        self.residualWeight = self.add_weight(name="residualWeight",
                                              shape=(1,),
                                              initializer=tf.keras.initializers.Zeros(),
                                              trainable=True)
        super(ReZero, self).build(input_shape)

    def call(self, inputs):
        assert isinstance(inputs, list)
        inputX, inputFx = inputs
        return inputX + self.residualWeight * inputFx


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
        x = tf.keras.layers.Dense(convBlockShape[-1],
                                  activation="relu")(xInput)
        return x
    else:
        return xInput


class DynamicGraphUpdate(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DynamicGraphUpdate, self).__init__(**kwargs)

    def call(self, inputs):
        assert isinstance(inputs, list)
        X, A, I = inputs

        # grab indices from I
        I = tf.cast(I, tf.dtypes.int32)
        i_n = tf.math.bincount(I)
        i_n_cum = tf.concat([tf.math.cumsum(i_n)], 0)

        sub_Xs = tf.split(X, i_n_cum, axis=0)
        print(sub_Xs)

        return A


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
                       E,
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
                           kernel_initializer=kernel_initializer)([x, A, E])
    fx = sp.layers.ECCConv(n_filter,
                           use_bias=False,
                           kernel_initializer=kernel_initializer)([fx, A, E])

    # adjust the number of filters in the short cut path by eventually applying a Conv1D with kernel size 1
    x = adjustChannelSize(x, fx)

    # apply rezero layer
    x = ReZero()([x, fx])

    # apply activation function at the end
    x = tf.keras.layers.Activation(activation)(x)

    return x


def EdgeConvResNetBlock(x,
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
    fx = sp.layers.EdgeConv(channels=n_filter,
                            activation=conv_activation,
                            use_bias=True,
                            kernel_initializer=kernel_initializer)([x, A])
    fx = sp.layers.EdgeConv(channels=n_filter,
                            use_bias=False,
                            kernel_initializer=kernel_initializer)([fx, A])

    # adjust the number of filters in the short cut path by eventually applying a Conv1D with kernel size 1
    x = adjustChannelSize(x, fx)

    # apply rezero layer
    x = ReZero()([x, fx])

    # apply activation function at the end
    x = tf.keras.layers.Activation(activation)(x)

    return x


def GATConvResNetBlock(x,
                       A,
                       n_filter=64,
                       attn_heads=8,
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
    fx = sp.layers.GATConv(n_filter,
                           attn_heads=attn_heads,
                           concat_heads=True,
                           use_bias=True,
                           kernel_initializer=kernel_initializer)([x, A])
    fx = sp.layers.GATConv(n_filter,
                           attn_heads=attn_heads,
                           concat_heads=True,
                           use_bias=False,
                           kernel_initializer=kernel_initializer)([fx, A])

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
