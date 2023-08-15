import tensorflow as tf


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
