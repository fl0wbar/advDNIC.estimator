import numpy as np
import tensorflow.compat.v1 as tf

""" Kervolutional Operations """


class LinearKernel(tf.keras.layers.Layer):
    def __init__(self):
        super(LinearKernel, self).__init__()

    def call(self, x, w, b):
        out_channels = w.get_shape().as_list()[-1]
        w = tf.reshape(w, (-1, out_channels))
        x = tf.reshape(x, (-1, x.get_shape().as_list()[-1]))
        out = tf.matmul(x, w)
        if b is not None:
            return out + b
        return out


class DifferenceLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(DifferenceLayer, self).__init__()

    def call(self, x, w):
        out_channels = w.get_shape().as_list()[-1]
        w = tf.reshape(w, (-1, out_channels))
        input_shape = x.get_shape().as_list()
        x = tf.reshape(x, (-1, input_shape[1] * input_shape[2], input_shape[-1]))
        x = x[:, :, :, None]
        out = x - w
        return out


class LPNormKernel(DifferenceLayer):
    def __init__(self, p=1):
        super(LPNormKernel, self).__init__()
        self.ord = p

    def call(self, x, w, b):
        out = super(LPNormKernel, self).call(x, w)
        out = tf.norm(out, ord=self.ord, axis=2)
        if b is not None:
            return out + b
        return out


class PolynomialKernel(LinearKernel):
    def __init__(self, cp=1.0, dp=3.0, train_pars=False):
        super(PolynomialKernel, self).__init__()
        self.initial_cp = cp
        self.dp = dp
        self.train_pars = train_pars

    def build(self, input_shape):
        if self.train_pars:
            self.cp = self.add_variable(
                name="cp",
                shape=(),
                initializer=tf.keras.initializers.get("zeros"),
                trainable=True,
            )
            tf.summary.scalar("cp", self.cp)
        else:
            self.cp = self.initial_cp

        self.built = True

    def call(self, x, w, b):
        conv = super(PolynomialKernel, self).call(x, w, None)
        s = conv + self.cp
        out = s ** self.dp
        if b is not None:
            return out + b
        return out


class SigmoidKernel(LinearKernel):
    def __init__(self):
        super(SigmoidKernel, self).__init__()

    def call(self, x, w, b):
        out = super(SigmoidKernel, self).call(x, w, None)
        out = tf.math.tanh(out)
        if b is not None:
            return out + b
        return out


class GaussianKernel(DifferenceLayer):
    def __init__(self, gamma=1.0, train_gamma=False):
        super(GaussianKernel, self).__init__()
        self.initial_gamma = gamma
        self.train_gamma = train_gamma

    def build(self, input_shape):
        if self.train_gamma:
            self.gamma = self.add_variable(
                name="gamma",
                shape=(),
                initializer=tf.keras.initializers.get("ones"),
                trainable=True,
            )
        else:
            self.gamma = self.initial_gamma
        self.built = True

    def call(self, x, w, b):
        diff = super(GaussianKernel, self).call(x, w)
        diff_norm = tf.reduce_sum(tf.square(diff), axis=-2)
        out = tf.exp(-self.gamma * diff_norm)
        if b is not None:
            return out + b
        return out


class KernelConv2D(tf.keras.layers.Conv2D):
    def __init__(
        self,
        filters,
        kernel_size,
        kernel_fn=GaussianKernel,
        strides=(1, 1),
        padding="SAME",
        dilation_rate=(1, 1),
        use_bias=True,
    ):

        super(KernelConv2D, self).__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            use_bias=use_bias,
        )
        self.kernel_fn = kernel_fn

    def call(self, x):
        patches = tf.image.extract_image_patches(
            x,
            sizes=[1, self.kernel_size[0], self.kernel_size[1], 1],
            strides=[1, self.strides[0], self.strides[1], 1],
            padding=self.padding.upper(),
            rates=[1, self.dilation_rate[0], self.dilation_rate[1], 1],
        )
        output = self.kernel_fn(patches, self.kernel, self.bias)
        output_shape = [-1] + patches.get_shape().as_list()[1:3] + [output.shape[-1]]
        output = tf.reshape(output, output_shape)
        return output


def get_kernel(kernel_name, **kwargs):
    if kernel_name == "polynomial":
        return PolynomialKernel(cp=kwargs["cp"], dp=kwargs["dp"])
    elif kernel_name == "gaussian":
        return GaussianKernel(gamma=kwargs["gamma"])
    elif kernel_name == "sigmoid":
        return SigmoidKernel()
    elif kernel_name == "L1":
        return LPNormKernel(p=1)
    elif kernel_name == "L2":
        return LPNormKernel(p=2)
    else:
        return LinearKernel()


""" Dynamic Convolutions """


def dynamic_conv(input, filter, strides=[1, 1, 1], padding="SAME", dilation_rate=None):
    """
    Equivalent to tf.nn.convolution, but filter has additional
    batch dimension. This allows the filter to be a function
    of some input, hence, enabling dynamic convolutions.

    Parameters
    ----------
    input:  A Tensor. Must be one of the following types: float32, float64, int64, int32,
            uint8, uint16, int16, int8, complex64, complex128, qint8, quint8, qint32, half.
            2d case:
            Shape [batch, in_depth, in_height, in_channels].
            3d case:
            Shape [batch, in_depth, in_height, in_width, in_channels].

    filter: A Tensor. Must have the same type as input.
            in_channels must match between input and filter.
            2d case:
            Shape [batch, filter_x, filter_y, in_ch, out_ch].
            3d case:
            Shape [batch, filter_x, filter_y, filter_z, in_ch, out_ch] .

    strides:    A list of ints that has length >= 5. 1-D tensor of length 5.
                The stride of the sliding window for each dimension of input.
                Must have strides[0] = strides[4] = 1.
    padding:    A string from: "SAME", "VALID". The type of padding algorithm to use.

    dilation_rate: Optional.
                Sequence of N ints >= 1. Specifies the filter upsampling/input downsampling rate.
                In the literature, the same parameter is sometimes called input stride or dilation.
                The effective filter size used for the convolution will be
                spatial_filter_shape + (spatial_filter_shape - 1) * (rate - 1),
                obtained by inserting (dilation_rate[i]-1) zeros between consecutive elements of
                the original filter in each spatial dimension i.
                If any value of dilation_rate is > 1, then all values of strides must be 1.

    Returns
    -------
            A Tensor. Has the same type as input.
    """

    assert len(filter.get_shape()) == len(input.get_shape()) + 1
    assert filter.get_shape()[0] == input.get_shape()[0]

    split_inputs = tf.split(input, input.get_shape().as_list()[0], axis=0)
    split_filters = tf.unstack(filter, input.get_shape().as_list()[0], axis=0)

    output_list = []
    for split_input, split_filter in zip(split_inputs, split_filters):
        output_list.append(
            tf.nn.convolution(
                split_input,
                split_filter,
                strides=strides,
                padding=padding,
                dilation_rate=dilation_rate,
            )
        )
    output = tf.concat(output_list, axis=0)
    return output


# """ Causal Attention """

# class CausalAttention(tf.keras.layers.Layer):
#     """
#     This tf.keras layer implements causal attention from the pixelSNAIL paper
#     """
#     def __init__(self, key, mixin, query, downsample=1, use_pos_enc=False,
#                  name="causal_attention", **kwargs):
#         super(CausalAttention, self).__init__(**kwargs)
#         self.key = key
#         self.mixin = mixin
#         self.query = query
#         self.downsample = downsample
#         self.use_pos_enc = use_pos_enc
#         self._block_name = name


