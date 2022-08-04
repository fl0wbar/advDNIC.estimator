from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops


class MaskedConv2D(Layer):
    """
    MaskedConv2D : tf.keras Layer for Masked 2D Convolutions for Autoregressive Ops
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="VALID",
        mask_type="B",
        mask_n_channels=3,
        data_format=None,
        activation=None,
        use_bias=None,
        kernel_initializer="glorot_uniform",
        bias_initializer=None,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(MaskedConv2D, self).__init__(**kwargs)
        self.rank = 2
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, self.rank, "kernel_size"
        )
        self.strides = conv_utils.normalize_tuple(strides, self.rank, "strides")
        self.padding = conv_utils.normalize_padding(padding)
        self.mask_type = mask_type
        self.mask_n_channels = mask_n_channels
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        self.mask = None

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError(
                "The channel dimension of the inputs should be" "defined. Found 'None'"
            )
        input_dim = int(input_shape[channel_axis])
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None

        # Set input spec
        self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim})

        # building kernel masks for convolution
        self.mask = np.ones(kernel_shape)
        assert self.mask.shape[0] == self.mask.shape[1]
        filter_size = self.mask.shape[0]
        filter_center = filter_size / 2

        self.mask[math.ceil(filter_center) :] = 0
        self.mask[math.floor(filter_center) :, math.ceil(filter_center) :] = 0

        mask_op = np.greater_equal if self.mask_type == "A" else np.greater
        for i in range(self.mask_n_channels):
            for j in range(self.mask_n_channels):
                if mask_op(i, j):
                    self.mask[
                        math.floor(filter_center),
                        math.floor(filter_center),
                        i :: self.mask_n_channels,
                        j :: self.mask_n_channels,
                    ] = 0

        self.mask = tf.Variable(self.mask, dtype=tf.float32, name="mask")
        self.mask = tf.stop_gradient(self.mask)

        self.built = True

    def call(self, inputs):
        outputs = backend.conv2d(
            inputs,
            self.kernel * self.mask,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
        )
        if self.use_bias:
            outputs = backend.bias_add(outputs, self.bias, data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == "channels_last":
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                )
                new_space.append(new_dim)
            return tensor_shape.TensorShape(
                [input_shape[0]] + new_space + [self.filters]
            )
        else:
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                )
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0], self.filters] + new_space)

    def get_config(self):
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "mask_type": self.mask_type,
            "data_format": self.data_format,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        return dict(list(config.items()))


# """ Causal Attention """

# class CausalAttention(Layer):
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
