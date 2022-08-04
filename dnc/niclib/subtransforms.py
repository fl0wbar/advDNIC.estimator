"""
    Contains definitions for creating analysis and synthesis transformexp
    based on EfficientNet architecture
    Mingxing Tan, Quoc V. Le
    EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
    ICML'19, https://arxiv.org/abs/1905.11946
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import tensorflow_compression as tfc


GlobalParams = collections.namedtuple(
    "GlobalParams",
    [
        "use_batchnorm",
        "batch_norm_momentum",
        "batch_norm_epsilon",
        "data_format",
        "width_coefficient",
        "depth_coefficient",
        "depth_divisor",
        "min_depth",
        "drop_connect_rate",
        "relu_fn",
        "stem_conv_type",
    ],
)
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

batchnorm = tf.layers.BatchNormalization
# batchnorm = utils.TpuBatchNormalization  # TPU-specific requirement.

BlockArgs = collections.namedtuple(
    "BlockArgs",
    [
        "kernel_size",
        "num_repeat",
        "input_filters",
        "output_filters",
        "expand_ratio",
        "id_skip",
        "strides",
        "se_ratio",
        "conv_type",
    ],
)
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def custom_conv_kernel_initializer(shape, dtype=None, partition_info=None):
    """Initialization for convolutional kernels.

    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas here we use a normal distribution. Similarly,
    tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
    a corrected standard deviation.

    Args:
        shape: shape of variable
        dtype: dtype of variable
        partition_info: unused

    Returns:
        an initialization for the variable
    """
    del partition_info
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random_normal(shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def drop_connect(inputs, is_training, drop_connect_rate):
    """Apply drop connect."""
    if not is_training:
        return inputs

    # Compute keep_prob
    # TODO(tanmingxing): add support for training progress.
    keep_prob = 1.0 - drop_connect_rate

    # Compute drop_connect tensor
    batch_size = tf.shape(inputs)[0]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.div(inputs, keep_prob) * binary_tensor
    return output


class DepthwiseConv2D(tf.keras.layers.DepthwiseConv2D, tf.layers.Layer):
    """Wrap keras DepthwiseConv2D to tf.layers."""

    pass


class MBSignalConvBlock(tf.keras.layers.Layer):
    """A class of MBSignalConv: Mobile Inverted Residual Bottleneck using tfc.SignalConv2D
    
    Attributes:
        endpoints (dict): a list of internel tensors for feature extraction
    """

    def __init__(self, block_args, global_params):
        """Initializes a MBSignalConv block.

        Args:
            block_args: BlockArgs, arguments to create a Block.
            global_params: GlobalParams, a set of global parameters.
        """
        super(MBSignalConvBlock, self).__init__()
        self._block_args = block_args
        self._batch_norm_momentum = global_params.batch_norm_momentum
        self._batch_norm_epsilon = global_params.batch_norm_epsilon
        self._data_format = global_params.data_format
        self._use_batchnorm = global_params.use_batchnorm
        if self._data_format == "channels_first":
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]

        self._relu_fn = global_params.relu_fn or tf.nn.swish
        self._has_se = (
            (self._block_args.se_ratio is not None)
            and (self._block_args.se_ratio > 0)
            and (self._block_args.se_ratio <= 1)
        )

        self.endpoints = None

        # Builds the block accordings to arguments.
        self._build()

    def block_args(self):
        return self._block_args

    def _build(self):
        """Builds block according to the arguments."""
        filters = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            # Expansion phase:
            self._expand_conv = tfc.SignalConv2D(
                filters,
                kernel_support=(1, 1),
                kernel_initializer=custom_conv_kernel_initializer,
                strides_down=1,
                corr=True,
                padding="same_zeros",
                data_format=self._data_format,
                use_bias=False,
                activation=tfc.GDN(data_format="channels_last"),
            )
            self._bn0 = batchnorm(
                axis=self._channel_axis,
                momentum=self._batch_norm_momentum,
                epsilon=self._batch_norm_epsilon,
            )

        kernel_size = self._block_args.kernel_size
        # Depth-wise convolution phase:
        self._depthwise_conv = DepthwiseConv2D(
            [kernel_size, kernel_size],
            strides=self._block_args.strides,
            depthwise_initializer=custom_conv_kernel_initializer,
            padding="same",
            data_format=self._data_format,
            use_bias=False,
        )
        self._bn1 = batchnorm(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon,
        )

        if self._has_se:
            num_reduced_filters = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio)
            )
            # Squeeze and Excitation layer.
            self._se_reduce = tfc.SignalConv2D(
                num_reduced_filters,
                kernel_support=(1, 1),
                kernel_initializer=custom_conv_kernel_initializer,
                strides_down=1,
                corr=True,
                padding="same_zeros",
                data_format=self._data_format,
                use_bias=True,
                activation=tfc.GDN(data_format="channels_last"),
            )
            self._se_expand = tfc.SignalConv2D(
                filters,
                kernel_support=(1, 1),
                kernel_initializer=custom_conv_kernel_initializer,
                strides_down=1,
                corr=True,
                padding="same_zeros",
                data_format=self._data_format,
                use_bias=True,
            )

        # Output phase:
        filters = self._block_args.output_filters
        self._project_conv = tfc.SignalConv2D(
            filters,
            kernel_support=(1, 1),
            kernel_initializer=custom_conv_kernel_initializer,
            strides_down=1,
            corr=True,
            padding="same_zeros",
            data_format=self._data_format,
            use_bias=False,
        )
        self._bn2 = batchnorm(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon,
        )

    def _call_se(self, input_tensor):
        """Call Squeeze and Excitation layer.

        Args:
          input_tensor: Tensor, a single input tensor for Squeeze/Excitation layer.

        Returns:
          A output tensor, which should have the same shape as input.
        """
        se_tensor = tf.reduce_mean(input_tensor, self._spatial_dims, keepdims=True)
        se_tensor = self._se_expand(self._se_reduce(se_tensor))
        tf.logging.info(
            "Built Squeeze and Excitation with tensor shape: %s" % (se_tensor.shape)
        )
        return tf.sigmoid(se_tensor) * input_tensor

    def call(self, inputs, training=True, drop_connect_rate=None):
        """Implementation of call().

        Args:
          inputs: the inputs tensor.
          training: boolean, whether the model is constructed for training.
          drop_connect_rate: float, between 0 to 1, drop connect rate.

        Returns:
          A output tensor.
        """
        tf.logging.info(
            "MBSignalConvBlock-/Input: %s shape: %s" % (inputs.name, inputs.shape)
        )
        if self._block_args.expand_ratio != 1:
            if self._use_batchnorm:
                x = self._bn0(self._expand_conv(inputs), training=training)
            else:
                x = self._expand_conv(inputs)
        else:
            x = inputs
        tf.logging.info("MBSignalConvBlock-/Expand: %s shape: %s" % (x.name, x.shape))

        if self._use_batchnorm:
            x = self._relu_fn(self._bn1(self._depthwise_conv(x), training=training))
        else:
            x = self._relu_fn(self._depthwise_conv(x))

        tf.logging.info("MBSignalConvBlock-/DWConv: %s shape: %s" % (x.name, x.shape))

        if self._has_se:
            with tf.variable_scope("se"):
                x = self._call_se(x)

        self.endpoints = {"expansion_output": x}

        if self._use_batchnorm:
            x = self._bn2(self._project_conv(x), training=training)
        else:
            x = self._project_conv(x)
        if self._block_args.id_skip:
            if (
                all(s == 1 for s in self._block_args.strides)
                and self._block_args.input_filters == self._block_args.output_filters
            ):
                # only apply drop_connect if skip presents.
                if drop_connect_rate:
                    x = drop_connect(x, training, drop_connect_rate)
                x = tf.add(x, inputs)
        tf.logging.info("MBSignalConvBlock-/Project: %s shape: %s" % (x.name, x.shape))
        return x


class MBSignalConvBlockWithoutDepthwise(MBSignalConvBlock):
    """MBConv-like block without depthwise convolution and squeeze-and-excite."""

    def _build(self):
        """Builds block according to the arguments."""
        filters = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            # Expansion phase:
            self._expand_conv = tfc.SignalConv2D(
                filters,
                kernel_support=(3, 3),
                kernel_initializer=custom_conv_kernel_initializer,
                strides_down=1,
                corr=True,
                padding="same_zeros",
                data_format=self._data_format,
                use_bias=False,
                activation=tfc.GDN(data_format="channels_last"),
            )
            self._bn0 = batchnorm(
                axis=self._channel_axis,
                momentum=self._batch_norm_momentum,
                epsilon=self._batch_norm_epsilon,
            )

        # Output phase:
        filters = self._block_args.output_filters
        self._project_conv = tfc.SignalConv2D(
            filters,
            kernel_support=(1, 1),
            kernel_initializer=custom_conv_kernel_initializer,
            strides_down=1,
            corr=True,
            padding="same_zeros",
            data_format=self._data_format,
            use_bias=False,
            activation=tfc.GDN(data_format="channels_last"),
        )
        self._bn1 = batchnorm(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon,
        )

    def call(self, inputs, training=True, drop_connect_rate=None):
        """Implementation of call().

        Args:
          inputs: the inputs tensor.
          training: boolean, whether the model is constructed for training.
          drop_connect_rate: float, between 0 to 1, drop connect rate.

        Returns:
          A output tensor.
        """
        tf.logging.info(
            "MBSignalConvBlock_ndw-/Input: %s shape: %s" % (inputs.name, inputs.shape)
        )
        if self._block_args.expand_ratio != 1:
            if self._use_batchnorm:
                x = self._bn0(self._expand_conv(inputs), training=training)
            else:
                x = self._expand_conv(inputs)
        else:
            x = inputs
        tf.logging.info(
            "MBSignalConvBlock_ndw-/Expand: %s shape: %s" % (x.name, x.shape)
        )

        self.endpoints = {"expansion_output": x}

        if self._use_batchnorm:
            x = self._bn1(self._project_conv(x), training=training)
        else:
            x = self._project_conv(x)

        if self._block_args.id_skip:
            if (
                all(s == 1 for s in self._block_args.strides)
                and self._block_args.input_filters == self._block_args.output_filters
            ):
                # only apply drop_connect if skip presents.
                if drop_connect_rate:
                    x = drop_connect(x, training, drop_connect_rate)
                x = tf.add(x, inputs)
        tf.logging.info(
            "MBSignalConvBlock_ndw-/Project: %s shape: %s" % (x.name, x.shape)
        )
        return x


class MBConvBlock(tf.keras.layers.Layer):
    """A class of MBConv: Mobile Inverted Residual Bottleneck.

      Attributes:
        endpoints: dict. A list of internal tensors.
      """

    def __init__(self, block_args, global_params):
        """Initializes a MBConv block.

        Args:
          block_args: BlockArgs, arguments to create a Block.
          global_params: GlobalParams, a set of global parameters.
        """
        super(MBConvBlock, self).__init__()
        self._block_args = block_args
        self._batch_norm_momentum = global_params.batch_norm_momentum
        self._batch_norm_epsilon = global_params.batch_norm_epsilon
        self._data_format = global_params.data_format
        self._use_batchnorm = global_params.use_batchnorm
        if self._data_format == "channels_first":
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]

        self._relu_fn = global_params.relu_fn or tf.nn.swish
        self._has_se = (
            (self._block_args.se_ratio is not None)
            and (self._block_args.se_ratio > 0)
            and (self._block_args.se_ratio <= 1)
        )

        self.endpoints = None

        # Builds the block accordings to arguments.
        self._build()

    def block_args(self):
        return self._block_args

    def _build(self):
        """Builds block according to the arguments."""
        filters = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            # Expansion phase:
            self._expand_conv = tf.layers.Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=custom_conv_kernel_initializer,
                padding="same",
                data_format=self._data_format,
                use_bias=False,
            )
            self._bn0 = batchnorm(
                axis=self._channel_axis,
                momentum=self._batch_norm_momentum,
                epsilon=self._batch_norm_epsilon,
            )

        kernel_size = self._block_args.kernel_size
        # Depth-wise convolution phase:
        self._depthwise_conv = DepthwiseConv2D(
            [kernel_size, kernel_size],
            strides=self._block_args.strides,
            depthwise_initializer=custom_conv_kernel_initializer,
            padding="same",
            data_format=self._data_format,
            use_bias=False,
        )
        self._bn1 = batchnorm(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon,
        )

        if self._has_se:
            num_reduced_filters = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio)
            )
            # Squeeze and Excitation layer.
            self._se_reduce = tf.layers.Conv2D(
                num_reduced_filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=custom_conv_kernel_initializer,
                padding="same",
                data_format=self._data_format,
                use_bias=True,
            )
            self._se_expand = tf.layers.Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=custom_conv_kernel_initializer,
                padding="same",
                data_format=self._data_format,
                use_bias=True,
            )

        # Output phase:
        filters = self._block_args.output_filters
        self._project_conv = tf.layers.Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=custom_conv_kernel_initializer,
            padding="same",
            data_format=self._data_format,
            use_bias=False,
        )
        self._bn2 = batchnorm(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon,
        )

    def _call_se(self, input_tensor):
        """Call Squeeze and Excitation layer.

        Args:
          input_tensor: Tensor, a single input tensor for Squeeze/Excitation layer.

        Returns:
          A output tensor, which should have the same shape as input.
        """
        se_tensor = tf.reduce_mean(input_tensor, self._spatial_dims, keepdims=True)
        se_tensor = self._se_expand(self._relu_fn(self._se_reduce(se_tensor)))
        tf.logging.info(
            "Built Squeeze and Excitation with tensor shape: %s" % (se_tensor.shape)
        )
        return tf.sigmoid(se_tensor) * input_tensor

    def call(self, inputs, training=True, drop_connect_rate=None):
        """Implementation of call().

        Args:
          inputs: the inputs tensor.
          training: boolean, whether the model is constructed for training.
          drop_connect_rate: float, between 0 to 1, drop connect rate.

        Returns:
          A output tensor.
        """
        tf.logging.info("Block input: %s shape: %s" % (inputs.name, inputs.shape))
        if self._block_args.expand_ratio != 1:
            if self._block_args.use_batchnorm:
                x = self._relu_fn(
                    self._bn0(self._expand_conv(inputs), training=training)
                )
            else:
                x = self._relu_fn(self._expand_conv(inputs))
        else:
            x = inputs
        tf.logging.info("Expand: %s shape: %s" % (x.name, x.shape))

        if self._block_args.use_batchnorm:
            x = self._relu_fn(self._bn1(self._depthwise_conv(x), training=training))
        else:
            x = self._relu_fn(self._depthwise_conv(x))

        tf.logging.info("DWConv: %s shape: %s" % (x.name, x.shape))

        if self._has_se:
            with tf.variable_scope("se"):
                x = self._call_se(x)

        self.endpoints = {"expansion_output": x}

        if self._block_args.use_batchnorm:
            x = self._bn2(self._project_conv(x), training=training)
        else:
            x = self._project_conv(x)

        if self._block_args.id_skip:
            if (
                all(s == 1 for s in self._block_args.strides)
                and self._block_args.input_filters == self._block_args.output_filters
            ):
                # only apply drop_connect if skip presents.
                if drop_connect_rate:
                    x = drop_connect(x, training, drop_connect_rate)
                x = tf.add(x, inputs)
        tf.logging.info("Project: %s shape: %s" % (x.name, x.shape))
        return x


class MBConvBlockWithoutDepthwise(MBConvBlock):
    """MBConv-like block without depthwise convolution and squeeze-and-excite."""

    def _build(self):
        """Builds block according to the arguments."""
        filters = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            # Expansion phase:
            self._expand_conv = tf.layers.Conv2D(
                filters,
                kernel_size=[3, 3],
                strides=[1, 1],
                kernel_initializer=custom_conv_kernel_initializer,
                padding="same",
                use_bias=False,
            )
            self._bn0 = batchnorm(
                axis=self._channel_axis,
                momentum=self._batch_norm_momentum,
                epsilon=self._batch_norm_epsilon,
            )

        # Output phase:
        filters = self._block_args.output_filters
        self._project_conv = tf.layers.Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=self._block_args.strides,
            kernel_initializer=custom_conv_kernel_initializer,
            padding="same",
            use_bias=False,
        )
        self._bn1 = batchnorm(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon,
        )

    def call(self, inputs, training=True, drop_connect_rate=None):
        """Implementation of call().

    Args:
      inputs: the inputs tensor.
      training: boolean, whether the model is constructed for training.
      drop_connect_rate: float, between 0 to 1, drop connect rate.

    Returns:
      A output tensor.
    """
        tf.logging.info("Block input: %s shape: %s" % (inputs.name, inputs.shape))
        if self._block_args.expand_ratio != 1:
            if self._block_args.use_batchnorm:
                x = self._relu_fn(
                    self._bn0(self._expand_conv(inputs), training=training)
                )
            else:
                x = self._relu_fn(self._expand_conv(inputs))
        else:
            x = inputs
        tf.logging.info("Expand: %s shape: %s" % (x.name, x.shape))

        self.endpoints = {"expansion_output": x}

        if self._block_args.use_batchnorm:
            x = self._bn1(self._project_conv(x), training=training)
        else:
            x = self._project_conv(x)

        if self._block_args.id_skip:
            if (
                all(s == 1 for s in self._block_args.strides)
                and self._block_args.input_filters == self._block_args.output_filters
            ):
                # only apply drop_connect if skip presents.
                if drop_connect_rate:
                    x = drop_connect(x, training, drop_connect_rate)
                x = tf.add(x, inputs)
        tf.logging.info("Project: %s shape: %s" % (x.name, x.shape))
        return x

