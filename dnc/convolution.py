from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow.compat.v1 as tf
import tensorflow.keras.backend as K
from tensorflow.python.framework import tensor_shape
import tensorflow_compression as tfc
from .niclib.layers import SubpixelConvBlock, UpsampleLayer, Mish
from .niclib import normalizers


def conv_kernel_initializer(shape, dtype=None, partition_info=None):
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


class DepthwiseConv2D(tf.keras.layers.DepthwiseConv2D, tf.layers.Layer):
    """Wrap keras DepthwiseConv2D to tf.layers."""

    pass


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


class MBConvBlock(tf.keras.layers.Layer):
    """
        A class of Mobile Inverted Residual Bottleneck Layer
        from EfficientNet paper
    """

    def __init__(
        self,
        input_filters: int,
        output_filters: int,
        kernel_size,
        strides,
        expand_ratio: int,
        se_ratio: float,
        id_skip: bool,
        name: str,
        use_batchnorm: bool = False,
        batch_norm_momentum: float = 0.99,
        batch_norm_epsilon: float = 1e-3,
        data_format: str = "channels_last",
        relu_fn=tf.nn.swish,
        **kwargs
    ):
        """
            Initializes a MBConv block
        """
        super(MBConvBlock, self).__init__(**kwargs)
        self._input_filters = input_filters
        self._output_filters = output_filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._expand_ratio = expand_ratio
        self._se_ratio = se_ratio
        self._id_skip = id_skip
        self._block_name = name
        self._use_batchnorm = use_batchnorm
        self._batch_norm_momentum = batch_norm_momentum
        self._batch_norm_epsilon = batch_norm_epsilon
        self._data_format = data_format
        if self._data_format == "channels_first":
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]
        self._relu_fn = relu_fn or tf.nn.swish
        self._has_se = (
            (self._se_ratio is not None)
            and (self._se_ratio > 0)
            and (self._se_ratio <= 1)
        )
        self.endpoints = None

        # Build the blocks according to the arguments
        self._build()

    def _build(self):
        """
            Builds the block according to the arguments
        """
        filters = self._expand_ratio * self._input_filters
        if self._expand_ratio != 1:
            # Expansion phase
            self._expand_conv = tf.layers.Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding="same",
                data_format=self._data_format,
                use_bias=False,
            )
            if self._use_batchnorm:
                self._bn0 = tf.layers.BatchNormalization(
                    axis=self._channel_axis,
                    momentum=self._batch_norm_momentum,
                    epsilon=self._batch_norm_epsilon,
                )

        # Depth-wise Convolution phase
        self._depthwise_conv = DepthwiseConv2D(
            [self._kernel_size, self._kernel_size],
            strides=self._strides,
            depthwise_initializer=conv_kernel_initializer,
            padding="same",
            data_format=self._data_format,
            use_bias=False,
        )
        if self._use_batchnorm:
            self._bn1 = tf.layers.BatchNormalization(
                axis=self._channel_axis,
                momentum=self._batch_norm_momentum,
                epsilon=self._batch_norm_epsilon,
            )

        if self._has_se:
            num_reduced_filters = max(1, int(self._input_filters * self._se_ratio))
            # Squeeze and Excitation Layer
            self._se_reduce = tf.layers.Conv2D(
                num_reduced_filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding="same",
                data_format=self._data_format,
                use_bias=True,
            )
            self._se_expand = tf.layers.Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding="same",
                data_format=self._data_format,
                use_bias=True,
            )

        self._project_conv = tf.layers.Conv2D(
            self._output_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding="same",
            data_format=self._data_format,
            use_bias=False,
        )
        if self._use_batchnorm:
            self._bn2 = tf.layers.BatchNormalization(
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
        se_tensor = tf.reduce_mean(input_tensor, self._spatial_dims, keep_dims=True)
        se_tensor = self._se_expand(self._relu_fn(self._se_reduce(se_tensor)))
        tf.logging.info(
            "Built Squeeze and Excitation Layer with tensor shape %s"
            % (se_tensor.shape)
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
        with tf.variable_scope(self._block_name):
            tf.logging.info(
                "MBConvBlock-/Input: %s, shape: %s" % (inputs.name, inputs.shape)
            )
            if self._expand_ratio != 1:
                if self._use_batchnorm:
                    x = self._relu_fn(
                        self._bn0(self._expand_conv(inputs), training=training)
                    )
                else:
                    x = self._relu_fn(self._expand_conv(inputs))
            else:
                x = inputs
            tf.logging.info("MBConvBlock-/Expand: %s shape: %s" % (x.name, x.shape))

            if self._use_batchnorm:
                x = self._relu_fn(self._bn1(self._depthwise_conv(x), training=training))
            else:
                x = self._relu_fn(self._depthwise_conv(x))
            tf.logging.info("MBConvBlock-/DWConv: %s shape: %s" % (x.name, x.shape))

            if self._has_se:
                with tf.variable_scope("se"):
                    x = self._call_se(x)

            self.endpoints = {"expansion_output": x}

            if self._use_batchnorm:
                x = self._bn2(self._project_conv(x), training=training)
            else:
                x = self._project_conv(x)

            if self._id_skip:
                if (
                    all(s == 1 for s in self._strides)
                    and self._input_filters == self._output_filters
                ):
                    # only apply drop_connect if skip presents
                    if drop_connect_rate:
                        x = drop_connect(x, training, drop_connect_rate)
                    x = tf.add(x, inputs)
            tf.logging.info("MBConvBlock-/Project: %s shape: %s" % (x.name, x.shape))
        return x


class MBConvBlockWithoutDepthwise(MBConvBlock):
    """MBConv-like block without depthwise convolution and squeeze-and-excite."""

    def _build(self):
        """Builds block according to the arguments"""
        filters = self._expand_ratio * self._input_filters
        if self._expand_ratio != 1:
            # Expansion phase
            self._expand_conv = tf.layers.Conv2D(
                filters,
                kernel_size=[3, 3],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding="same",
                use_bias=False,
            )
            if self._use_batchnorm:
                self._bn0 = tf.layers.BatchNormalization(
                    axis=self._channel_axis,
                    momentum=self._batch_norm_momentum,
                    epsilon=self._batch_norm_epsilon,
                )

        # Output phase
        filters = self._output_filters
        self._project_conv = tf.layers.Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=self._strides,
            kernel_initializer=conv_kernel_initializer,
            padding="same",
            use_bias=False,
        )
        if self._use_batchnorm:
            self._bn1 = tf.layers.BatchNormalization(
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
        with tf.variable_scope(self._block_name):
            tf.logging.info("Block input: %s shape: %s" % (inputs.name, inputs.shape))
            if self._expand_ratio != 1:
                if self._use_batchnorm:
                    x = self._relu_fn(
                        self._bn0(self._expand_conv(inputs), training=training)
                    )
                else:
                    x = self._relu_fn(self._expand_conv(inputs))
            else:
                x = inputs
            tf.logging.info("Expand: %s shape: %s" % (x.name, x.shape))

            self.endpoints = {"expansion_output": x}

            if self._use_batchnorm:
                x = self._bn1(self._project_conv(x), training=training)
            else:
                x = self._project_conv(x)

            if self._id_skip:
                if (
                    all(s == 1 for s in self._strides)
                    and self._input_filters == self._output_filters
                ):
                    # only apply drop-connect if skip presents
                    if drop_connect_rate:
                        x = drop_connect(x, training, drop_connect_rate)
                    x = tf.add(x, inputs)
            tf.logging.info("Project: %s shape: %s" % (x.name, x.shape))
        return x


class MBSignalConvBlock(tf.keras.layers.Layer):
    """A class of MBSignalConv: Mobile Inverted Residual Bottleneck using tfc.SignalConv2D
    
    Attributes:
        endpoints (dict): a list of internal tensors for feature extraction
    """

    def __init__(
        self,
        input_filters: int,
        output_filters: int,
        kernel_size,
        strides,
        expand_ratio: int,
        se_ratio: float,
        id_skip: bool,
        name: str,
        use_batchnorm: bool = False,
        batch_norm_momentum: float = 0.99,
        batch_norm_epsilon: float = 1e-3,
        data_format: str = "channels_last",
        relu_fn=tf.nn.swish,
        **kwargs
    ):
        """
            Initializes a MBSignalConv block
        """
        super(MBSignalConvBlock, self).__init__(**kwargs)
        self._input_filters = input_filters
        self._output_filters = output_filters
        self._kernel_size = kernel_size
        self._stride = strides
        self._expand_ratio = expand_ratio
        self._se_ratio = se_ratio
        self._id_skip = id_skip
        self._block_name = name
        self._use_batchnorm = use_batchnorm
        self._batch_norm_momentum = batch_norm_momentum
        self._batch_norm_epsilon = batch_norm_epsilon
        self._data_format = data_format
        if self._data_format == "channels_first":
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]
        self._relu_fn = relu_fn or tf.nn.swish
        self._has_se = (
            (self._se_ratio is not None)
            and (self._se_ratio > 0)
            and (self._se_ratio <= 1)
        )
        self.endpoints = None

        # Build the blocks according to the arguments
        self._build()

    def _build(self):
        """
            Builds the block according to the arguments
        """
        filters = self._expand_ratio * self._input_filters
        if self._expand_ratio != 1:
            # Expansion phase:
            self._expand_conv = tfc.SignalConv2D(
                filters,
                kernel_support=(1, 1),
                kernel_initializer=conv_kernel_initializer,
                strides_down=1,
                corr=True,
                padding="same_zeros",
                data_format=self._data_format,
                use_bias=False,
                activation=tfc.GDN(
                    data_format=self._data_format, name="expandconv_gdn"
                ),
            )
            if self._use_batchnorm:
                self._bn0 = tf.layers.BatchNormalization(
                    axis=self._channel_axis,
                    momentum=self._batch_norm_momentum,
                    epsilon=self._batch_norm_epsilon,
                )

        if self._stride > 1:
            self._downsample = tfc.SignalConv2D(
                filters,
                (2, 2),
                name="downsample_sigconv",
                corr=True,
                strides_down=self._stride,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(
                    data_format=self._data_format,
                    inverse=True,
                    name="downsampleconv_igdn",
                ),
            )
        else:
            # Depth-wise Signal-Convolution phase
            self._downsample = tfc.SignalConv2D(
                filters=1,  # here represents the depthwise_multiplier for DepthwiseConv2D
                kernel_support=(self._kernel_size, self._kernel_size),
                kernel_initializer=conv_kernel_initializer,
                strides_down=self._stride,
                corr=True,
                channel_separable=True,  # This changes it from normal sigconv to depthwise sigconv
                padding="same_zeros",
                data_format=self._data_format,
                use_bias=False,
                activation=tfc.GDN(
                    data_format=self._data_format, name="depthwiseconv_gdn"
                ),
            )
        if self._use_batchnorm:
            self._bn1 = tf.layers.BatchNormalization(
                axis=self._channel_axis,
                momentum=self._batch_norm_momentum,
                epsilon=self._batch_norm_epsilon,
            )

        if self._has_se:
            num_reduced_filters = max(1, int(self._input_filters * self._se_ratio))
            # Squeeze and Excitation Layer
            self._se_reduce = tfc.SignalConv2D(
                num_reduced_filters,
                kernel_support=(1, 1),
                kernel_initializer=conv_kernel_initializer,
                strides_down=1,
                corr=True,
                padding="same_zeros",
                data_format=self._data_format,
                use_bias=True,
                activation=tfc.GDN(data_format=self._data_format, name="se_reduce_gdn"),
            )
            self._se_expand = tfc.SignalConv2D(
                filters,
                kernel_support=(1, 1),
                kernel_initializer=conv_kernel_initializer,
                strides_down=1,
                corr=True,
                padding="same_zeros",
                data_format=self._data_format,
                use_bias=True,
            )

        self._project_conv = tfc.SignalConv2D(
            self._output_filters,
            kernel_support=(1, 1),
            kernel_initializer=conv_kernel_initializer,
            strides_down=1,
            corr=True,
            padding="same_zeros",
            data_format=self._data_format,
            use_bias=False,
        )
        if self._use_batchnorm:
            self._bn2 = tf.layers.BatchNormalization(
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
        se_tensor = tf.reduce_mean(input_tensor, self._spatial_dims, keep_dims=True)
        se_tensor = self._se_expand(self._se_reduce(se_tensor))
        tf.logging.info(
            "Built Squeeze and Excitation Layer with tensor shape %s"
            % (se_tensor.shape)
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
        with tf.variable_scope(self._block_name):
            tf.logging.info(
                "MBSignalConvBlock-/Input: %s shape: %s" % (inputs.name, inputs.shape)
            )
            if self._expand_ratio != 1:
                if self._use_batchnorm:
                    x = self._bn0(self._expand_conv(inputs), training=training)
                else:
                    x = self._expand_conv(inputs)
            else:
                x = inputs
            tf.logging.info(
                "MBSignalConvBlock-/Expand: %s shape: %s" % (x.name, x.shape)
            )

            if self._use_batchnorm:
                x = self._bn1(self._downsample(x), training=training)
            else:
                x = self._downsample(x)
            tf.logging.info(
                "MBSignalConvBlock-/Downsample(DWConv): %s shape: %s"
                % (x.name, x.shape)
            )

            if self._has_se:
                with tf.variable_scope("se"):
                    x = self._call_se(x)

            self.endpoints = {"expansion_output": x}

            if self._use_batchnorm:
                x = self._bn2(self._project_conv(x), training=training)
            else:
                x = self._project_conv(x)

            if self._id_skip:
                if (self._stride == 1) and (
                    self._input_filters == self._output_filters
                ):
                    # only apply drop_connect if skip presents
                    if drop_connect_rate:
                        x = drop_connect(x, training, drop_connect_rate)
                    x = tf.add(x, inputs)
            tf.logging.info(
                "MBSignalConvBlock-/Project: %s shape: %s" % (x.name, x.shape)
            )
        return x


class MBSignalConvBlockWithoutDepthwise(MBSignalConvBlock):
    """MBConv-like block without depthwise convolution and squeeze-and-excite."""

    def _build(self):
        """Builds block according to the arguments"""
        filters = self._expand_ratio * self._input_filters
        if self._expand_ratio != 1:
            # Expansion phase:
            self._expand_conv = tfc.SignalConv2D(
                filters,
                kernel_support=(3, 3),
                kernel_initializer=conv_kernel_initializer,
                strides_down=1,
                corr=True,
                padding="same_zeros",
                data_format=self._data_format,
                use_bias=False,
                activation=tfc.GDN(data_format=self._data_format, name="expand_conv"),
            )
            if self._use_batchnorm:
                self._bn0 = tf.layers.BatchNormalization(
                    axis=self._channel_axis,
                    momentum=self._batch_norm_momentum,
                    epsilon=self._batch_norm_epsilon,
                )

        # Output phase
        filters = self._output_filters
        self._project_conv = tfc.SignalConv2D(
            filters,
            kernel_support=(1, 1),
            kernel_initializer=conv_kernel_initializer,
            strides_down=1,
            corr=True,
            padding="same_zeros",
            data_format=self._data_format,
            use_bias=False,
            activation=tfc.GDN(data_format=self._data_format),
        )
        if self._use_batchnorm:
            self._bn1 = tf.layers.BatchNormalization(
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
        with tf.variable_scope(self._block_name):
            tf.logging.info(
                "MBSignalConvBlock_ndw-/Input: %s shape: %s"
                % (inputs.name, inputs.shape)
            )
            if self._expand_ratio != 1:
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

            if self._id_skip:
                if (
                    all(s == 1 for s in self._strides)
                    and self._input_filters == self._output_filters
                ):
                    # only apply drop-connect if skip presents
                    if drop_connect_rate:
                        x = drop_connect(x, training, drop_connect_rate)
                    x = tf.add(x, inputs)
            tf.logging.info(
                "MBSignalConvBlock_ndw-/Project: %s shape: %s" % (x.name, x.shape)
            )
        return x


class MBUpsampleSCBlock(tf.keras.layers.Layer):
    """A class of MBUpsampleBlock: Mobile Inverted Residual Bottleneck using tfc.SignalConv2D
        and PixelShuffle with ICNR and blur kernels for upsampling

    Attributes:
        endpoints (dict): a list of internal tensors for feature extraction
    """

    def __init__(
        self,
        input_filters: int,
        output_filters: int,
        kernel_size,
        scale,
        expand_ratio: int,
        se_ratio: float,
        id_skip: bool,
        name: str,
        use_subpixel: bool = False,
        use_blur: bool = True,
        use_batchnorm: bool = False,
        batch_norm_momentum: float = 0.99,
        batch_norm_epsilon: float = 1e-3,
        data_format: str = "channels_last",
        leaky_alpha: int = 0.2,
        **kwargs
    ):
        """
            Initializes a MBUpsampleSC block
        """
        super(MBUpsampleSCBlock, self).__init__(**kwargs)
        self._input_filters = input_filters
        self._output_filters = output_filters
        self._kernel_size = kernel_size
        self._scale = scale
        self._expand_ratio = expand_ratio
        self._se_ratio = se_ratio
        self._id_skip = id_skip
        self._block_name = name
        self._use_subpixconv = use_subpixel
        self._use_blur = use_blur
        self._use_batchnorm = use_batchnorm
        self._batch_norm_momentum = batch_norm_momentum
        self._batch_norm_epsilon = batch_norm_epsilon
        self._data_format = data_format
        if self._data_format == "channels_first":
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]
        self._leaky_alpha = leaky_alpha
        self._has_se = (
            (self._se_ratio is not None)
            and (self._se_ratio > 0)
            and (self._se_ratio <= 1)
        )
        self.endpoints = None

        # Build the blocks according to the arguments
        self._build()

    def _build(self):
        """
            Builds the block according to the arguments
        """
        filters = self._expand_ratio * self._input_filters
        if self._expand_ratio != 1:
            # Expansion phase:
            self._expand_conv = tfc.SignalConv2D(
                filters,
                kernel_support=(1, 1),
                kernel_initializer=conv_kernel_initializer,
                strides_up=1,
                corr=False,
                padding="same_zeros",
                data_format=self._data_format,
                use_bias=False,
                activation=tfc.GDN(
                    data_format=self._data_format, inverse=True, name="expandconv_igdn"
                ),
            )
            if self._use_batchnorm:
                self._bn0 = tf.layers.BatchNormalization(
                    axis=self._channel_axis,
                    momentum=self._batch_norm_momentum,
                    epsilon=self._batch_norm_epsilon,
                )

        if self._scale > 1:
            # Upsample phase
            if self._use_subpixconv:
                self._upsample = SubpixelConvBlock(
                    scale=self._scale,
                    use_blur=self._use_blur,
                    leaky_alpha=self._leaky_alpha,
                    data_format=self._data_format,
                )
            else:
                self._upsample = tfc.SignalConv2D(
                    filters,
                    (2, 2),
                    name="upsample_sigconv",
                    corr=False,
                    strides_up=self._scale,
                    padding="same_zeros",
                    use_bias=True,
                    activation=tfc.GDN(
                        data_format=self._data_format,
                        inverse=True,
                        name="upsampleconv_igdn",
                    ),
                )

        else:
            # Depth-wise Signal-Convolution phase
            self._upsample = tfc.SignalConv2D(
                filters=1,  # here represents the depthwise_multiplier for DepthwiseConv2D
                kernel_support=(self._kernel_size, self._kernel_size),
                kernel_initializer=conv_kernel_initializer,
                strides_up=self._scale,
                corr=False,
                channel_separable=True,  # This changes it from normal sigconv to depthwise sigconv
                padding="same_zeros",
                data_format=self._data_format,
                use_bias=False,
                activation=tfc.GDN(
                    data_format=self._data_format,
                    inverse=True,
                    name="depthwiseconv_igdn",
                ),
            )
        if self._use_batchnorm:
            self._bn1 = tf.layers.BatchNormalization(
                axis=self._channel_axis,
                momentum=self._batch_norm_momentum,
                epsilon=self._batch_norm_epsilon,
            )

        if self._has_se:
            num_reduced_filters = max(1, int(self._input_filters * self._se_ratio))
            # Squeeze and Excitation Layer
            self._se_reduce = tfc.SignalConv2D(
                num_reduced_filters,
                kernel_support=(1, 1),
                kernel_initializer=conv_kernel_initializer,
                strides_up=1,
                corr=False,
                padding="same_zeros",
                data_format=self._data_format,
                use_bias=True,
                activation=tfc.GDN(
                    data_format=self._data_format, inverse=True, name="se_reduce_igdn"
                ),
            )
            self._se_expand = tfc.SignalConv2D(
                filters,
                kernel_support=(1, 1),
                kernel_initializer=conv_kernel_initializer,
                strides_up=1,
                corr=False,
                padding="same_zeros",
                data_format=self._data_format,
                use_bias=True,
            )

        self._project_conv = tfc.SignalConv2D(
            self._output_filters,
            kernel_support=(1, 1),
            kernel_initializer=conv_kernel_initializer,
            strides_up=1,
            corr=False,
            padding="same_zeros",
            data_format=self._data_format,
            use_bias=False,
        )
        if self._use_batchnorm:
            self._bn2 = tf.layers.BatchNormalization(
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
        se_tensor = tf.reduce_mean(input_tensor, self._spatial_dims, keep_dims=True)
        se_tensor = self._se_expand(self._se_reduce(se_tensor))
        tf.logging.info(
            "Built Squeeze and Excitation Layer with tensor shape %s"
            % (se_tensor.shape)
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
        with tf.variable_scope(self._block_name):
            tf.logging.info(
                "MBUpsampleBlock-/Input: %s shape: %s" % (inputs.name, inputs.shape)
            )
            if self._expand_ratio != 1:
                if self._use_batchnorm:
                    x = self._bn0(self._expand_conv(inputs), training=training)
                else:
                    x = self._expand_conv(inputs)
            else:
                x = inputs
            tf.logging.info("MBUpsampleBlock-/Expand: %s shape: %s" % (x.name, x.shape))

            if self._use_batchnorm:
                x = self._bn1(self._upsample(x), training=training)
            else:
                x = self._upsample(x)
            tf.logging.info(
                "MBUpsampleBlock-/Upsample: %s shape: %s" % (x.name, x.shape)
            )

            if self._has_se:
                with tf.variable_scope("se"):
                    x = self._call_se(x)

            self.endpoints = {"expansion_output": x}

            if self._use_batchnorm:
                x = self._bn2(self._project_conv(x), training=training)
            else:
                x = self._project_conv(x)

            if self._id_skip:
                if (self._scale == 1) and (self._input_filters == self._output_filters):
                    # only apply drop_connect if skip presents
                    if drop_connect_rate:
                        x = drop_connect(x, training, drop_connect_rate)
                    x = tf.add(x, inputs)
            tf.logging.info(
                "MBUpsampleBlock-/Project: %s shape: %s" % (x.name, x.shape)
            )
        return x


class MBUpsampleConvBlock(tf.keras.layers.Layer):
    """A class of MBUpsampleConvBlock: Mobile Inverted Residual Bottleneck using tf.keras.layers.Conv2D
        and PixelShuffle with ICNR and blur kernels for upsampling

    Attributes:
        endpoints (dict): a list of internal tensors for feature extraction
    """

    def __init__(
        self,
        input_filters: int,
        output_filters: int,
        kernel_size,
        scale,
        expand_ratio: int,
        se_ratio: float,
        id_skip: bool,
        name: str,
        use_subpixel: bool = True,
        use_blur: bool = True,
        use_batchnorm: bool = False,
        batch_norm_momentum: float = 0.99,
        batch_norm_epsilon: float = 1e-3,
        data_format: str = "channels_last",
        leaky_alpha: int = 0.2,
        relu_fn=tf.nn.swish,
        **kwargs
    ):
        """
            Initializes a MBUpsampleConv block
        """
        super(MBUpsampleConvBlock, self).__init__(**kwargs)
        self._input_filters = input_filters
        self._output_filters = output_filters
        self._kernel_size = kernel_size
        self._scale = scale
        self._expand_ratio = expand_ratio
        self._se_ratio = se_ratio
        self._id_skip = id_skip
        self._block_name = name
        self._use_subpixconv = use_subpixel
        self._use_blur = use_blur
        self._use_batchnorm = use_batchnorm
        self._batch_norm_momentum = batch_norm_momentum
        self._batch_norm_epsilon = batch_norm_epsilon
        self._data_format = data_format
        if self._data_format == "channels_first":
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]
        self._leaky_alpha = leaky_alpha
        self._relu_fn = relu_fn or tf.nn.swish
        self._has_se = (
            (self._se_ratio is not None)
            and (self._se_ratio > 0)
            and (self._se_ratio <= 1)
        )
        self.endpoints = None

        # Build the blocks according to the arguments
        self._build()

    def _build(self):
        """
            Builds the block according to the arguments
        """
        filters = self._expand_ratio * self._input_filters
        if self._expand_ratio != 1:
            # Expansion phase:
            self._expand_conv = tf.layers.Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding="same",
                data_format=self._data_format,
                use_bias=False,
            )
            if self._use_batchnorm:
                self._bn0 = tf.layers.BatchNormalization(
                    axis=self._channel_axis,
                    momentum=self._batch_norm_momentum,
                    epsilon=self._batch_norm_epsilon,
                )

        if self._scale > 1:
            # Upsample phase
            if self._use_subpixconv:
                self._upsample = SubpixelConvBlock(
                    scale=self._scale,
                    use_blur=self._use_blur,
                    leaky_alpha=self._leaky_alpha,
                    data_format=self._data_format,
                )
            else:
                self._upsample = UpsampleLayer(
                    output_filters=filters,
                    stride=[self._scale, self._scale],
                    method="conv2d_transpose",
                )

        else:
            # Depth-wise Convolution phase
            self._upsample = DepthwiseConv2D(
                [self._kernel_size, self._kernel_size],
                strides=[self._scale, self._scale],
                depthwise_initializer=conv_kernel_initializer,
                padding="same",
                data_format=self._data_format,
                use_bias=False,
            )
        if self._use_batchnorm:
            self._bn1 = tf.layers.BatchNormalization(
                axis=self._channel_axis,
                momentum=self._batch_norm_momentum,
                epsilon=self._batch_norm_epsilon,
            )

        if self._has_se:
            num_reduced_filters = max(1, int(self._input_filters * self._se_ratio))
            # Squeeze and Excitation Layer
            self._se_reduce = tf.layers.Conv2D(
                num_reduced_filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding="same",
                data_format=self._data_format,
                use_bias=True,
            )
            self._se_expand = tf.layers.Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding="same",
                data_format=self._data_format,
                use_bias=True,
            )

        self._project_conv = tf.layers.Conv2D(
            self._output_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding="same",
            data_format=self._data_format,
            use_bias=False,
        )
        if self._use_batchnorm:
            self._bn2 = tf.layers.BatchNormalization(
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
        se_tensor = tf.reduce_mean(input_tensor, self._spatial_dims, keep_dims=True)
        se_tensor = self._se_expand(self._relu_fn(self._se_reduce(se_tensor)))
        tf.logging.info(
            "Built Squeeze and Excitation Layer with tensor shape %s"
            % (se_tensor.shape)
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
        with tf.variable_scope(self._block_name):
            tf.logging.info(
                "MBUpsampleConvBlock-/Input: %s shape: %s" % (inputs.name, inputs.shape)
            )
            if self._expand_ratio != 1:
                if self._use_batchnorm:
                    x = self._relu_fn(
                        self._bn0(self._expand_conv(inputs), training=training)
                    )
                else:
                    x = self._relu_fn(self._expand_conv(inputs))
            else:
                x = inputs
            tf.logging.info(
                "MBUpsampleConvBlock-/Expand: %s shape: %s" % (x.name, x.shape)
            )

            if self._use_batchnorm:
                x = self._relu_fn(self._bn1(self._upsample(x), training=training))
            else:
                x = self._relu_fn(self._upsample(x))
            tf.logging.info(
                "MBUpsampleConvBlock-/Upsample: %s shape: %s" % (x.name, x.shape)
            )

            if self._has_se:
                with tf.variable_scope("se"):
                    x = self._call_se(x)

            self.endpoints = {"expansion_output": x}

            if self._use_batchnorm:
                x = self._bn2(self._project_conv(x), training=training)
            else:
                x = self._project_conv(x)

            if self._id_skip:
                if (self._scale == 1) and (self._input_filters == self._output_filters):
                    # only apply drop_connect if skip presents
                    if drop_connect_rate:
                        x = drop_connect(x, training, drop_connect_rate)
                    x = tf.add(x, inputs)
            tf.logging.info(
                "MBUpsampleConvBlock-/Project: %s shape: %s" % (x.name, x.shape)
            )
        return x


def _get_conv2d(filters, kernel_size, use_keras, **kwargs):
    """A helper function to create Conv2D layer."""
    if use_keras:
        return tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, **kwargs
        )
    else:
        return tf.layers.Conv2D(filters=filters, kernel_size=kernel_size, **kwargs)


def _split_channels(total_filters, num_groups):
    split = [total_filters // num_groups for _ in range(num_groups)]
    split[0] += total_filters - sum(split)
    return split


class GroupedConv2D(object):
    """
        Groupped convolution.

        Currently tf.keras and tf.layers don't support group convolution, so here we
        use split/concat to implement this op. It reuses kernel_size for group
        definition, where len(kernel_size) is number of groups. Notably, it allows
        different group has different kernel size.
    """

    def __init__(self, filters, kernel_size, use_keras, **kwargs):
        """Initialize the layer.
        Args:
          filters: Integer, the dimensionality of the output space.
          kernel_size: An integer or a list. If it is a single integer, then it is
            same as the original Conv2D. If it is a list, then we split the channels
            and perform different kernel for each group.
          use_keras: An boolean value, whether to use keras layer.
          **kwargs: other parameters passed to the original conv2d layer.
        """
        self._groups = len(kernel_size)
        self._channel_axis = -1

        self._convs = []
        splits = _split_channels(filters, self._groups)
        for i in range(self._groups):
            self._convs.append(
                _get_conv2d(splits[i], kernel_size[i], use_keras, **kwargs)
            )

    def __call__(self, inputs):
        if len(self._convs) == 1:
            return self._convs[0](inputs)

        filters = inputs.shape[self._channel_axis].value
        splits = _split_channels(filters, len(self._convs))
        x_splits = tf.split(inputs, splits, self._channel_axis)
        x_outputs = [c(x) for x, c in zip(x_splits, self._convs)]
        x = tf.concat(x_outputs, self._channel_axis)
        return x


class MixConv(object):
    """
        MixConv with mixed depthwise convolutional kernels.

        MDConv is an improved depthwise convolution that mixes multiple kernels (e.g.
        3x3, 5x5, etc). Right now, we use an naive implementation that split channels
        into multiple groups and perform different kernels for each group.
        See Mixnet paper for more details.
    """

    def __init__(self, kernel_size, strides, dilated=False, **kwargs):
        """
        Initialize the layer.
            Most of args are the same as tf.keras.layers.DepthwiseConv2D except it has
            an extra parameter "dilated" to indicate whether to use dilated conv to
            simulate large kernel size. If dilated=True, then dilation_rate is ignored.
            Args:
              kernel_size: An integer or a list. If it is a single integer, then it is
                same as the original tf.keras.layers.DepthwiseConv2D. If it is a list,
                then we split the channels and perform different kernel for each group.
              strides: An integer or tuple/list of 2 integers, specifying the strides of
                the convolution along the height and width.
              dilated: Bool. indicate whether to use dilated conv to simulate large
                kernel size.
              **kwargs: other parameters passed to the original depthwise_conv layer.
        """
        self._channel_axis = -1
        self._dilated = dilated

        self._convs = []
        for s in kernel_size:
            d = 1
            if strides[0] == 1 and self._dilated:
                # Only apply dilated conv for stride 1 if needed.
                d, s = (s - 1) // 2, 3
                tf.logging.info("Use dilated conv with dilation rate = {}".format(d))
            self._convs.append(
                tf.keras.layers.DepthwiseConv2D(
                    s, strides=strides, dilation_rate=d, **kwargs
                )
            )

    def __call__(self, inputs):
        if len(self._convs) == 1:
            return self._convs[0](inputs)

        filters = inputs.shape[self._channel_axis].value
        splits = _split_channels(filters, len(self._convs))
        x_splits = tf.split(inputs, splits, self._channel_axis)
        x_outputs = [c(x) for x, c in zip(x_splits, self._convs)]
        x = tf.concat(x_outputs, self._channel_axis)
        return x


class DualPathBlock(tf.keras.layers.Layer):
    """
        Creates a Dual Path Block. The first path is a ResNeXt type
        grouped convolution block. The second is a DenseNet type dense
        convolution block.
        Dual Path Networks are highly efficient networks, which combine the strength
        of both ResNeXt (Aggregated Residual Transformations for Deep Neural Networks)
        and DenseNets (Densely Connected Convolutional Networks).
    """

    def __init__(
        self,
        pointwise_filters_a: int = 96,
        grouped_conv_filters_b: int = 96,
        pointwise_filters_c: int = 128,
        filter_increment: int = 64,
        cardinality: int = 32,
        block_width: int = 3,
        block_type="projection",
        data_format="channels_last",
        name="dpn_block",
        **kwargs
    ):
        """
        Initialize a Dual Path Block
        Parameters
        ----------
        pointwise_filters_a : number of filters for bottleneck pointwise convolution
        grouped_conv_filters_b : number of filters for grouped convolution block
        pointwise_filters_c : number of filters for bottleneck pointwise convolution
        filter_increment : number of filters to be added
        cardinality : cardinality factor (resnext concept)
        block_width : int : can be used for calculating pointwise_filter_a and grouped_conv_filters_b if given
        block_type : determines what action the block will perform
                    - `projection`: adds a projection connection
                    - `downsample`: downsamples the spatial resolution
                    - `normal`    : simple adds a dual path connection
        """
        super(DualPathBlock, self).__init__(**kwargs)
        self._pointwise_filters_a = pointwise_filters_a
        self._grouped_conv_filters_b = grouped_conv_filters_b
        self._pointwise_filters_c = pointwise_filters_c
        self._filter_increment = filter_increment
        self._cardinality = cardinality
        self._width = block_width
        self.block_type = block_type
        self._data_format = data_format
        if self._data_format == "channels_first":
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]

        self.block_name = name

        if self._pointwise_filters_a is None or self._grouped_conv_filters_b is None:
            self._pointwise_filters_a = int(self._cardinality * self._width)
            self._grouped_conv_filters_b = int(self._cardinality * self._width)

        if self._filter_increment is None:
            self._filter_increment = int(self._pointwise_filters_c / 2)

        self._grouped_channels = int(self._grouped_conv_filters_b / self._cardinality)

        if self.block_type == "projection":
            self.stride = (1, 1)
            self.projection = True
        elif self.block_type == "downsample":
            self.stride = (2, 2)
            self.projection = False
        elif self.block_type == "normal":
            self.stride = (1, 1)
            self.projection = False
        else:
            raise ValueError(
                '`block_type` must be one of ["projection", "downsample", "normal"]. Given %s'
                % block_type
            )

        # Build the blocks according to the arguments
        self._build()

    def _convblock(
        self, inputs, filters, kernel=(3, 3), stride=(1, 1), weight_decay=5e-4
    ):
        """
        Adds a Conv-Mish-Conv Block for DPN
        Parameters
        ----------
        inputs : tf.Tensor
                 input tensor to convblock
        filters : int
                  number of output filters
        kernel : conv kernel size
        stride : convolution stride
        weight_decay : parameter for weight decay of conv kernel

        Returns
        -------
        A tf.keras Tensor
        """
        outputs = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay),
            strides=stride,
        )(inputs)
        outputs = tf.keras.layers.Activation("Mish", name="mish_convblock")(outputs)
        outputs = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay),
            strides=stride,
        )(outputs)

        return outputs

    def _grouped_conv(
        self, inputs, grouped_channels, cardinality, strides, weight_decay=5e-4
    ):
        """
        Adds a grouped convolutional block (ResNext layer)
        Parameters
        ----------
        inputs : tf.Tensor
            input tensor to the layer
        grouped_channels : int
            grouped number of filters
        cardinality : int
            cardinality factor describing the number of groups
        strides : list(int) or tuple(int)
            performs strided convolution for downscaling if > 1
        weight_decay : float
            weight decay factor for l2 regularizer

        Returns
        -------
        tf.keras Tensor
        """
        residual = inputs
        group_list = []
        if cardinality == 1:
            # with cardinality = 1 it is a normal resblock
            outputs = tf.keras.layers.Conv2D(
                grouped_channels,
                (3, 3),
                padding="same",
                use_bias=False,
                strides=strides,
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay),
            )(inputs)
            outputs = tf.keras.layers.Activation("Mish", name="mish_input")(outputs)
            return outputs

        for c in range(cardinality):
            outputs = tf.keras.layers.Lambda(
                lambda z: z[:, :, :, c * grouped_channels : (c + 1) * grouped_channels]
                if self._data_format == "channels_last"
                else lambda z: z[
                    :, c * grouped_channels : (c + 1) * grouped_channels, :, :
                ]
            )(inputs)
            outputs = tf.keras.layers.Conv2D(
                grouped_channels,
                (3, 3),
                padding="same",
                use_bias=False,
                strides=strides,
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay),
            )(outputs)
            group_list.append(outputs)

        group_merge = tf.keras.layers.concatenate(group_list, axis=self._channel_axis)
        group_merge = tf.keras.layers.Activation("Mish", name="mish_merged")(
            group_merge
        )
        return group_merge

    def _build(self):
        """
        Builds the block according to the arguments
        """
        self._output_residual_path = tf.keras.layers.Lambda(
            lambda z: z[:, :, :, : self._pointwise_filters_c]
            if self._data_format == "channels_last"
            else z[:, : self._pointwise_filters_c, :, :]
        )
        self._output_dense_path = tf.keras.layers.Lambda(
            lambda z: z[:, :, :, self._pointwise_filters_c :]
            if self._data_format == "channels_last"
            else z[:, self._pointwise_filters_c :, :, :]
        )

    def call(self, inputs):
        with tf.variable_scope(self.block_name):
            input_init = (
                tf.keras.layers.concatenate(inputs, axis=self._channel_axis)
                if isinstance(inputs, list)
                else inputs
            )

            if self.projection:
                projection_path = self._convblock(
                    input_init,
                    filters=self._pointwise_filters_c + self._filter_increment,
                    kernel=(1, 1),
                    stride=self.stride,
                )
                input_residual_path = tf.keras.layers.Lambda(
                    lambda z: z[:, :, :, : self._pointwise_filters_c]
                    if self._data_format == "channels_last"
                    else z[:, : self._pointwise_filters_c, :, :]
                )(projection_path)
                input_dense_path = tf.keras.layers.Lambda(
                    lambda z: z[:, :, :, self._pointwise_filters_c :]
                    if self._data_format == "channels_last"
                    else z[:, self._pointwise_filters_c :, :, :]
                )(projection_path)
                tf.logging.info(
                    "===<<< DPN -> input_dense_path : %s : %s >>>==="
                    % (input_dense_path.name, input_dense_path.shape)
                )
            else:
                input_residual_path = inputs[0]
                input_dense_path = inputs[1]

            outputs = self._convblock(
                input_init, filters=self._pointwise_filters_a, kernel=(1, 1)
            )
            outputs = self._grouped_conv(
                outputs,
                grouped_channels=self._grouped_channels,
                cardinality=self._cardinality,
                strides=self.stride,
            )
            outputs = self._convblock(
                outputs,
                filters=self._pointwise_filters_c + self._filter_increment,
                kernel=(1, 1),
            )

            output_residual_path = tf.keras.layers.Lambda(
                lambda z: z[:, :, :, : self._pointwise_filters_c]
                if self._data_format == "channels_last"
                else z[:, : self._pointwise_filters_c, :, :]
            )(outputs)
            output_dense_path = tf.keras.layers.Lambda(
                lambda z: z[:, :, :, self._pointwise_filters_c :]
                if self._data_format == "channels_last"
                else z[:, self._pointwise_filters_c :, :, :]
            )(outputs)
            tf.logging.info(
                "===<<< DPN -> output_dense_path : %s : %s >>>==="
                % (output_dense_path.name, output_dense_path.shape)
            )

            residual_path = tf.keras.layers.add(
                [input_residual_path, output_residual_path]
            )
            dense_path = tf.keras.layers.concatenate(
                [input_dense_path, output_dense_path], axis=self._channel_axis
            )
            tf.logging.info(
                "===<<< DPN -> dense_path : %s : %s >>>==="
                % (dense_path.name, dense_path.shape)
            )

            outputs = tf.keras.layers.multiply([residual_path, dense_path])

        return outputs


class GlobalContextBlock(tf.keras.layers.Layer):
    """
    Adds a Global Context Attention Block for self attention on input tensor
    """

    def __init__(
        self,
        reduction_ratio=16,
        transform_activation="linear",
        data_format="channels_last",
        name="global_context",
        **kwargs
    ):
        """
        Iniializes the Global Context block module
        Parameters
        ----------
        reduction_ratio : int
            Reduces the input filters by this factor for the
            bottleneck block of the transform submodule. Node: the reduction
            ratio must be set such that it divides the input number of channels
        transform_activation : string
            activation function to apply to the output
            of the transform block. Can be any string activation function available
            to tf.keras
        """
        super(GlobalContextBlock, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.transform_activation = transform_activation
        self._block_name = name
        self._data_format = data_format
        if self._data_format == "channels_first":
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]

    def build(self, input_shape):
        """
        Builds the block according to arguments
        """
        self._input_shape = tensor_shape.TensorShape(input_shape)
        if self._input_shape.dims[self._channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        # initialize the dimensions
        self._dim1, self._dim2, self._dim3 = None, None, None

        # check rank and calculate the input shape
        if len(self._input_shape) == 3:  # temporal/time-series data
            self._input_rank = 3
            self._batchsize, self._dim1, self._channels = self._input_shape.as_list()
        elif len(self._input_shape) == 4:  # spatial / image data
            self._input_rank = 4
            if self._channel_axis == 1:
                self._batchsize, self._channels, self._dim1, self._dim2 = (
                    self._input_shape.as_list()
                )
            else:
                self._batchsize, self._dim1, self._dim2, self._channels = (
                    self._input_shape.as_list()
                )
        elif len(self._input_shape) == 5:  # spatio-temporal / video / voxel data
            self._input_rank = 5
            if self._channel_axis == 1:
                self._batchsize, self._channels, self._dim1, self._dim2, self._dim3 = (
                    self._input_shape.as_list()
                )
            else:
                self._batchsize, self._dim1, self._dim2, self._dim3, self._channels = (
                    self._input_shape.as_list()
                )
        else:
            raise ValueError(
                "Input dimension has to be either 3 (temporal), 4 (spatial) or 5 (spatio-temporal)"
            )

        if self._input_rank > 3:
            self._flat_spatial_dim = -1 if self._data_format == "channels_first" else 1
        else:
            self._flat_spatial_dim = 1

        super(GlobalContextBlock, self).build(input_shape)

    def _convND(self, inputs, rank, channels, kernel=1):
        assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

        if rank == 3:
            x = tf.keras.layers.Conv1D(
                channels,
                kernel,
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal",
                data_format=self._data_format
            )(inputs)
        elif rank == 4:
            x = tf.keras.layers.Conv2D(
                channels,
                (kernel, kernel),
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal",
                data_format=self._data_format
            )(inputs)
        else:
            x = tf.keras.layers.Conv3D(
                channels,
                (kernel, kernel, kernel),
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal",
                data_format=self._data_format
            )(inputs)

        return x

    def _spatial_flattenND(self, ip, rank):
        assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

        if rank == 3:
            x = ip  # identity op for rank 3

        elif rank == 4:
            if self._channel_axis == 1:
                # [C, D1, D2] -> [C, D1 * D2]
                shape = [self._channels, self._dim1 * self._dim2]
            else:
                # [D1, D2, C] -> [D1 * D2, C]
                shape = [self._dim1 * self._dim2, self._channels]

            x = tf.keras.layers.Reshape(shape)(ip)

        else:
            if self._channel_axis == 1:
                # [C, D1, D2, D3] -> [C, D1 * D2 * D3]
                shape = [self._channels, self._dim1 * self._dim2 * self._dim3]
            else:
                # [D1, D2, D3, C] -> [D1 * D2 * D3, C]
                shape = [self._dim1 * self._dim2 * self._dim3, self._channels]

            x = tf.keras.layers.Reshape(shape)(ip)

        return x

    def _spatial_expandND(self, ip, rank):
        assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

        if rank == 3:
            x = tf.keras.layers.Permute((2, 1))(ip)  # identity op for rank 3

        elif rank == 4:
            if self._channel_axis == 1:
                # [C, D1, D2] -> [C, D1 * D2]
                shape = [-1, 1, 1]
            else:
                # [D1, D2, C] -> [D1 * D2, C]
                shape = [1, 1, -1]

            x = tf.keras.layers.Reshape(shape)(ip)

        else:
            if self._channel_axis == 1:
                # [C, D1, D2, D3] -> [C, D1 * D2 * D3]
                shape = [-1, 1, 1, 1]
            else:
                # [D1, D2, D3, C] -> [D1 * D2 * D3, C]
                shape = [1, 1, 1, -1]

            x = tf.keras.layers.Reshape(shape)(ip)

        return x

    def call(self, inputs):
        """
        Method `call` for keras layer
        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor to the layer

        Returns
        -------
            tf.Tensor
        """
        with tf.variable_scope(self._block_name):
            """ Context Modelling Block """
            # [B, ***, C] or [B, C, ***]
            input_flat = self._spatial_flattenND(inputs, self._input_rank)
            # [B, ..., C] or [B, C, ...]
            context = self._convND(inputs, self._input_rank, channels=1, kernel=1)
            # [B, ..., 1] or [B, 1, ...]
            context = self._spatial_flattenND(context, self._input_rank)
            # [B, ***, 1] or [B, 1, ***]
            context = tf.keras.layers.Softmax(axis=self._flat_spatial_dim)(context)

            # Compute context block outputs
            context = tf.keras.layers.dot(
                [input_flat, context], axes=self._flat_spatial_dim
            )
            # [B, C, 1]
            context = self._spatial_expandND(context, self._input_rank)
            # [B, C, 1...] or [B, 1..., C]

            """ Transform block """
            # Transform bottleneck
            # [B, C // R, 1...] or [B, 1..., C // R]
            transform = self._convND(
                context, self._input_rank, self._channels // self.reduction_ratio, kernel=1
            )
            # Group normalization acts as Layer Normalization when groups = 1
            transform = normalizers.GroupNormalization(
                groups=1, axis=self._channel_axis
            )(transform)
            transform = tf.keras.layers.Activation("Mish", name="mish_aftergroupnorm")(
                transform
            )

            # Transform output block
            # [B, C, 1...] or [B, 1..., C]
            transform = self._convND(transform, self._input_rank, self._channels, kernel=1)
            transform = tf.keras.layers.Activation(self.transform_activation)(transform)

            # apply context transform
            out = tf.keras.layers.add([inputs, transform])

        return out
