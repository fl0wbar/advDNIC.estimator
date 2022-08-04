from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.python.framework import tensor_shape
import tensorflow_compression as tfc
from . import convolution

K = tf.keras.backend


def shape_list(x):
    """Return list of dims, statically where possible."""
    # x = tf.convert_to_tensor(x)
    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i, dim in enumerate(static):
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret

def _convND(tensor, rank, channels):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"
    tf.logging.info("===<<< NLM -> _convND : tensor : %s : %s : %s >>>===" % (tensor.name, tensor.shape, channels))
    if rank == 3:
        x = tf.keras.layers.Conv1D(
            channels, 1, padding="same", use_bias=False, kernel_initializer="he_normal"
        )(tensor)
    elif rank == 4:
        x = tf.keras.layers.Conv2D(
            channels,
            (1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )(tensor)
    else:
        x = tf.keras.layers.Conv3D(
            channels,
            (1, 1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )(tensor)
    return x


class NonLocalBlock(tf.keras.layers.Layer):
    """
        keras.Layer Class for Non-Local Attention Block
        from the paper `Non-Local Neural Networks`
    """

    def __init__(
        self,
        name: str = "NonLocalBlock",
        bottleneck_dim: bool = None,
        compression: int = 2,
        mode: str = "embedded",
        add_residual: bool = True,
        data_format: str = "channels_last",
    ):
        """
            Initializes a Non-Local Attention Block
        :param name: variable scope name for the block
        :param bottleneck_dim: The dimension of the intermediate representation.
                                Can be `None` or a positive integer greater than 0.
                                If `None`, computes the intermediate dimension as half of
                                the input channel dimension.
        :param compression: None or positive integer. Compresses the intermediate
                            representation during the dot products to reduce memory
                            consumption. Default is set to 2, which states halve the
                            time/space/spatio-time dimension for the intermediate step.
                            Set to 1 to prevent computation compression.
                            None or 1 causes no reduction.
        :param mode: Mode of operation. Can be one of `embedded`, `gaussian`, `dot` or
                     `concatenate`.
        :param add_residual: Boolean value to decide if the residual connection should be
                             added or not.
                             Default is True for ResNets, and False for Self Attention.
        """
        super(NonLocalBlock, self).__init__()
        self._block_name = name
        self._bottleneck_dim = bottleneck_dim
        self._compression = compression
        self._nlmode = mode
        self._add_residual = add_residual
        self._data_format = data_format
        if self._data_format == "channels_first":
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]
        # get the channel and spatial dimensions from the input tensor
        self._channel_dim = 1 if K.image_data_format() == "channels_first" else -1

        # check mode of operation
        if self._nlmode not in ["gaussian", "embedded", "dot", "concatenate"]:
            raise ValueError(
                "`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`"
            )

        # check compression mode value
        if self._compression is None:
            self._compression = 1  # default to compression value of 1

    def _f_block(self, tensor):
        """ Builds the `f` block in non-local operation"""
        if self._nlmode == "gaussian":  # Gaussian instantiation
            x1 = tf.keras.layers.Reshape((-1, self._channels))(tensor)  # xi
            x2 = tf.keras.layers.Reshape((-1, self._channels))(tensor)  # xj
            f = tf.keras.layers.dot([x1, x2], axes=2)
            f = tf.keras.layers.Activation("softmax")(f)

        elif self._nlmode == "dot":  # Dot instantiation
            # theta path
            theta = _convND(tensor, self._input_rank, self._bottleneck_dim)
            theta = tf.keras.layers.Reshape((-1, self._bottleneck_dim))(theta)
            # phi patch
            phi = _convND(tensor, self._input_rank, self._bottleneck_dim)
            phi = tf.keras.layers.Reshape((-1, self._bottleneck_dim))(phi)

            f = tf.keras.layers.dot([theta, phi], axes=2)
            size = K.int_shape(f)
            # scale the values to make it size invariant
            f = tf.keras.layers.Lambda(lambda z: (1.0 / float(size[-1])) * z)(f)

        elif self._nlmode == "concatenate":  # Concatenation instantiation
            raise NotImplementedError("Concatenate model has not been implemented yet")

        else:  # Embedded Gaussian instantiation (default)
            # theta path
            theta = _convND(tensor, self._input_rank, self._bottleneck_dim)
            theta = tf.keras.layers.Reshape((-1, self._bottleneck_dim))(theta)
            # phi patch
            phi = _convND(tensor, self._input_rank, self._bottleneck_dim)
            phi = tf.keras.layers.Reshape((-1, self._bottleneck_dim))(phi)

            if self._compression > 1:
                # shielded computation
                phi = tf.keras.layers.MaxPool1D(self._compression)(phi)

            f = tf.keras.layers.dot([theta, phi], axes=2)
            f = tf.keras.layers.Activation("softmax")(f)

        return f

    def _g_block(self, tensor):
        """ Builds the `g` block in non-local operation"""
        g = _convND(tensor, self._input_rank, self._bottleneck_dim)
        g = tf.keras.layers.Reshape((-1, self._bottleneck_dim))(g)

        if self._compression > 1 and self._nlmode == "embedded":
            # shielded computation
            g = tf.keras.layers.MaxPool1D(self._compression)(g)

        return g

    def _fgexpand(self, f_tensor, g_tensor):
        """ Builds the `f*g` block with output reshape"""
        fgtensor = tf.keras.layers.dot([f_tensor, g_tensor], axes=[2, 1])
        # reshape to input tensor format
        if self._input_rank == 3:
            fgtensor = tf.keras.layers.Reshape((self._dim1, self._bottleneck_dim))(
                fgtensor
            )
        elif self._input_rank == 4:
            if self._channel_dim == -1:
                fgtensor = tf.keras.layers.Reshape(
                    (self._dim1, self._dim2, self._bottleneck_dim)
                )(fgtensor)
            else:
                fgtensor = tf.keras.layers.Reshape(
                    (self._bottleneck_dim, self._dim1, self._dim2)
                )(fgtensor)
        else:
            if self._channel_dim == -1:
                fgtensor = tf.keras.layers.Reshape(
                    (self._dim1, self._dim2, self._dim3, self._bottleneck_dim)
                )(fgtensor)
            else:
                fgtensor = tf.keras.layers.Reshape(
                    (self._bottleneck_dim, self._dim1, self._dim2, self._dim3)
                )(fgtensor)
        # project filters
        fgtensor = _convND(fgtensor, self._input_rank, self._channels)
        return fgtensor

    def build(self, input_shape):
        """
            Builds the Non-Local Attention Block according to the arguments
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
            if self._channel_dim == 1:
                self._batchsize, self._channels, self._dim1, self._dim2 = (
                    self._input_shape.as_list()
                )
            else:
                self._batchsize, self._dim1, self._dim2,  self._channels = (
                    self._input_shape.as_list()
                )
        elif len(self._input_shape) == 5:  # spatio-temporal / video / voxel data
            self._input_rank = 5
            if self._channel_dim == 1:
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

        # verify if correct intermediate dimension is specified
        if self._bottleneck_dim is None:
            self._bottleneck_dim = self._channels // 2
            # tf.logging.info("===<<< NonLocalBlock Bottleneck Dimension : %s >>>===" % (self._bottleneck_dim))
            if self._bottleneck_dim < 1:
                self._bottleneck_dim = 1
        else:
            self._bottleneck_dim = int(self._bottleneck_dim)
            if self._bottleneck_dim < 1:
                raise ValueError(
                    "`intermediate_dim` must be either `None` or positive integer greater than 1."
                )

        self._f = self._f_block
        self._g = self._g_block
        self._fg = self._fgexpand

        super(NonLocalBlock, self).build(input_shape)

    def call(self, inputs):
        """Implementation of call().

            Args:
              inputs: the inputs tensor.

            Returns:
              A output tensor.
        """
        self._shape = shape_list(inputs)
        # Save the spatial tensor dimensions
        # check rank and calculate the input shape
        if len(self._shape) == 3:  # temporal/time-series data
            self._input_rank = 3
            self._batchsize, self._dim1, self._channels = self._shape
        elif len(self._shape) == 4:  # spatial / image data
            self._input_rank = 4
            if self._channel_dim == 1:
                self._batchsize, self._channels, self._dim1, self._dim2 = (
                    self._shape
                )
            else:
                self._batchsize, self._dim1, self._dim2, self._channels = (
                    self._shape
                )
        elif len(self._shape) == 5:  # spatio-temporal / video / voxel data
            self._input_rank = 5
            if self._channel_dim == 1:
                self._batchsize, self._channels, self._dim1, self._dim2, self._dim3 = (
                    self._shape
                )
            else:
                self._batchsize, self._dim1, self._dim2, self._dim3, self._channels = (
                    self._shape
                )
        else:
            raise ValueError(
                "Input dimension has to be either 3 (temporal), 4 (spatial) or 5 (spatio-temporal)"
            )

        with tf.variable_scope(self._block_name):
            tf.logging.info(
                "NonLocalBlock input: %s, shape: %s" % (inputs.name, inputs.shape)
            )

            with tf.variable_scope("nonlocal_f"):
                x_f = self._f(inputs)
                tf.logging.info("nonlocal_F: %s, shape: %s" % (x_f.name, x_f.shape))

            with tf.variable_scope("nonlocal_g"):
                x_g = self._g(inputs)
                tf.logging.info("nonlocal_G: %s, shape: %s" % (x_g.name, x_g.shape))

            with tf.variable_scope("nonlocal_fg"):
                x = self._fg(x_f, x_g)
                tf.logging.info("nonlocal_FG: %s, shape: %s" % (x.name, x.shape))

            if self._add_residual:
                x = tf.keras.layers.add([x, inputs])
                tf.logging.info("nonlocal_Project: %s shape: %s" % (x.name, x.shape))

            return x


class NLAM(tf.keras.layers.Layer):
    """
        keras.Layer class for Non-Local Attention Feature Extraction Module
    """

    def __init__(
        self,
        num_filters,
        kernel_size,
        name: str,
        data_format: str = "channels_last",
        relu_fn=tf.nn.swish,
        *args,
        **kwargs
    ):
        super(NLAM, self).__init__(*args, **kwargs)
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._block_name = name
        self._data_format = data_format
        if self._data_format == "channels_first":
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]
        self._relu_fn = relu_fn or tf.nn.swish

        # Build the blocks according to the arguments
        self._build()

    def _build(self):
        """
            Builds the block according to the arguments
        """
        self._nlm = NonLocalBlock(
            name="NLM", mode="embedded", compression=2, add_residual=True
        )
        self._mbresblock0 = convolution.MBConvBlock(
            input_filters=self._num_filters,
            output_filters=self._num_filters,
            kernel_size=3,
            strides=[1, 1],
            expand_ratio=1,
            se_ratio=0.25,
            id_skip=True,
            name="NLAM_mbresconv0",
        )
        self._downsample = convolution.MBConvBlock(
            input_filters=self._num_filters,
            output_filters=self._num_filters,
            kernel_size=3,
            strides=[2, 2],
            expand_ratio=1,
            se_ratio=0.25,
            id_skip=True,
            name="NLAM_downsample",
        )
        self._mbresblock1 = convolution.MBConvBlock(
            input_filters=self._num_filters,
            output_filters=self._num_filters,
            kernel_size=3,
            strides=[1, 1],
            expand_ratio=1,
            se_ratio=0.25,
            id_skip=True,
            name="NLAM_mbresconv1",
        )
        self._upsample = convolution.MBUpsampleConvBlock(
            input_filters=self._num_filters,
            output_filters=self._num_filters,
            kernel_size=3,
            scale=2,
            expand_ratio=1,
            se_ratio=0.25,
            id_skip=True,
            use_blur=True,
            use_batchnorm=False,
            name="NLAM_upsample",
        )
        self._mbresblock2 = convolution.MBConvBlock(
            input_filters=self._num_filters,
            output_filters=self._num_filters,
            kernel_size=3,
            strides=[1, 1],
            expand_ratio=1,
            se_ratio=0.25,
            id_skip=True,
            name="NLAM_mbresconv2",
        )
        self._conv1x1 = tf.keras.layers.Conv2D(
            self._num_filters,
            (1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )

        self._downsample_p2 = convolution.MBConvBlock(
            input_filters=self._num_filters,
            output_filters=self._num_filters,
            kernel_size=3,
            strides=[2, 2],
            expand_ratio=1,
            se_ratio=0.25,
            id_skip=True,
            name="NLAM_downsample",
        )
        self._mbresblock_p2 = convolution.MBConvBlock(
            input_filters=self._num_filters,
            output_filters=self._num_filters,
            kernel_size=3,
            strides=[1, 1],
            expand_ratio=1,
            se_ratio=0.25,
            id_skip=True,
            name="NLAM_mbresconv1",
        )
        self._upsample_p2 = convolution.MBUpsampleConvBlock(
            input_filters=self._num_filters,
            output_filters=self._num_filters,
            kernel_size=3,
            scale=2,
            expand_ratio=1,
            se_ratio=0.25,
            id_skip=True,
            use_blur=True,
            use_batchnorm=False,
            name="NLAM_upsample",
        )

    def call(self, inputs):
        """Implementation of call().

            Args:
              inputs: the inputs tensor.

            Returns:
              A output tensor.
        """
        with tf.variable_scope(self._block_name):
            block_residual = inputs
            with tf.variable_scope("nlam_p1"):
                x1 = self._nlm(inputs)
                x1 = self._mbresblock0(x1)
                res_x1 = x1
                x1 = self._downsample(x1)
                x1 = self._mbresblock1(x1)
                x1 = self._upsample(x1)
                x1 = tf.add(x1, res_x1)
                x1 = self._mbresblock2(x1)
                x1 = self._conv1x1(x1)
                x1 = tf.nn.sigmoid(x1)

            with tf.variable_scope("nlam_p2"):
                x2 = self._downsample_p2(inputs)
                x2 = self._mbresblock_p2(x2)
                x2 = self._upsample_p2(x2)

            x = tf.multiply(x1, x2)
            # adding the skip connection
            x = tf.add(x, block_residual)

        return x
