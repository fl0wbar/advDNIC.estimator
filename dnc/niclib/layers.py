from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.utils import get_custom_objects
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
from tensorflow.keras.layers import Activation
import tensorflow_compression as tfc

from . import subtransforms
from . import normalizers


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


def conv2d_weightnorm(
    filters,
    kernel_size,
    strides,
    dilation_rate,
    activation,
    use_bias,
    kernel_initializer,
    padding,
    data_format,
    **kwargs
):
    return normalizers.WeightNormalization(
        tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            padding=padding,
            data_format=data_format,
            **kwargs
        ),
        data_init=False,
    )


def sigconv2d_weightnorm(
    filters,
    kernel_support,
    corr,
    strides_down=1,
    strides_up=1,
    padding="valid",
    extra_pad_end=True,
    channel_separable=False,
    data_format="channels_last",
    activation=None,
    use_bias=False,
    use_explicit=False,
    **kwargs
):
    return normalizers.WeightNormalization(
        tfc.SignalConv2D(
            filters=filters,
            kernel_support=kernel_support,
            corr=corr,
            strides_down=strides_down,
            strides_up=strides_up,
            padding=padding,
            extra_pad_end=extra_pad_end,
            channel_separable=channel_separable,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            use_explicit=use_explicit,
            **kwargs
        )
    )


def _conv_layer(filters, kernel_size, strides=(1, 1), padding="same", name="base_conv"):
    return tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=True,
        kernel_initializer="he_normal",
        name=name,
    )


def _normalize_depth_vars(depth_k, depth_v, filters):
    """
    Accepts depth_k and depth_v as either floats or integers
    and normalizes them to integers.
    Args:
        depth_k: float or int.
        depth_v: float or int.
        filters: number of output filters.
    Returns:
        depth_k, depth_v as integers.
    """

    if type(depth_k) == float:
        depth_k = int(filters * depth_k)
    else:
        depth_k = int(depth_k)

    if type(depth_v) == float:
        depth_v = int(filters * depth_v)
    else:
        depth_v = int(depth_v)

    return depth_k, depth_v


class AttentionAugmentation2D(tf.keras.layers.Layer):
    def __init__(
        self, depth_k, depth_v, num_heads, relative=True, name="attn_aug2d", **kwargs
    ):
        """
        Applies attention augmentation on a convolutional layer
        output.

        Args:
            depth_k: float or int. Number of filters for k.
            Computes the number of filters for `v`.
            If passed as float, computed as `filters * depth_k`.
        depth_v: float or int. Number of filters for v.
            Computes the number of filters for `k`.
            If passed as float, computed as `filters * depth_v`.
        num_heads: int. Number of attention heads.
            Must be set such that `depth_k // num_heads` is > 0.
        relative: bool, whether to use relative encodings.

        Raises:
            ValueError: if depth_v or depth_k is not divisible by
                num_heads.

        Returns:
            Output tensor of shape
            -   [Batch, Height, Width, Depth_V] if
                channels_last data format.
            -   [Batch, Depth_V, Height, Width] if
                channels_first data format.
        """
        super(AttentionAugmentation2D, self).__init__(**kwargs)

        if depth_k % num_heads != 0:
            raise ValueError(
                "`depth_k` (%d) is not divisible by `num_heads` (%d)"
                % (depth_k, num_heads)
            )

        if depth_v % num_heads != 0:
            raise ValueError(
                "`depth_v` (%d) is not divisible by `num_heads` (%d)"
                % (depth_v, num_heads)
            )

        if depth_k // num_heads < 1.0:
            raise ValueError(
                "depth_k / num_heads cannot be less than 1 ! "
                "Given depth_k = %d, num_heads = %d" % (depth_k, num_heads)
            )

        if depth_v // num_heads < 1.0:
            raise ValueError(
                "depth_v / num_heads cannot be less than 1 ! "
                "Given depth_v = %d, num_heads = %d" % (depth_v, num_heads)
            )

        self.depth_k = depth_k
        self.depth_v = depth_v
        self.num_heads = num_heads
        self.relative = relative
        self._block_name = name

        self.axis = 1 if K.image_data_format() == "channels_first" else -1

    def build(self, input_shape):
        self._shape = input_shape

        # normalize the format of depth_v and depth_k
        self.depth_k, self.depth_v = _normalize_depth_vars(
            self.depth_k, self.depth_v, input_shape
        )

        if self.axis == 1:
            _, channels, height, width = input_shape
        else:
            _, height, width, channels = input_shape

        if self.relative:
            dk_per_head = self.depth_k // self.num_heads

            if dk_per_head == 0:
                print("dk per head", dk_per_head)

            self.key_relative_w = self.add_weight(
                "key_rel_w",
                shape=[2 * width - 1, dk_per_head],
                initializer=tf.keras.initializers.RandomNormal(
                    stddev=dk_per_head ** -0.5
                ),
            )

            self.key_relative_h = self.add_weight(
                "key_rel_h",
                shape=[2 * height - 1, dk_per_head],
                initializer=tf.keras.initializers.RandomNormal(
                    stddev=dk_per_head ** -0.5
                ),
            )

        else:
            self.key_relative_w = None
            self.key_relative_h = None

    def call(self, inputs, **kwargs):
        with tf.variable_scope(self._block_name):
            if self.axis == 1:
                # If channels first, force it to be channels last for these ops
                inputs = K.permute_dimensions(inputs, [0, 2, 3, 1])

            q, k, v = tf.split(
                inputs, [self.depth_k, self.depth_k, self.depth_v], axis=-1
            )

            q = self.split_heads_2d(q)
            k = self.split_heads_2d(k)
            v = self.split_heads_2d(v)

            # scale query
            depth_k_heads = self.depth_k / self.num_heads
            q *= depth_k_heads ** -0.5

            # [Batch, num_heads, height * width, depth_k or depth_v] if axis == -1
            qk_shape = [
                self._batch,
                self.num_heads,
                self._height * self._width,
                self.depth_k // self.num_heads,
            ]
            v_shape = [
                self._batch,
                self.num_heads,
                self._height * self._width,
                self.depth_v // self.num_heads,
            ]
            flat_q = K.reshape(q, K.stack(qk_shape))
            flat_k = K.reshape(k, K.stack(qk_shape))
            flat_v = K.reshape(v, K.stack(v_shape))

            # [Batch, num_heads, HW, HW]
            logits = tf.matmul(flat_q, flat_k, transpose_b=True)

            # Apply relative encodings
            if self.relative:
                h_rel_logits, w_rel_logits = self.relative_logits(q)
                logits += h_rel_logits
                logits += w_rel_logits

            weights = K.softmax(logits, axis=-1)
            attn_out = tf.matmul(weights, flat_v)

            attn_out_shape = [
                self._batch,
                self.num_heads,
                self._height,
                self._width,
                self.depth_v // self.num_heads,
            ]
            attn_out_shape = K.stack(attn_out_shape)
            attn_out = K.reshape(attn_out, attn_out_shape)
            attn_out = self.combine_heads_2d(attn_out)
            # [batch, height, width, depth_v]

            if self.axis == 1:
                # return to [batch, depth_v, height, width] for channels first
                attn_out = K.permute_dimensions(attn_out, [0, 3, 1, 2])

        return attn_out

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = self.depth_v
        return tuple(output_shape)

    def split_heads_2d(self, ip):
        tensor_shape = K.shape(ip)

        # batch, height, width, channels for axis = -1
        tensor_shape = [tensor_shape[i] for i in range(len(self._shape))]

        batch = tensor_shape[0]
        height = tensor_shape[1]
        width = tensor_shape[2]
        channels = tensor_shape[3]

        # Save the spatial tensor dimensions
        self._batch = batch
        self._height = height
        self._width = width

        ret_shape = K.stack(
            [batch, height, width, self.num_heads, channels // self.num_heads]
        )
        split = K.reshape(ip, ret_shape)
        transpose_axes = (0, 3, 1, 2, 4)
        split = K.permute_dimensions(split, transpose_axes)

        return split

    def relative_logits(self, q):
        shape = K.shape(q)
        # [batch, num_heads, H, W, depth_v]
        shape = [shape[i] for i in range(5)]

        height = shape[2]
        width = shape[3]

        rel_logits_w = self.relative_logits_1d(
            q, self.key_relative_w, height, width, transpose_mask=[0, 1, 2, 4, 3, 5]
        )

        rel_logits_h = self.relative_logits_1d(
            K.permute_dimensions(q, [0, 1, 3, 2, 4]),
            self.key_relative_h,
            width,
            height,
            transpose_mask=[0, 1, 4, 2, 5, 3],
        )

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, transpose_mask):
        rel_logits = tf.einsum("bhxyd,md->bhxym", q, rel_k)
        rel_logits = K.reshape(rel_logits, [-1, self.num_heads * H, W, 2 * W - 1])
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = K.reshape(rel_logits, [-1, self.num_heads, H, W, W])
        rel_logits = K.expand_dims(rel_logits, axis=3)
        rel_logits = K.tile(rel_logits, [1, 1, 1, H, 1, 1])
        rel_logits = K.permute_dimensions(rel_logits, transpose_mask)
        rel_logits = K.reshape(rel_logits, [-1, self.num_heads, H * W, H * W])
        return rel_logits

    def rel_to_abs(self, x):
        shape = K.shape(x)
        shape = [shape[i] for i in range(3)]
        B, Nh, L, = shape
        col_pad = K.zeros(K.stack([B, Nh, L, 1]))
        x = K.concatenate([x, col_pad], axis=3)
        flat_x = K.reshape(x, [B, Nh, L * 2 * L])
        flat_pad = K.zeros(K.stack([B, Nh, L - 1]))
        flat_x_padded = K.concatenate([flat_x, flat_pad], axis=2)
        final_x = K.reshape(flat_x_padded, [B, Nh, L + 1, 2 * L - 1])
        final_x = final_x[:, :, :L, L - 1 :]
        return final_x

    def combine_heads_2d(self, inputs):
        # [batch, num_heads, height, width, depth_v // num_heads]
        transposed = K.permute_dimensions(inputs, [0, 2, 3, 1, 4])
        # [batch, height, width, num_heads, depth_v // num_heads]
        shape = K.shape(transposed)
        shape = [shape[i] for i in range(5)]

        a, b = shape[-2:]
        ret_shape = K.stack(shape[:-2] + [a * b])
        # [batch, height, width, depth_v]
        return K.reshape(transposed, ret_shape)

    def get_config(self):
        config = {
            "depth_k": self.depth_k,
            "depth_v": self.depth_v,
            "num_heads": self.num_heads,
            "relative": self.relative,
        }
        base_config = super(AttentionAugmentation2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AugmentedConv2D(tf.keras.layers.Layer):
    """
        A class for tf.keras.layer of Attention Augmented Convolutional Layer
    """

    def __init__(
        self,
        filters: int,
        kernel_size=(3, 3),
        strides=(1, 1),
        depth_k=0.2,
        depth_v=0.2,
        num_heads: int = 8,
        relative_encodings: bool = True,
        name: str = "aug_conv2d",
        **kwargs
    ):
        """
            Initializes the AugmentedConv2D block

        :param filters: (int) number of output convolutional filters.
        :param kernel_size: (list/tuple) convolutional kernel size {tuple/list of 2 numbers}.
        :param strides: (list/tuple) strides of the convolution.
        :param depth_k: (float/int) Number of filters for k.
                                Computes the number of filters for `v`.
                                If passed as float, computed as `filters * depth_k`.
        :param depth_v: (float/int) Number of filters for v.
                                Computes the number of filters for `k`.
                                If passed as float, computed as `filters * depth_v`.
        :param num_heads: (int) Number of attention heads.
                                Must be set such that `depth_k // num_heads` is > 0.
        :param relative_encodings: (bool) Whether to use relative encodings or not.
        :param name: (str) Name of the variable scope block
        :param kwargs:
        """
        super(AugmentedConv2D, self).__init__(**kwargs)
        self._output_filters = filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._depth_k = depth_k
        self._depth_v = depth_v
        self._num_heads = num_heads
        self._relative_encodings = relative_encodings
        self._block_name = name
        self._channel_axis = 1 if K.image_data_format() == "channels_first" else -1

        # Build the blocks according to the arguments
        self._build()

    def _build(self):
        """
            Builds the block according to the arguments
        """
        norm_depth_k, norm_depth_v = _normalize_depth_vars(
            self._depth_k, self._depth_v, self._output_filters
        )
        self._out_convolution = _conv_layer(
            filters=(self._output_filters - norm_depth_v),
            kernel_size=self._kernel_size,
            strides=self._strides,
            name="out_convolution",
        )
        # Augmented Attention Block
        self._qkv_convolution = _conv_layer(
            filters=(2 * norm_depth_k + norm_depth_v),
            kernel_size=(1, 1),
            strides=self._strides,
            name="qkv_convolution",
        )
        self._attn_augmentation = AttentionAugmentation2D(
            depth_k=norm_depth_k,
            depth_v=norm_depth_v,
            num_heads=self._num_heads,
            relative=self._relative_encodings,
        )
        self._out_attn_convolution = _conv_layer(
            filters=norm_depth_v,
            kernel_size=(1, 1),
            strides=(1, 1),
            name="out_attn_convolution",
        )

    def call(self, inputs):
        """Implementation of call().

            Args:
              inputs: the inputs tensor.

            Returns:
              A output tensor.
        """
        with tf.variable_scope(self._block_name):

            x = self._out_convolution(inputs)

            # Augmented Attention Block
            attn_x = self._qkv_convolution(inputs)
            attn_x = self._attn_augmentation(attn_x)
            attn_x = self._out_attn_convolution(attn_x)

            x_out = K.concatenate([x, attn_x], axis=self._channel_axis)

        return x_out


class ReflectionPad2D(tf.keras.layers.Layer):
    def __init__(
        self,
        paddings,
        data_format: str = "channels_last",
        name: str = "reflection_pad2d",
        **kwargs
    ):
        self._paddings = paddings
        self._tp, self._bp, self._lp, self._rp = self._paddings
        self._data_format = data_format
        if self._data_format == "channels_first":
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
            self._pad_mode = [
                [0, 0],
                [0, 0],
                [self._tp, self._bp],
                [self._lp, self._rp],
            ]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]
            self._pad_mode = [
                [0, 0],
                [self._tp, self._bp],
                [self._lp, self._rp],
                [0, 0],
            ]
        self._layer_name = name
        super(ReflectionPad2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        if self._data_format == "channels_first":
            out_shape = tf.TensorShape(
                [
                    shape[0],
                    shape[1],
                    shape[2] + self._tp + self._bp,
                    shape[3] + self._lp + self._rp,
                ]
            )

        else:
            out_shape = tf.TensorShape(
                [
                    shape[0],
                    shape[1] + self._tp + self._bp,
                    shape[2] + self._lp + self._rp,
                    shape[3],
                ]
            )

        return out_shape

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        with tf.variable_scope(self._layer_name):
            reflect_pad = tf.pad(inputs, self._pad_mode, "REFLECT")
        return reflect_pad


class SymmetricPad2D(tf.keras.layers.Layer):
    def __init__(
        self,
        paddings,
        data_format: str = "channels_last",
        name: str = "symmetric_pad2d",
        **kwargs
    ):
        self._paddings = paddings
        self._tp, self._bp, self._lp, self._rp = self._paddings
        self._data_format = data_format
        if self._data_format == "channels_first":
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
            self._pad_mode = [
                [0, 0],
                [0, 0],
                [self._tp, self._bp],
                [self._lp, self._rp],
            ]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]
            self._pad_mode = [
                [0, 0],
                [self._tp, self._bp],
                [self._lp, self._rp],
                [0, 0],
            ]
        self._layer_name = name
        super(SymmetricPad2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        if self._data_format == "channels_first":
            out_shape = tf.TensorShape(
                [
                    shape[0],
                    shape[1],
                    shape[2] + self._tp + self._bp,
                    shape[3] + self._lp + self._rp,
                ]
            )

        else:
            out_shape = tf.TensorShape(
                [
                    shape[0],
                    shape[1] + self._tp + self._bp,
                    shape[2] + self._lp + self._rp,
                    shape[3],
                ]
            )

        return out_shape

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        with tf.variable_scope(self._layer_name):
            symmetric_pad = tf.pad(inputs, self._pad_mode, "SYMMETRIC")
        return symmetric_pad


class ConstantPad2D(tf.keras.layers.Layer):
    def __init__(
        self,
        paddings,
        constant_value=0,
        data_format: str = "channels_last",
        name: str = "constant_pad2d",
        **kwargs
    ):
        self._paddings = paddings
        self._tp, self._bp, self._lp, self._rp = self._paddings
        self._constant_value = constant_value
        self._data_format = data_format
        if self._data_format == "channels_first":
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
            self._pad_mode = [
                [0, 0],
                [0, 0],
                [self._tp, self._bp],
                [self._lp, self._rp],
            ]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]
            self._pad_mode = [
                [0, 0],
                [self._tp, self._bp],
                [self._lp, self._rp],
                [0, 0],
            ]
        self._layer_name = name
        super(ConstantPad2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        if self._data_format == "channels_first":
            out_shape = tf.TensorShape(
                [
                    shape[0],
                    shape[1],
                    shape[2] + self._tp + self._bp,
                    shape[3] + self._lp + self._rp,
                ]
            )

        else:
            out_shape = tf.TensorShape(
                [
                    shape[0],
                    shape[1] + self._tp + self._bp,
                    shape[2] + self._lp + self._rp,
                    shape[3],
                ]
            )

        return out_shape

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        with tf.variable_scope(self._layer_name):
            constant_pad = tf.pad(
                inputs, self._pad_mode, "CONSTANT", constant_values=self._constant_value
            )
        return constant_pad


class ICNR:
    """ICNR initializer for checkerboard artifact free sub pixel convolution
    Ref:
     [1] Andrew Aitken et al. Checkerboard artifact free sub-pixel convolution
     https://arxiv.org/pdf/1707.02937.pdf)
     Code taken from: https://github.com/kostyaev/ICNR
    Args:
    initializer: initializer used for sub kernels (orthogonal, glorot uniform, etc.)
    scale: scale factor of sub pixel convolution
    """

    def __init__(
        self,
        initializer=tf.initializers.he_normal,
        data_format: str = "channels_last",
        scale: int = 2,
        name: str = "icnr_init",
    ):
        self.scale = scale
        self.initializer = initializer
        self._data_format = data_format
        if self._data_format == "channels_first":
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]
        self._name = name

    def __call__(self, shape, dtype, partition_info=None):
        with tf.variable_scope(self._name):
            shape = list(shape)
            if self.scale == 1:
                return self.initializer(shape)

            new_shape = shape[:3] + [shape[3] // (self.scale ** 2)]
            x = self.initializer(new_shape, dtype, partition_info)
            x = tf.transpose(x, perm=[2, 0, 1, 3])
            x = tf.image.resize_nearest_neighbor(
                x, size=(shape[0] * self.scale, shape[1] * self.scale)
            )
            x = tf.space_to_depth(x, block_size=self.scale)
            x = tf.transpose(x, perm=[1, 2, 0, 3])
            return x


class SubpixelConv1D(tf.keras.layers.Layer):
    """
        A tf.keras.layer for 1D sub-pixel up-sampling
        Calls a TensorFlow function that directly implements this functionality.
        better if input has dim (batch, width, r)
    """

    def __init__(
        self,
        scale: int = 2,
        in_channels: int = None,
        data_format: str = "channels_last",
        name: str = "subpixel_conv1d",
    ):
        super(SubpixelConv1D, self).__init__()
        self._scale = scale
        self._in_channels = in_channels
        self._out_channels = int(self._in_channels / self._scale)
        self._data_format = data_format
        if self._data_format == "channels_first":
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]
        self._batch_dim = 0
        self._layer_name = name

        if self._in_channels is not None:
            # Build the blocks according to the arguments
            self._build(None)

    def _build(self, inputs_shape):
        if inputs_shape is not None:
            self._in_channels = inputs_shape[self._channel_axis]
        self._out_channels = int(self._in_channels / self._scale)
        pass

    def _pixelshuffle(self, in_tensor, scale):
        out_tensor = tf.transpose(
            a=in_tensor,
            perm=[self._spatial_dims[1], self._spatial_dims[0], self._batch_dim],
        )
        out_tensor = tf.batch_to_space(
            input=out_tensor, block_shape=[scale], crops=[[0, 0]]
        )
        out_tensor = tf.transpose(
            a=out_tensor,
            perm=[self._spatial_dims[1], self._spatial_dims[0], self._batch_dim],
        )
        return out_tensor

    def call(self, inputs):
        with tf.variable_scope(self._layer_name):
            outputs = self._pixelshuffle(inputs, scale=self._scale)
        return outputs


class SubpixelConv2D(tf.keras.layers.Layer):
    """
        A keras Layer for 2D sub-pixel upsampling
    """

    def __init__(
        self,
        scale: int = 2,
        in_channels: int = None,
        out_channels: int = None,
        data_format: str = "channels_last",
        use_periodic_resample: bool = True,
        name: str = "subpixel_conv2d",
    ):
        super(SubpixelConv2D, self).__init__()
        self._scale = scale
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._layer_name = name
        self._data_format = data_format
        if self._data_format == "channels_first":
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]
        self._periodic_resample = use_periodic_resample

        if self._in_channels is not None:
            self._build(None)

    def _build(self, inputs_shape):

        if inputs_shape is not None:
            self._in_channels = inputs_shape[self._channel_axis]

        if (self._in_channels / (self._scale ** 2)) % 1 != 0:
            raise Exception(
                "SubpixelConv2d: The number of input channels == (scale x scale) x The number of output channels"
            )

        self._out_channels = int(self._in_channels / (self._scale ** 2))

    def _pixelshuffle(self, tensor, scale, out_channels):

        _error_log = "SubpixelConv2d: The number of input channels == (scale x scale) x The number of output channels"

        if self._out_channels >= 1:
            if (
                int(tensor.get_shape()[self._channel_axis])
                != (scale ** 2) * out_channels
            ):
                raise Exception(_error_log)

            tensor = tf.depth_to_space(input=tensor, block_size=scale)
        else:
            raise RuntimeError(_error_log)

        return tensor

    def call(self, inputs):
        with tf.variable_scope(self._layer_name):
            outputs = self._pixelshuffle(
                tensor=inputs, scale=self._scale, out_channels=self._out_channels
            )
        return outputs


def conv1x1(x, output_dim, name="conv1x1"):
    """Builds graph for a if specified spectrally normalized 1 by 1 convolution.
      This is used in the context of non-local networks to reduce channel count for
      strictly computational reasons.
      Args:
        x: A 4-D tensorflow tensor.
        output_dim: An integer representing desired channel count in the output.
        name: String to pass to the variable scope context.
      Returns:
        A new volume with the same batch, height, and width as the input.
    """
    with tf.compat.v1.variable_scope(name):
        w = tf.compat.v1.get_variable(
            "weights",
            [1, 1, x.get_shape()[-1], output_dim],
            initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1.0, mode="fan_avg", distribution="uniform"
            ),
        )
        conv = tf.nn.conv2d(input=x, filters=w, strides=[1, 1, 1, 1], padding="SAME")
        return conv


class SelfAttention(tf.keras.layers.Layer):
    """ A keras Layer for Self-Attention from SAGAN paper (https://arxiv.org/pdf/1805.08318.pdf)"""

    def __init__(
        self,
        name: str = "self_attn",
        init=tf.initializers.glorot_normal,
        data_format: str = "channels_last",
        **kwargs
    ):
        super(SelfAttention, self).__init__(**kwargs)
        self._layer_name = name
        self._initializer = init
        self._data_format = data_format
        if self._data_format == "channels_first":
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        with tf.variable_scope(self._layer_name):
            input_shape = inputs.shape.as_list()
            h = input_shape[self._spatial_dims[0]]
            w = input_shape[self._spatial_dims[1]]
            num_channels = input_shape[self._channel_axis]
            location_num = h * w
            downsampled_num = location_num // 4

            # theta path
            theta = conv1x1(inputs, num_channels // 8, name="conv1x1_theta")
            theta = tf.reshape(theta, [-1, location_num, num_channels // 8])

            # phi path
            phi = conv1x1(inputs, num_channels // 8, name="conv1x1_phi")
            phi = tf.compat.v1.layers.max_pooling2d(
                inputs=phi, pool_size=[2, 2], strides=2
            )
            phi = tf.reshape(phi, [-1, downsampled_num, num_channels // 8])

            attn = tf.matmul(theta, phi, transpose_b=True)
            attn = tf.nn.softmax(attn)

            # g path
            g = conv1x1(inputs, num_channels // 2, name="conv1x1_g")
            g = tf.compat.v1.layers.max_pooling2d(inputs=g, pool_size=[2, 2], strides=2)
            g = tf.reshape(g, [-1, downsampled_num, num_channels // 2])

            attn_g = tf.matmul(attn, g)
            attn_g = tf.reshape(attn_g, [-1, h, w, num_channels // 2])
            sigma = tf.compat.v1.get_variable(
                "sigma_ratio", [], initializer=tf.compat.v1.initializers.constant(0.0)
            )

            attn_g = conv1x1(attn_g, num_channels, name="conv1x1_attn")

            return inputs + sigma * attn_g


class UpsampleLayer(tf.keras.layers.Layer):
    """ Upsamples the given inputs"""

    def __init__(
        self,
        filters,
        kernel_size=(3, 3),
        strides=[2, 2],
        apply_activ=False,
        activation=tf.keras.layers.LeakyReLU(alpha=0.2),
        method="conv2d_transpose",
        norm_type="batch",
        batch_norm_momentum: float = 0.99,
        batch_norm_epsilon: float = 1e-3,
        use_ghostbatchnorm: bool = False,
        self_attention: bool = False,
        data_format="channels_last",
        name="upsamplelayer",
    ):
        """

        :param output_filters: (int) The number of output filters.
        :param stride: A list of 2 scalars or a 1x2 Tensor indicating the scale,
                      relative to the inputs, of the output dimensions. For example, if kernel
                      size is [2, 3], then the output height and width will be twice and three
                      times the input size.
        :param method: The upsampling method: 'nn_upsample_conv',
                        'bilinear_upsample_conv', 'conv2d_transpose' or 'subpixel_conv'.
        :param batch_norm_momentum: (float) momentum value for batch normalization.
        :param batch_norm_epsilon: (float) eps value for batch normalization.
        :param use_ghostbatchnorm: (bool) perform "Ghost Batch Normalization",
                                          which creates virtual sub-batches
                                          which are each normalized separately
                                          (with shared gamma, beta, and moving statistics)
        :param leaky: (float) LeakyRELU alpha value (negative slope)
        :param transpose: (bool) if True uses transpose conv
        :param self_attention: (bool) if True uses self-attention layer / attention-augmented conv2d
        :param name: (str) name of the variable scope
        """
        super(UpsampleLayer, self).__init__()
        self._output_filters = filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._apply_activ = apply_activ
        self._activ = activation
        self._method = method
        self._norm_type = norm_type
        self._batch_norm_momentum = batch_norm_momentum
        self._batch_norm_eps = batch_norm_epsilon
        self._use_ghostbn = use_ghostbatchnorm
        self._apply_attention = self_attention
        self._data_format = data_format
        if self._data_format == "channels_first":
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]
        self._layer_name = name

        # Build the blocks according to the arguments
        self._build()

    def _build(self):
        """ Builds the layer """

        # Reflection pad by 1 in spatial dimensions (axes 1, 2 = h, w) to make a
        # 3x3 "valid" convolution produce an output with the same dimension as the
        # input.
        self._spatial_pad_1 = np.array([[0, 0], [1, 1], [1, 1], [0, 0]])
        if self._method == "conv2d_transpose":
            self._conv = tf.keras.layers.Conv2DTranspose(
                filters=self._output_filters,
                kernel_size=self._kernel_size,
                strides=self._strides,
            )
        if self._method == "subpixel_conv":
            self._conv = tf.keras.layers.Conv2D(
                filters=self._output_filters * (self._strides[0] ** 2),
                kernel_size=self._kernel_size,
                strides=(1, 1),
            )
        else:
            self._conv = tf.keras.layers.Conv2D(
                filters=self._output_filters,
                kernel_size=self._kernel_size,
                strides=(1, 1),
            )

        if self._norm_type == "weight":
            self._conv = normalizers.WeightNormalization(self._conv, data_init=False)
        elif self._norm_type == "spectral":
            self._conv = normalizers.SpectralNormalization(self._conv)
        elif self._norm_type == "batch":
            self._batchnorm = tf.keras.layers.BatchNormalization(
                axis=self._channel_axis,
                momentum=self._batch_norm_momentum,
                epsilon=self._batch_norm_eps,
                virtual_batch_size=self._virtual_batch_size,
            )
            self._conv = self._batchnorm(self._conv)
        else:
            self._conv = self._conv

        if self._apply_attention:
            self._selfattention = SelfAttention(
                name="conv_self_attn", init=self._kernel_initializer
            )

    def call(self, inputs):
        with tf.variable_scope(self._layer_name):
            tf.logging.info(
                "UpsampleLayer-/Input: %s shape: %s" % (inputs.name, inputs.shape)
            )
            input_shape = tf.shape(inputs)
            height = input_shape[self._spatial_dims[0]]
            width = input_shape[self._spatial_dims[1]]

            if self._method == "nn_upsample_conv":
                outputs = tf.image.resize_nearest_neighbor(
                    inputs, [self._strides[0] * height, self._strides[1] * width]
                )
                tf.logging.info(
                    "UpsampleLayer-/ResizeNN: %s shape: %s"
                    % (outputs.name, outputs.shape)
                )
                outputs = tf.pad(outputs, self._spatial_pad_1, "REFLECT")
                tf.logging.info(
                    "UpsampleLayer-/ReflectPad: %s shape: %s"
                    % (outputs.name, outputs.shape)
                )
                outputs = self._conv(outputs)
                tf.logging.info(
                    "UpsampleLayer-/NNUpConv: %s shape: %s"
                    % (outputs.name, outputs.shape)
                )
            elif self._method == "bilinear_upsample_conv":
                outputs = tf.image.resize_bilinear(
                    inputs, [self._strides[0] * height, self._strides[1] * width]
                )
                tf.logging.info(
                    "UpsampleLayer-/ResizeBilinear: %s shape: %s"
                    % (outputs.name, outputs.shape)
                )
                outputs = tf.pad(outputs, self._spatial_pad_1, "REFLECT")
                tf.logging.info(
                    "UpsampleLayer-/ReflectPad: %s shape: %s"
                    % (outputs.name, outputs.shape)
                )
                net = self._conv(outputs)
                tf.logging.info(
                    "UpsampleLayer-/BilinearUpConv: %s shape: %s"
                    % (outputs.name, outputs.shape)
                )
            elif self._method == "subpixel_conv":
                if (self._strides[0] != self._strides[1]) and (self._stride[0] == 1):
                    raise ValueError(
                        "`strides` for subpixel_conv should be greater than 1 and even"
                    )
                outputs = tf.pad(inputs, self._spatial_pad_1, "REFLECT")
                tf.logging.info(
                    "UpsampleLayer-/ReflectPad: %s shape: %s"
                    % (outputs.name, outputs.shape)
                )
                outputs = self._conv(outputs)
                tf.logging.info(
                    "UpsampleLayer-/SubPixUpConv: %s shape: %s"
                    % (outputs.name, outputs.shape)
                )
                outputs = tf.nn.depth_to_space(outputs, block_size=self._strides[0])
                tf.logging.info(
                    "UpsampleLayer-/PixelShuffle: %s shape: %s"
                    % (outputs.name, outputs.shape)
                )
            elif self._method == "conv2d_transpose":
                # This corrects 1 pixel offset for images with even width and height.
                # conv2d is left aligned and conv2d_transpose is right aligned for even
                # sized images (while doing "SAME" padding).
                # Note: This doesn"t reflect actual model in paper.
                outputs = self._conv(inputs)
                tf.logging.info(
                    "UpsampleLayer-/Conv2DTranspose: %s shape: %s"
                    % (outputs.name, outputs.shape)
                )
                # if self._data_format == "channels_first":
                #     outputs = outputs[:, :, 1:, 1:]
                # else:
                #     outputs = outputs[:, 1:, 1:, :]
                # tf.logging.info(
                #     "UpsampleLayer-/AfterConv2DT_correction: %s shape: %s" % (outputs.name, outputs.shape)
                # )
            else:
                raise ValueError("Unknown method: [%s]" % self._method)

            if self._apply_attention:
                outputs = self._selfattention(outputs)
                tf.logging.info(
                    "UpsampleLayer-/SelfAttention: %s shape: %s"
                    % (outputs.name, outputs.shape)
                )

            if self._apply_activ:
                outputs = self._activ(outputs)

            return outputs


class SubpixelConvBlock(tf.keras.layers.Layer):
    """
        A tf.keras.layer for upsampling using subpixel-convolution
        using ICNR kernel initialization and optional blur smoothing
    """

    def __init__(
        self,
        input_channels: int = None,
        output_channels: int = None,
        scale: int = 2,
        use_blur: bool = False,
        leaky_alpha: float = None,
        data_format: str = "channels_last",
        name: str = "subpixelconvblock",
        use_sigconv: bool = False,
        **kwargs
    ):
        super(SubpixelConvBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = (
            self.input_channels if output_channels is None else output_channels
        )
        self._scale = scale
        self._use_blur = use_blur
        self._leaky_alpha = leaky_alpha
        self._data_format = data_format
        if self._data_format == "channels_first":
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]
        self._block_name = name
        self._use_sigconv = use_sigconv

        # Build the blocks according to the arguments
        self._build()

    def _pixelshufflr(self, scale):
        return lambda x: tf.nn.depth_to_space(x, scale)

    def _build(self):
        """
            Builds the layer according to the arguments
        """
        self._shuf = tf.keras.layers.Lambda(self._pixelshufflr(self._scale))
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self._pad = SymmetricPad2D(paddings=(1, 0, 1, 0))
        self._blur = tf.keras.layers.AveragePooling2D(
            pool_size=(2, 2), strides=(1, 1), padding="valid"
        )
        self._relu = (
            tf.keras.layers.ReLU
            if self._leaky_alpha is None
            else tf.keras.layers.LeakyReLU(alpha=self._leaky_alpha)
        )

    def call(self, inputs):
        with tf.variable_scope(self._block_name):
            input_shape = shape_list(inputs)
            i_size = [
                input_shape[self._spatial_dims[0]],
                input_shape[self._spatial_dims[1]],
            ]
            req_out_shape = [self._scale * i_size[0], self._scale * i_size[1]]
            tf.logging.info(
                "===<<< SubpixelConv -> req_out_shape: %s >>>===" % (req_out_shape)
            )

            self.input_channels = input_shape[self._channel_axis]
            self.output_channels = (
                self.input_channels
                if self.output_channels is None
                else self.output_channels
            )
            self.output_channels = self.output_channels * (self._scale ** 2)

            if self._use_sigconv:
                outputs = tfc.SignalConv2D(
                    filters=self.output_channels,
                    kernel_support=(1, 1),
                    kernel_initializer=ICNR(
                        subtransforms.custom_conv_kernel_initializer
                    ),
                    strides_up=1,
                    corr=False,
                    padding="same_zeros",
                    data_format=self._data_format,
                    use_bias=False,
                    activation=None,
                )(inputs)
            else:
                outputs = tf.keras.layers.Conv2D(
                    filters=self.output_channels,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    kernel_initializer=ICNR(
                        subtransforms.custom_conv_kernel_initializer
                    ),
                )(inputs)

            outputs = self._shuf(self._relu(outputs))

            outputs = self._blur(self._pad(outputs)) if self._use_blur else outputs

            outputs_shape = shape_list(outputs)
            output_tensorshape = [
                outputs_shape[self._spatial_dims[0]],
                outputs_shape[self._spatial_dims[1]],
            ]
            tf.logging.info(
                "===<<< SubpixelConv -> output_tensorshape: %s >>>==="
                % (output_tensorshape)
            )

            # if output_tensorshape != req_out_shape:
            #     outputs = tf2.image.resize_with_pad(
            #         image=outputs,
            #         target_height=req_out_shape[0],
            #         target_width=req_out_shape[1],
            #         method="nearest",
            #         antialias=False
            #     )

            return outputs


class ConvolutionLayer(tf.keras.layers.Layer):
    """
        Create a sequence of convolutional, ReLU and batchnorm (if `use_batchnorm`) layers.
    """

    def __init__(
        self,
        filters: int,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding: str = "same",
        pad_scheme=(0, 0, 0, 0),
        data_format="channels_last",
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=subtransforms.custom_conv_kernel_initializer,
        bias_initializer: str = "zeros",
        name: str = "conv_block",
        is_1d: bool = False,
        use_relu: bool = False,
        norm_type="batch",
        batch_norm_momentum: float = 0.99,
        batch_norm_epsilon: float = 1e-3,
        use_ghostbatchnorm: bool = False,
        leaky: float = None,
        transpose: bool = False,
        self_attention: bool = False,
        extra_bn: bool = False,
        **kwargs
    ):
        """
        Initializes a Convolutional Block

        :param filters: (int) Number of output filters from the block
        :param kernel_size: (list/tuple) convolution kernel size {list/tuple of 2 numbers}
        :param strides: (list/tuple) stride size for the convolutional kernel
        :param padding: (str) Available padding modes: "same", "valid", "reflect", "symmetric", "constant"
        :param pad_scheme: (tuple) Padding scheme for tensor =>(pad_top, pad_bottom, pad_left, pad_right)
        :param data_format: (str) "channels_last" for `NHWC` and "channels_first" for `NCHW`
        :param dilation_rate: (list/tuple) dilation size/stride/rate for atrous/dilated convolution
        :param activation: (func) Activation function to use. If not specified, no activation is applied (ie. a(x) = x).
        :param use_bias: (bool) whether the layer uses a bias vector.
        :param kernel_initializer: (func) Initializer for the kernel weights matrix.
        :param bias_initializer: (func) Initializer for the bias vector.
        :param name: (str) scope name of the block.
        :param is_1d: (bool) if True uses Conv1D else Conv2D
        :param use_relu: (bool) if True uses the activation layer {ReLU or LeakyReLU} after convolution
        :param norm_type: (str / None) "batch" uses BatchNorm, "weight" uses WeightNorm, "spectral" uses SpectralNorm.
        :param batch_norm_momentum: (float) momentum value for batch normalization.
        :param batch_norm_epsilon: (float) eps value for batch normalization.
        :param use_ghostbatchnorm: (bool) perform "Ghost Batch Normalization",
                                          which creates virtual sub-batches
                                          which are each normalized separately
                                          (with shared gamma, beta, and moving statistics)
        :param leaky: (float) LeakyRELU alpha value (negative slope)
        :param transpose: (bool) if True uses transpose conv
        :param self_attention: (bool) if True uses self-attention layer / attention-augmented conv2d
        :param extra_bn: (bool) if True uses an extra BatchNorm layer at the end
        :param kwargs:
        """
        super(ConvolutionLayer, self).__init__(**kwargs)
        self._filters = filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        if self._padding == "same":
            self._conv_padding = "same"
        else:
            self._conv_padding = "valid"
        self._padding_scheme = pad_scheme
        self._data_format = data_format
        if self._data_format == "channels_first":
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]
        self._dilation_rate = dilation_rate
        self._activation = activation
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._block_name = name
        self._is_1d = is_1d
        self._use_relu = use_relu
        self._norm_type = norm_type
        self._batch_norm_momentum = batch_norm_momentum
        self._batch_norm_eps = batch_norm_epsilon
        self._use_ghostbn = use_ghostbatchnorm
        self._leakyalpha = leaky
        self._transpose = transpose
        self._apply_attention = self_attention
        self._extra_bn = extra_bn

        if (self._norm_type == "batch") or self._extra_bn:
            self._apply_bn = True
            self._virtual_batch_size = 2 if self._use_ghostbn else None
        else:
            self._apply_bn = False

        self._apply_bias = self._use_bias
        if self._apply_bn:
            self._apply_bias = not self._apply_bn

        if self._use_relu:
            if self._leakyalpha is None:
                self._activation = tf.keras.layers.ReLU
            else:
                self._activation = tf.keras.layers.LeakyReLU(alpha=self._leakyalpha)

        # Build the blocks according to the arguments
        self._build()

    def _build(self):
        """
            Builds the block according to the arguments
        """
        if self._padding == "reflect":
            self._padlayer = ReflectionPad2D(paddings=self._padding_scheme)
        elif self._padding == "symmetric":
            self._padlayer = SymmetricPad2D(paddings=self._padding_scheme)
        elif self._padding == "constant":
            self._padlayer = ConstantPad2D(paddings=self._padding_scheme)
        self._conv_func = (
            tf.keras.layers.Conv2DTranspose
            if self._transpose
            else tf.keras.layers.Conv1D
            if self._is_1d
            else tf.keras.layers.Conv2D
        )

        self._conv = self._conv_func(
            filters=self._filters,
            kernel_size=self._kernel_size,
            strides=self._strides,
            padding=self._conv_padding,
            data_format=self._data_format,
            dilation_rate=self._dilation_rate,
            activation=self._activation,
            use_bias=self._apply_bias,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
        )
        if self._norm_type == "weight":
            self._conv = normalizers.WeightNormalization(self._conv, data_init=False)
        elif self._norm_type == "spectral":
            self._conv = normalizers.SpectralNormalization(self._conv)
        else:
            self._conv = self._conv

        if self._apply_bn:
            self._batchnorm = tf.keras.layers.BatchNormalization(
                axis=self._channel_axis,
                momentum=self._batch_norm_momentum,
                epsilon=self._batch_norm_eps,
                virtual_batch_size=self._virtual_batch_size,
            )

        if self._apply_attention:
            self._selfattention = SelfAttention(
                name="conv_self_attn", init=self._kernel_initializer
            )

    def call(self, inputs, training=True):
        """Implementation of call().

            Args:
              inputs: the inputs tensor.
              training: boolean, whether the model is constructed for training.

            Returns:
              A output tensor.
        """
        with tf.variable_scope(self._block_name):
            tf.logging.info(
                "ConvolutionalLayer-/Input: %s shape: %s" % (inputs.name, inputs.shape)
            )
            if self._padding in ("reflect", "symmetric", "constant"):
                inputs = self._padlayer(inputs)
                tf.logging.info(
                    "ConvolutionalLayer-/PadOp: %s shape: %s"
                    % (inputs.name, inputs.shape)
                )

            outputs = self._conv(inputs)
            tf.logging.info(
                "ConvolutionalLayer-/Conv: %s shape: %s" % (outputs.name, outputs.shape)
            )
            if self._apply_bn:
                outputs = self._batchnorm(outputs, training=training)
                tf.logging.info(
                    "ConvolutionalLayer-/BatchNorm: %s shape: %s"
                    % (outputs.name, outputs.shape)
                )
            if self._apply_attention:
                outputs = self._selfattention(outputs)
                tf.logging.info(
                    "ConvolutionalLayer-/SelfAttention: %s shape: %s"
                    % (outputs.name, outputs.shape)
                )

            return outputs


class SignalConvolutionLayer(tf.keras.layers.Layer):
    """
        Create a sequence of sigconv, ReLU and batchnorm (if `use_batchnorm`) layers.
    """

    def __init__(
        self,
        filters: int,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding: str = "same",
        pad_scheme=(0, 0, 0, 0),
        use_explicit_padding=False,
        data_format="channels_last",
        activation=None,
        use_bias=True,
        kernel_initializer=subtransforms.custom_conv_kernel_initializer,
        bias_initializer: str = "zeros",
        name: str = "sigconv_block",
        is_1d: bool = False,
        use_relu: bool = False,
        use_activ: bool = False,
        norm_type="weight",
        batch_norm_momentum: float = 0.99,
        batch_norm_epsilon: float = 1e-3,
        use_ghostbatchnorm: bool = False,
        leaky: float = None,
        transpose: bool = False,
        self_attention: bool = False,
        extra_bn: bool = False,
        **kwargs
    ):
        """
        Initializes a Convolutional Block

        :param filters: (int) Number of output filters from the block
        :param kernel_size: (list/tuple) convolution kernel size {list/tuple of 2 numbers}
        :param strides: (list/tuple) stride size for the convolutional kernel
        :param padding: (str) Available padding modes: "same", "valid", "reflect", "symmetric", "constant"
        :param pad_scheme: (tuple) Padding scheme for tensor =>(pad_top, pad_bottom, pad_left, pad_right)
        :param use_explicit_padding: (bool) whether to use `EXPLICIT` padding mode (supported in TensorFlow >1.14).
        :param data_format: (str) "channels_last" for `NHWC` and "channels_first" for `NCHW`
        :param activation: (func) Activation function to use. If not specified, no activation is applied (ie. a(x) = x).
        :param use_bias: (bool) whether the layer uses a bias vector.
        :param kernel_initializer: (func) Initializer for the kernel weights matrix.
        :param bias_initializer: (func) Initializer for the bias vector.
        :param name: (str) scope name of the block.
        :param is_1d: (bool) if True uses Conv1D else Conv2D
        :param use_relu: (bool) if True uses the activation layer {ReLU or LeakyReLU} after convolution
        :param use_activ: (bool) if True uses the activation layer {ReLU or LeakyReLU} after convolution
        :param norm_type: (str) "batch" uses BatchNorm, "weight" uses WeightNorm, "spectral" uses SpectralNorm.
        :param batch_norm_momentum: (float) momentum value for batch normalization.
        :param batch_norm_epsilon: (float) eps value for batch normalization.
        :param use_ghostbatchnorm: (bool) perform "Ghost Batch Normalization",
                                          which creates virtual sub-batches
                                          which are each normalized separately
                                          (with shared gamma, beta, and moving statistics)
        :param leaky: (float) LeakyRELU alpha value (negative slope)
        :param transpose: (bool) if True uses transpose conv
        :param self_attention: (bool) if True uses self-attention layer / attention-augmented conv2d
        :param extra_bn: (bool) if True uses an extra BatchNorm layer at the end
        :param kwargs:
        """
        super(SignalConvolutionLayer, self).__init__(**kwargs)
        self._filters = filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        if self._padding == "same":
            self._conv_padding = "same_reflect"
        else:
            self._conv_padding = "valid"
        self._padding_scheme = pad_scheme
        self._use_explicit_padding = use_explicit_padding
        self._data_format = data_format
        if self._data_format == "channels_first":
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]
        self._activation = activation
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._block_name = name
        self._is_1d = is_1d
        self._use_relu = use_relu
        self._norm_type = norm_type
        self._batch_norm_momentum = batch_norm_momentum
        self._batch_norm_eps = batch_norm_epsilon
        self._use_ghostbn = use_ghostbatchnorm
        self._leakyalpha = leaky
        self._transpose = transpose
        self._apply_attention = self_attention
        self._extra_bn = extra_bn

        if (self._norm_type == "batch") or self._extra_bn:
            self._apply_bn = True
            self._virtual_batch_size = 2 if self._use_ghostbn else None
        else:
            self._apply_bn = False

        self._apply_bias = self._use_bias
        if self._apply_bn:
            self._apply_bias = not self._apply_bn

        if self._use_relu:
            if self._leakyalpha is None:
                self._activation = tf.keras.layers.ReLU
            else:
                self._activation = tf.keras.layers.LeakyReLU(alpha=self._leakyalpha)

        # Build the blocks according to the arguments
        self._build()

    def _build(self):
        """
            Builds the block according to the arguments
        """
        if self._padding == "reflect":
            self._padlayer = ReflectionPad2D(paddings=self._padding_scheme)
        elif self._padding == "symmetric":
            self._padlayer = SymmetricPad2D(paddings=self._padding_scheme)
        elif self._padding == "constant":
            self._padlayer = ConstantPad2D(paddings=self._padding_scheme)
        self._conv_func = tfc.SignalConv1D if self._is_1d else tfc.SignalConv2D

        if self._transpose:
            self._conv = self._conv_func(
                filters=self._filters,
                kernel_support=self._kernel_size,
                corr=False,
                strides_up=self._strides,
                padding=self._conv_padding,
                data_format=self._data_format,
                activation=self._activation,
                use_bias=self._apply_bias,
                use_explicit=self._use_explicit_padding,
                kernel_initializer=self._kernel_initializer,
                bias_initializer=self._bias_initializer,
                kernel_parameterizer=None,
                bias_parameterizer=None,
            )
        else:
            self._conv = self._conv_func(
                filters=self._filters,
                kernel_support=self._kernel_size,
                corr=True,
                strides_down=self._strides,
                padding=self._conv_padding,
                data_format=self._data_format,
                activation=self._activation,
                use_bias=self._apply_bias,
                use_explicit=self._use_explicit_padding,
                kernel_initializer=self._kernel_initializer,
                bias_initializer=self._bias_initializer,
                kernel_parameterizer=None,
                bias_parameterizer=None,
            )

        if self._norm_type == "weight":
            self._conv = normalizers.WeightNormalization(self._conv, data_init=False)
        elif self._norm_type == "spectral":
            self._conv = normalizers.SpectralNormalization(self._conv)
        else:
            self._conv = self._conv

        if self._apply_bn:
            self._batchnorm = tf.keras.layers.BatchNormalization(
                axis=self._channel_axis,
                momentum=self._batch_norm_momentum,
                epsilon=self._batch_norm_eps,
                virtual_batch_size=self._virtual_batch_size,
            )

        if self._apply_attention:
            self._selfattention = SelfAttention(
                name="conv_self_attn", init=self._kernel_initializer
            )

    def call(self, inputs, training=True):
        """Implementation of call().

            Args:
              inputs: the inputs tensor.
              training: boolean, whether the model is constructed for training.

            Returns:
              A output tensor.
        """
        with tf.variable_scope(self._block_name):
            if self._padding in ("reflect", "symmetric", "constant"):
                inputs = self._padlayer(inputs)

            outputs = self._conv(inputs)
            if self._apply_bn:
                outputs = self._batchnorm(outputs, training=training)
            if self._apply_attention:
                outputs = self._selfattention(outputs)

            return outputs


class Mish(Activation):
    """
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    """

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = "Mish"


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))


get_custom_objects().update({"Mish": Mish(mish)})

# TensorFlow Better Bicubic Downsample


def bicubic_kernel(x, a=-0.5):
    """https://clouard.users.greyc.fr/Pantheon/experiments/rescaling/index-en.html#bicubic"""
    if abs(x) <= 1:
        return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
    elif 1 < abs(x) < 2:
        return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
    else:
        return 0


def build_filter(factor, num_channels):
    size = factor * 4
    k = np.zeros((size))
    for i in range(size):
        x = (1 / factor) * (i - np.floor(size / 2) + 0.5)
        k[i] = bicubic_kernel(x)
    k = k / np.sum(k)
    # make 2d
    k = np.outer(k, k.T)
    return tf.constant(k, dtype=tf.float32) * tf.eye(num_channels, dtype=tf.float32)


def apply_bicubic_downsample(x, factor, data_format="channels_first"):
    """
        Downsample x by a factor of factor, using the filter built by build_filter()
        x: a rank 4 tensor
        filter: from build_filter(factor)
        factor: downsampling factor (ex: factor=2 means the output size is (h/2, w/2))
    """
    # using padding calculations from https://www.tensorflow.org/api_guides/python/nn#Convolution
    filter_height = factor * 4
    filter_width = factor * 4
    strides = factor
    pad_along_height = max(filter_height - strides, 0)
    pad_along_width = max(filter_width - strides, 0)
    # compute actual padding values for each side
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    channel_axis = 1 if data_format == "channels_first" else -1
    input_format = "NCHW" if data_format == "channels_first" else "NHWC"
    # apply mirror padding
    if channel_axis == 1:
        x = tf.pad(
            x,
            [[0, 0], [0, 0], [pad_top, pad_bottom], [pad_left, pad_right]],
            mode="REFLECT",
        )
    else:
        x = tf.pad(
            x,
            [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
            mode="REFLECT",
        )

    filter = build_filter(factor=factor, num_channels=tf.shape(input=x)[channel_axis])

    # downsampling performed by strided conv
    x = tf.nn.depthwise_conv2d(
        x,
        filter=filter,
        strides=[1, strides, strides, 1],
        padding="VALID",
        data_format=input_format,
    )
    return x
