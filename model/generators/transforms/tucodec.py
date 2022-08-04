from __future__ import absolute_import, division, print_function

import functools

import tensorflow.compat.v1 as tf
import tensorflow.keras.backend as K
import tensorflow_compression as tfc
from dnc.nlam import NonLocalBlock, NLAM
from dnc import convolution
from dnc.niclib import layers as op_layers


from dnc.autoregressive.pixelSNAILadv import _base_noup_smallkey_spec, _base_spec


# from dnc.autoregressive.pixelSNAIL import pxpp_spec, _base_noup_smallkey_spec
from dnc.vis import _activation_summary

"""
Contains the base architecture from the paper
David Minnen, Johannes Ballé, George Toderici:
"Joint Autoregressive and Hierarchical Priors for Learned Image Compression"

https://arxiv.org/abs/1809.02736v1
"""


class AnalysisTransform(tf.keras.layers.Layer):
    """The analysis transform."""

    def __init__(
        self,
        num_filters,
        name: str = "Analysis",
        data_format="channels_last",
        *args,
        **kwargs
    ):
        self.num_filters = num_filters
        self._block_name = name
        super(AnalysisTransform, self).__init__(name=self._block_name, *args, **kwargs)
        self._data_format = data_format
        if self._data_format == "channels_first":
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]

    def build(self, input_shape):
        self.down1 = tfc.SignalConv2D(
            int(self.num_filters / 2),
            (3, 3),
            name="layer_0",
            corr=True,
            strides_down=2,
            padding="same_zeros",
            use_bias=True,
            activation=tfc.GDN(name="gdn_0"),
        )
        self.skip1 = tfc.SignalConv2D(
            int(self.num_filters / 2),
            (3, 3),
            name="skip_0",
            corr=True,
            strides_down=8,
            padding="same_zeros",
            use_bias=True,
            activation=None,
        )
        self.down2 = tfc.SignalConv2D(
            int(self.num_filters / 2),
            (3, 3),
            name="layer_1",
            corr=True,
            strides_down=2,
            padding="same_zeros",
            use_bias=True,
            activation=tfc.GDN(name="gdn_1"),
        )
        self.skip2 = tfc.SignalConv2D(
            int(self.num_filters / 2),
            (3, 3),
            name="skip_1",
            corr=True,
            strides_down=4,
            padding="same_zeros",
            use_bias=True,
            activation=None,
        )
        self.nlam1 = NLAM(
            num_filters=int(self.num_filters / 2), kernel_size=(3, 3), name="NLAM1"
        )
        self.down3 = tfc.SignalConv2D(
            int(self.num_filters * (3 / 4)),
            (3, 3),
            name="layer_2",
            corr=True,
            strides_down=2,
            padding="same_zeros",
            use_bias=True,
            activation=tfc.GDN(name="gdn_2"),
        )
        self.skip3 = tfc.SignalConv2D(
            int(self.num_filters / 4),
            (3, 3),
            name="skip_2",
            corr=True,
            strides_down=2,
            padding="same_zeros",
            use_bias=True,
            activation=None,
        )
        self.down4 = tfc.SignalConv2D(
            self.num_filters,
            (3, 3),
            name="layer_3",
            corr=True,
            strides_down=2,
            padding="same_zeros",
            use_bias=True,
            activation=None,
        )
        self.nlam2 = NLAM(
            num_filters=self.num_filters, kernel_size=(3, 3), name="NLAM2"
        )
        self._featconcat = tf.keras.layers.Concatenate(axis=self._channel_axis)
        self.nlb = NonLocalBlock(name="nonlocal")
        self.qin = tfc.SignalConv2D(
            self.num_filters,
            (1, 1),
            name="q_in",
            corr=True,
            strides_down=1,
            padding="same_zeros",
            use_bias=True,
            activation=None,
        )

        super(AnalysisTransform, self).build(input_shape)

    def call(self, tensor):
        with tf.variable_scope(self._block_name):
            tensor = self.down1(tensor)
            _activation_summary(tensor, scope_name=self._block_name + "/Downsample1")
            skip8 = self.skip1(tensor)
            _activation_summary(tensor, scope_name=self._block_name + "/Skip1")
            tensor = self.down2(tensor)
            _activation_summary(tensor, scope_name=self._block_name + "/Downsample2")
            skip4 = self.skip2(tensor)
            _activation_summary(tensor, scope_name=self._block_name + "/Skip2")
            tensor = self.nlam1(tensor)
            _activation_summary(tensor, scope_name=self._block_name + "/NLAMsum1")
            tensor = self.down3(tensor)
            _activation_summary(tensor, scope_name=self._block_name + "/Downsample3")
            skip2 = self.skip3(tensor)
            _activation_summary(tensor, scope_name=self._block_name + "/Skip3")
            tensor = self.down4(tensor)
            _activation_summary(tensor, scope_name=self._block_name + "/Downsample4")
            tensor = self.nlam2(tensor)
            _activation_summary(tensor, scope_name=self._block_name + "/NLAMsum2")
            tensor = self._featconcat([skip8, skip4, skip2, tensor])
            tensor = self.nlb(tensor)
            _activation_summary(tensor, scope_name=self._block_name + "/NonLocal")
            tensor = self.qin(tensor)
            _activation_summary(tensor, scope_name=self._block_name + "/featureBlock")

        return tensor


class SynthesisTransform(tf.keras.layers.Layer):
    """The synthesis transform."""

    def __init__(
        self,
        num_filters,
        name: str = "Synthesis",
        data_format="channels_last",
        *args,
        **kwargs
    ):
        self.num_filters = num_filters
        self._block_name = name
        super(SynthesisTransform, self).__init__(name=self._block_name, *args, **kwargs)
        self._data_format = data_format
        if self._data_format == "channels_first":
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]

    def build(self, input_shape):
        self.qout = convolution.DualPathBlock(
            pointwise_filters_c=self.num_filters, name="DPN_block"
        )
        self.nlb = NonLocalBlock(name="nonlocal")
        self.nlam1 = NLAM(
            num_filters=self.num_filters, kernel_size=(3, 3), name="NLAM1"
        )
        self.up1 = tfc.SignalConv2D(
            self.num_filters,
            (3, 3),
            name="layer_1",
            corr=False,
            strides_up=2,
            padding="same_zeros",
            use_bias=True,
            activation=tfc.GDN(name="igdn_0", inverse=True),
        )
        self.up2 = tfc.SignalConv2D(
            int(self.num_filters * (3 / 4)),
            (3, 3),
            name="layer_2",
            corr=False,
            strides_up=2,
            padding="same_zeros",
            use_bias=True,
            activation=tfc.GDN(name="igdn_1", inverse=True),
        )
        self.nlam2 = NLAM(
            num_filters=int(self.num_filters * (3 / 4)),
            kernel_size=(3, 3),
            name="NLAM2",
        )
        self.up3 = tfc.SignalConv2D(
            int(self.num_filters / 2),
            (3, 3),
            name="layer_3",
            corr=False,
            strides_up=2,
            padding="same_zeros",
            use_bias=True,
            activation=tfc.GDN(name="igdn_2", inverse=True),
        )
        self.image_out = tfc.SignalConv2D(
            3,
            (3, 3),
            name="layer_4",
            corr=False,
            strides_up=2,
            padding="same_zeros",
            use_bias=True,
            activation=None,
        )

        super(SynthesisTransform, self).build(input_shape)

    def call(self, tensor):
        with tf.variable_scope(self._block_name):
            tensor = self.qout(tensor)
            _activation_summary(tensor, scope_name=self._block_name + "/DPN")
            tensor = self.nlb(tensor)
            _activation_summary(tensor, scope_name=self._block_name + "/NonLocal")
            tensor = self.nlam1(tensor)
            _activation_summary(tensor, scope_name=self._block_name + "/NLAMsum1")
            tensor = self.up1(tensor)
            _activation_summary(tensor, scope_name=self._block_name + "/UpConv1")
            tensor = self.up2(tensor)
            _activation_summary(tensor, scope_name=self._block_name + "/UpConv2")
            tensor = self.nlam2(tensor)
            _activation_summary(tensor, scope_name=self._block_name + "/NLAMsum2")
            tensor = self.up3(tensor)
            _activation_summary(tensor, scope_name=self._block_name + "/UpConv3")
            tensor = self.image_out(tensor)
            _activation_summary(tensor, scope_name=self._block_name + "/UpConv4")
        return tensor


class HyperAnalysisTransform(tf.keras.layers.Layer):
    """The analysis transform for the entropy model parameters."""

    def __init__(
        self,
        num_filters,
        name: str = "HyperAnalysis",
        data_format="channels_last",
        *args,
        **kwargs
    ):
        self.num_filters = num_filters
        self._block_name = name
        super(HyperAnalysisTransform, self).__init__(
            name=self._block_name, *args, **kwargs
        )
        self._data_format = data_format
        if self._data_format == "channels_first":
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]

    def build(self, input_shape):
        self._conv1 = op_layers.SignalConvolutionLayer(
            filters=int(self.num_filters / 4),
            kernel_size=(3, 3),
            strides=[1, 1],
            padding="same",
            use_bias=False,
            activation=None,
            norm_type=None,
            name="conv3x3_features",  # Conv3x3 (downsample)
            data_format=self._data_format,
        )
        self._conv1_act = tf.keras.layers.Activation("Mish", name="mish_1")
        self._residual1 = tfc.SignalConv2D(
            int(self.num_filters / 2),
            (3, 3),
            name="residual1",
            corr=True,
            strides_down=4,
            padding="same_zeros",
            use_bias=True,
            activation=None,
        )
        self.mbconv1 = convolution.MBConvBlock(
            input_filters=int(self.num_filters / 4),
            output_filters=int(self.num_filters / 2),
            kernel_size=3,
            strides=[2, 2],
            expand_ratio=1,
            se_ratio=0.25,
            id_skip=True,
            use_batchnorm=False,
            relu_fn=tf.keras.layers.Activation("Mish"),
            name="mbconv1_3x3_downsample",
            data_format=self._data_format,
        )
        self._residual2 = tfc.SignalConv2D(
            int(self.num_filters / 2),
            (3, 3),
            name="residual2",
            corr=True,
            strides_down=2,
            padding="same_zeros",
            use_bias=True,
            activation=None,
        )
        self.mbconv2 = convolution.MBConvBlock(
            input_filters=int(self.num_filters / 2),
            output_filters=int(self.num_filters / 2),
            kernel_size=3,
            strides=[1, 1],
            expand_ratio=1,
            se_ratio=0.25,
            id_skip=True,
            use_batchnorm=False,
            relu_fn=tf.keras.layers.Activation("Mish"),
            name="mbconv1_3x3_features",
            data_format=self._data_format,
        )
        self._residual3 = tfc.SignalConv2D(
            int(self.num_filters / 2),
            (3, 3),
            name="residual3",
            corr=True,
            strides_down=2,
            padding="same_zeros",
            use_bias=True,
            activation=None,
        )
        self._conv2 = op_layers.SignalConvolutionLayer(
            filters=int(self.num_filters * (3 / 4)),
            kernel_size=(3, 3),
            strides=[2, 2],
            padding="same",
            use_bias=False,
            activation=None,
            norm_type=None,
            name="conv3x3_downsample",  # Conv3x3 (downsample)
            data_format=self._data_format,
        )
        self._conv3 = op_layers.ConvolutionLayer(
            filters=self.num_filters,
            kernel_size=(3, 3),
            strides=[1, 1],
            padding="same",
            use_bias=False,
            activation=None,
            norm_type=None,
            name="conv3x3_out",
            data_format=self._data_format,
        )
        super(HyperAnalysisTransform, self).build(input_shape)

    def call(self, tensor):
        with tf.variable_scope(self._block_name):
            tensor = self._conv1_act(self._conv1(tensor))
            _activation_summary(tensor, scope_name=self._block_name + "/conv1")
            tensor_r1 = self._residual1(tensor)
            tensor = self.mbconv1(tensor)
            _activation_summary(tensor, scope_name=self._block_name + "/mbconv1")
            tensor_r2 = self._residual2(tensor)
            tensor = self.mbconv2(tensor)
            _activation_summary(tensor, scope_name=self._block_name + "/mbconv2")
            tensor_r3 = self._residual3(
                tensor[:, :, :, ::2]
                if self._data_format == "channels_last"
                else tensor[:, ::2, :, :]
            )

            tensor = self._conv2(tensor)
            _activation_summary(tensor, scope_name=self._block_name + "/conv2")

            # tf.logging.info(
            #     "===<<< HAT -> tensor_r1 : %s : %s >>>==="
            #     % (tensor_r1.name, tensor_r1.shape)
            # )
            # tf.logging.info(
            #     "===<<< HAT -> tensor_r2 : %s : %s >>>==="
            #     % (tensor_r2.name, tensor_r2.shape)
            # )
            # tf.logging.info(
            #     "===<<< HAT -> tensor_r3 : %s : %s >>>==="
            #     % (tensor_r3.name, tensor_r3.shape)
            # )
            tensor_df = tf.concat(
                [tensor, tensor_r1, tensor_r2, tensor_r3],
                axis=self._channel_axis,
                name="concat1",
            )

            tensor = self._conv3(tensor_df)
        return tensor


class HyperSynthesisTransform(tf.keras.layers.Layer):
    """The synthesis transform for the entropy model parameters."""

    def __init__(
        self,
        num_filters,
        name: str = "HyperSynthesis",
        data_format="channels_last",
        *args,
        **kwargs
    ):
        self.num_filters = num_filters
        self._block_name = name
        super(HyperSynthesisTransform, self).__init__(
            name=self._block_name, *args, **kwargs
        )
        self._data_format = data_format
        if self._data_format == "channels_first":
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]

    def build(self, input_shape):
        self._conv1 = op_layers.SignalConvolutionLayer(
            filters=self.num_filters,
            kernel_size=(3, 3),
            strides=[1, 1],
            padding="same",
            use_bias=False,
            activation=None,
            norm_type=None,
            name="conv3x3_features",
        )
        self.upsample1 = op_layers.SubpixelConvBlock(
            scale=2, leaky_alpha=0.2, name="subpixconv_upsample"  # (upsample)
        )
        self._upsample1_act = tf.keras.layers.Activation("Mish", name="mish_1")
        self.mbupsample1 = convolution.MBUpsampleConvBlock(
            input_filters=int(self.num_filters),
            output_filters=int(self.num_filters * 1.5),
            kernel_size=3,
            scale=1,
            expand_ratio=1,
            se_ratio=0.25,
            id_skip=True,
            use_blur=False,
            relu_fn=tf.keras.layers.Activation("Mish"),
            name="mbupsample1_3x3_features",
        )
        self.mbupsample2 = convolution.MBUpsampleConvBlock(
            input_filters=int(self.num_filters * 1.5),
            output_filters=int(self.num_filters * 2),
            kernel_size=3,
            scale=2,
            expand_ratio=1,
            se_ratio=0.25,
            use_subpixel=True,
            id_skip=True,
            use_blur=False,
            use_batchnorm=False,
            relu_fn=tf.keras.layers.Activation("Mish"),
            name="mbupsample2_3x3_upsample",
        )
        self.conv2 = op_layers.SignalConvolutionLayer(
            filters=int(self.num_filters * 2),
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            use_bias=True,
            activation=None,
            norm_type=None,
            name="conv3x3_out",
        )
        super(HyperSynthesisTransform, self).build(input_shape)

    def call(self, tensor):
        with tf.variable_scope(self._block_name):
            tensor = self._conv1(tensor)
            tensor = self._upsample1_act(self.upsample1(tensor))
            _activation_summary(tensor, scope_name=self._block_name + "/upsample1")
            tensor = self.mbupsample1(tensor)
            _activation_summary(tensor, scope_name=self._block_name + "/mbupsample1")
            tensor = self.mbupsample2(tensor)
            _activation_summary(tensor, scope_name=self._block_name + "/mbupsample2")
            tensor = self.conv2(tensor)
            _activation_summary(tensor, scope_name=self._block_name + "/conv2")
        return tensor


class EntropyParameters(tf.keras.layers.Layer):
    """
    The entropy parameters network for prediction of mean and scale parameters
    of a conditional Gaussian Mixture Model
    """

    def __init__(self, num_filters, name: str = "EntropyParameters", *args, **kwargs):
        self.num_filters = num_filters
        self._block_name = name
        super(EntropyParameters, self).__init__(name=self._block_name, *args, **kwargs)

    def build(self, input_shape):
        self._ep_layers = [
            tfc.SignalConv2D(
                int(self.num_filters * 4.0),
                (1, 1),
                name="layer_0",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=functools.partial(tf.nn.leaky_relu, alpha=0.1),
            ),
            tfc.SignalConv2D(
                int(self.num_filters * 2.0),
                (1, 1),
                name="layer_1",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=functools.partial(tf.nn.leaky_relu, alpha=0.1),
            ),
        ]
        self._mean_layer = tfc.SignalConv2D(
            self.num_filters,
            (1, 1),
            name="layer_2",
            corr=True,
            strides_down=1,
            padding="same_zeros",
            use_bias=False,
            activation=None,
        )
        self._scale_layer = tfc.SignalConv2D(
            self.num_filters,
            (1, 1),
            name="layer_2",
            corr=True,
            strides_down=1,
            padding="same_zeros",
            use_bias=False,
            activation=None,
        )
        super(EntropyParameters, self).build(input_shape)

    def call(self, tensor):
        with tf.variable_scope(self._block_name):
            for layer in self._ep_layers:
                ep_tensor = layer(tensor)
                _activation_summary(ep_tensor, scope_name="EntropyParametersBody")
            mu_tensor = self._mean_layer(ep_tensor)
            _activation_summary(mu_tensor, scope_name="EntropyParametersMean")
            sigma_tensor = self._scale_layer(ep_tensor)
            _activation_summary(sigma_tensor, scope_name="EntropyParametersScale")

        return mu_tensor, sigma_tensor


class ContextModel(tf.keras.layers.Layer):
    """
    This class implements an autoregressive layer
    for tracing the entropy distribution from quantized representation
    generated by the encoder
    """

    def __init__(self, num_filters, name: str = "ContextModel", *args, **kwargs):
        self.num_filters = num_filters
        self._block_name = name
        self.axis = 1 if K.image_data_format() == "channels_first" else -1
        super(ContextModel, self).__init__(name=self._block_name, *args, **kwargs)

    def build(self, input_shape):
        self._shape = input_shape
        if self.axis == 1:
            _, channels, height, width = input_shape
        else:
            _, height, width, channels = input_shape
        self._autoregressive_layer = functools.partial(
            _base_noup_smallkey_spec,
            h=None,
            init=False,
            ema=None,
            dropout_p=0.5,
            nr_resnet=1,
            nr_filters=self.num_filters,
            attn_rep=2,
            output_units=int(self.num_filters * 2.0),
            att_downsample=1,
            resnet_nonlinearity="concat_elu",
        )
        # self._autoregressive_layer = functools.partial(
        #     _base_spec,
        #     h=None,
        #     init=False,
        #     ema=None,
        #     dropout_p=0.5,
        #     nr_resnet=1,
        #     nr_filters=self.num_filters,
        #     output_units=int(self.num_filters * 2.0),
        #     resnet_nonlinearity="concat_elu",
        # )
        # self._autoregressive_layer = functools.partial(
        #     _context_attn_spec,
        #     h=None,
        #     init=False,
        #     ema=None,
        #     dropout_p=0.5,
        #     nr_resnet=2,
        #     nr_filters=self.num_filters,
        #     attn_rep=2,
        #     output_units=int(self.num_filters * 2.0),
        #     att_downsample=1,
        #     resnet_nonlinearity="concat_elu",
        # )
        # self._autoregressive_layer = functools.partial(
        #     pxpp_spec,
        #     h=None,
        #     init=False,
        #     ema=None,
        #     dropout_p=0.5,
        #     nr_resnet=1,
        #     nr_filters=self.num_filters,
        #     output_units=int(self.num_filters * 2.0),
        #     resnet_nonlinearity="concat_elu",
        # )
        super(ContextModel, self).build(input_shape)

    def call(self, tensor):
        with tf.variable_scope(self._block_name):
            tensor = self._autoregressive_layer(x=tensor)
            _activation_summary(tensor, scope_name="ContextModel")
        return tensor
