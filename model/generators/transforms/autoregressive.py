from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow.compat.v1 as tf
import tensorflow.keras.backend as K
import tensorflow_compression as tfc
from dnc.autoregressive.pixelSNAILadv import (
    pxpp_spec,
    _base_noup_smallkey_spec,
    _context_attn_spec,
    _base_spec,
)

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

    def __init__(self, num_filters, name: str = "Analysis", *args, **kwargs):
        self.num_filters = num_filters
        super(AnalysisTransform, self).__init__(name=name, *args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_0",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_0"),
            ),
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_1",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_1"),
            ),
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_2",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_2"),
            ),
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_3",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=None,
            ),
        ]
        super(AnalysisTransform, self).build(input_shape)

    def call(self, tensor):
        # with tf.variable_scope("analysis_transform"):
        for layer in self._layers:
            tensor = layer(tensor)
            _activation_summary(tensor, scope_name="analysisActiv")
        return tensor


class SynthesisTransform(tf.keras.layers.Layer):
    """The synthesis transform."""

    def __init__(self, num_filters, name: str = "Synthesis", *args, **kwargs):
        self.num_filters = num_filters
        super(SynthesisTransform, self).__init__(name=name, *args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_0",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_0", inverse=True),
            ),
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_1",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_1", inverse=True),
            ),
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_2",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_2", inverse=True),
            ),
            tfc.SignalConv2D(
                3,
                (5, 5),
                name="layer_3",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                activation=None,
            ),
        ]
        super(SynthesisTransform, self).build(input_shape)

    def call(self, tensor):
        # with tf.variable_scope("synthesis_transform"):
        for layer in self._layers:
            tensor = layer(tensor)
            _activation_summary(tensor, scope_name="synthesisActiv")
        return tensor


class HyperAnalysisTransform(tf.keras.layers.Layer):
    """The analysis transform for the entropy model parameters."""

    def __init__(self, num_filters, name: str = "HyperAnalysis", *args, **kwargs):
        self.num_filters = num_filters
        super(HyperAnalysisTransform, self).__init__(name=name, *args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters,
                (3, 3),
                name="layer_0",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=functools.partial(tf.nn.leaky_relu, alpha=0.1),
            ),
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_1",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=functools.partial(tf.nn.leaky_relu, alpha=0.1),
            ),
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_2",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=False,
                activation=None,
            ),
        ]
        super(HyperAnalysisTransform, self).build(input_shape)

    def call(self, tensor):
        # with tf.variable_scope("hyper_analysis_transform"):
        for layer in self._layers:
            tensor = layer(tensor)
            _activation_summary(tensor, scope_name="hyperanalysisActiv")
        return tensor


class HyperSynthesisTransform(tf.keras.layers.Layer):
    """The synthesis transform for the entropy model parameters."""

    def __init__(self, num_filters, name: str = "HyperSynthesis", *args, **kwargs):
        self.num_filters = num_filters
        super(HyperSynthesisTransform, self).__init__(name=name, *args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_0",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                kernel_parameterizer=None,
                activation=functools.partial(tf.nn.leaky_relu, alpha=0.1),
            ),
            tfc.SignalConv2D(
                int(self.num_filters * 1.5),
                (5, 5),
                name="layer_1",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                kernel_parameterizer=None,
                activation=functools.partial(tf.nn.leaky_relu, alpha=0.1),
            ),
            tfc.SignalConv2D(
                int(self.num_filters * 2.0),
                (3, 3),
                name="layer_2",
                corr=False,
                strides_up=1,
                padding="same_zeros",
                use_bias=True,
                kernel_parameterizer=None,
                activation=None,
            ),
        ]
        super(HyperSynthesisTransform, self).build(input_shape)

    def call(self, tensor):
        # with tf.variable_scope("hyper_synthesis_transform"):
        for layer in self._layers:
            tensor = layer(tensor)
            _activation_summary(tensor, scope_name="hypersynthesisActiv")
        return tensor


class EntropyParameters(tf.keras.layers.Layer):
    """
    The entropy parameters network for prediction of mean and scale parameters
    of a conditional Gaussian Mixture Model
    """

    def __init__(self, num_filters, name: str = "EntropyParameters", *args, **kwargs):
        self.num_filters = num_filters
        super(EntropyParameters, self).__init__(name=name, *args, **kwargs)

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
        self.axis = 1 if K.image_data_format() == "channels_first" else -1
        super(ContextModel, self).__init__(name=name, *args, **kwargs)

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
            nr_resnet=2,
            nr_filters=self.num_filters,
            attn_rep=4,
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
        #     attn_rep=4,
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
        tensor = self._autoregressive_layer(x=tensor)
        return tensor
