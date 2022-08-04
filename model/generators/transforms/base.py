from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.stats
import tensorflow.compat.v1 as tf

from tensorflow.python.keras.engine import input_spec
from tensorflow_compression.python.ops import math_ops
from tensorflow_compression.python.ops import range_coding_ops

import tensorflow_compression as tfc
from dnc.vis import _activation_summary

"""
    Contains the base architecture from the paper
    David Minnen, Johannes Ballé, George Toderici:
    "Joint Autoregressive and Hierarchical Priors for Learned Image Compression"
    
    https://arxiv.org/abs/1809.02736v1
"""


class AnalysisTransform(tf.keras.layers.Layer):
    """The analysis transform."""

    def __init__(self, num_filters, name: str = "analysis", *args, **kwargs):
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

    def __init__(self, num_filters, name: str = "synthesis", *args, **kwargs):
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

    def __init__(self, num_filters, name: str = "hyper_analysis", *args, **kwargs):
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
                activation=tf.nn.leaky_relu,
            ),
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_1",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=tf.nn.leaky_relu,
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


class HyperSynthesisTransformScale(tf.keras.layers.Layer):
    """The synthesis transform for the entropy model parameters."""

    def __init__(self, num_filters, name: str = "hyper_synthesis_scale", *args, **kwargs):
        self.num_filters = num_filters
        super(HyperSynthesisTransformScale, self).__init__(name=name, *args, **kwargs)

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
                activation=tf.nn.leaky_relu,
            ),
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_1",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                kernel_parameterizer=None,
                activation=tf.nn.leaky_relu,
            ),
            tfc.SignalConv2D(
                self.num_filters,
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
        super(HyperSynthesisTransformScale, self).build(input_shape)

    def call(self, tensor):
        # with tf.variable_scope("hyper_synthesis_transform"):
        for layer in self._layers:
            tensor = layer(tensor)
            _activation_summary(tensor, scope_name="hypersynscaleActiv")
        return tensor


class HyperSynthesisTransformMean(tf.keras.layers.Layer):
    """The synthesis transform for the entropy model parameters."""

    def __init__(self, num_filters, name: str = "hyper_synthesis_mean", *args, **kwargs):
        self.num_filters = num_filters
        super(HyperSynthesisTransformMean, self).__init__(name=name, *args, **kwargs)

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
                activation=tf.nn.leaky_relu,
            ),
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_1",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                kernel_parameterizer=None,
                activation=tf.nn.leaky_relu,
            ),
            tfc.SignalConv2D(
                self.num_filters,
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
        super(HyperSynthesisTransformMean, self).build(input_shape)

    def call(self, tensor):
        # with tf.variable_scope("hyper_synthesis_transform"):
        for layer in self._layers:
            tensor = layer(tensor)
            _activation_summary(tensor, scope_name="hypersynmeanActiv")
        return tensor
