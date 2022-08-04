from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import sys

import numpy as np
import tensorflow as tf

import tensorflow_compression as tfc
from ..vis import _activation_summary
import dnc.convolution as effops

"""
    Contains the modifications to architecture from the paper
    J. Ballé, D. Minnen, S. Singh, S.J. Hwang, N. Johnston:
    "Variational Image Compression with a Scale Hyperprior"
    Int. Conf. on Learning Representations (ICLR), 2018
    https://arxiv.org/abs/1802.01436

    Modifications :
        1. Mobile-Bottleneck Residual Convolutional Layer (EfficientNet)

"""


class AnalysisTransform(tf.keras.layers.Layer):
    """The analysis transform."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(AnalysisTransform, self).__init__(*args, **kwargs)

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
            effops.MBConvBlock(
                input_filters=self.num_filters,
                output_filters=self.num_filters,
                kernel_size=3,
                strides=[1, 1],
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                name="layer_0_mbconv",
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
            effops.MBConvBlock(
                input_filters=self.num_filters,
                output_filters=self.num_filters,
                kernel_size=3,
                strides=[1, 1],
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                name="layer_1_mbconv",
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
            effops.MBConvBlock(
                input_filters=self.num_filters,
                output_filters=self.num_filters,
                kernel_size=3,
                strides=[1, 1],
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                name="layer_2_mbconv",
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
        with tf.variable_scope("analysis_transform"):
            for layer in self._layers:
                tensor = layer(tensor)
                _activation_summary(tensor, scope_name="AT_activations")
        return tensor


class SynthesisTransform(tf.keras.layers.Layer):
    """The synthesis transform."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(SynthesisTransform, self).__init__(*args, **kwargs)

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
            effops.MBConvBlock(
                input_filters=self.num_filters,
                output_filters=self.num_filters,
                kernel_size=3,
                strides=[1, 1],
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                name="layer_0_mbconv",
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
            effops.MBConvBlock(
                input_filters=self.num_filters,
                output_filters=self.num_filters,
                kernel_size=3,
                strides=[1, 1],
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                name="layer_1_mbconv",
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
            effops.MBConvBlock(
                input_filters=self.num_filters,
                output_filters=self.num_filters,
                kernel_size=3,
                strides=[1, 1],
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                name="layer_2_mbconv",
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
        with tf.variable_scope("synthesis_transform"):
            for layer in self._layers:
                tensor = layer(tensor)
                _activation_summary(tensor, scope_name="ST_activations")
        return tensor


class HyperAnalysisTransform(tf.keras.layers.Layer):
    """The analysis transform for the entropy model parameters."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(HyperAnalysisTransform, self).__init__(*args, **kwargs)

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
                activation=tf.nn.relu,
            ),
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_1",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=tf.nn.relu,
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
        with tf.variable_scope("hyper_analysis_transform"):
            for layer in self._layers:
                tensor = layer(tensor)
                _activation_summary(tensor, scope_name="HAT_activations")
        return tensor


class HyperSynthesisTransform(tf.keras.layers.Layer):
    """The synthesis transform for the entropy model parameters."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(HyperSynthesisTransform, self).__init__(*args, **kwargs)

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
                activation=tf.nn.relu,
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
                activation=tf.nn.relu,
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
        super(HyperSynthesisTransform, self).build(input_shape)

    def call(self, tensor):
        with tf.variable_scope("hyper_synthesis_transform"):
            for layer in self._layers:
                tensor = layer(tensor)
                _activation_summary(tensor, scope_name="HST_activations")
        return tensor
