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
import dnc.niclib.layers as efflayers

"""
    Contains the modified base architecture from the paper
    J. Ballé, D. Minnen, S. Singh, S.J. Hwang, N. Johnston:
    "Variational Image Compression with a Scale Hyperprior"
    Int. Conf. on Learning Representations (ICLR), 2018
    https://arxiv.org/abs/1802.01436
    
    Modifications :
        1. Using convolutions in place of tfc.SignalConv2D
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
        with tf.variable_scope("analysis_transform"):
            for layer in self._layers:
                tensor = layer(tensor)
                _activation_summary(tensor, scope_name="AT_activations")
        return tensor


# class AnalysisTransform(tf.keras.layers.Layer):
#     """The analysis transform."""

#     def __init__(self, num_filters, *args, **kwargs):
#         self.num_filters = num_filters
#         super(AnalysisTransform, self).__init__(*args, **kwargs)

#     def build(self, input_shape):
#         self._layers = [
#             efflayers.ConvolutionLayer(
#                 filters=self.num_filters,
#                 kernel_size=(5, 5),
#                 strides=(2, 2),
#                 padding="same",
#                 use_bias=True,
#                 activation=tf.keras.layers.LeakyReLU(alpha=0.2),
#                 norm_type=None,
#                 name="layer_0",
#             ),
#             efflayers.ConvolutionLayer(
#                 filters=self.num_filters,
#                 kernel_size=(5, 5),
#                 strides=(2, 2),
#                 padding="same",
#                 use_bias=True,
#                 activation=tf.keras.layers.LeakyReLU(alpha=0.2),
#                 norm_type=None,
#                 name="layer_1",
#             ),
#             efflayers.ConvolutionLayer(
#                 filters=self.num_filters,
#                 kernel_size=(5, 5),
#                 strides=(2, 2),
#                 padding="same",
#                 use_bias=True,
#                 activation=tf.keras.layers.LeakyReLU(alpha=0.2),
#                 norm_type=None,
#                 name="layer_2",
#             ),
#             efflayers.ConvolutionLayer(
#                 filters=self.num_filters,
#                 kernel_size=(5, 5),
#                 strides=(2, 2),
#                 padding="same",
#                 use_bias=True,
#                 activation=None,
#                 norm_type=None,
#                 name="layer_3",
#             ),
#         ]
#         super(AnalysisTransform, self).build(input_shape)

#     def call(self, tensor):
#         with tf.variable_scope("analysis_transform"):
#             for layer in self._layers:
#                 tensor = layer(tensor)
#                 _activation_summary(tensor, scope_name="AT_activations")
#         return tensor


# class SynthesisTransform(tf.keras.layers.Layer):
#     """The synthesis transform."""

#     def __init__(self, num_filters, *args, **kwargs):
#         self.num_filters = num_filters
#         super(SynthesisTransform, self).__init__(*args, **kwargs)

#     def build(self, input_shape):
#         self._layers = [
#             efflayers.UpsampleLayer(
#                 output_filters=self.num_filters,
#                 strides=(2, 2),
#                 apply_activ=True,
#                 activation=tf.keras.layers.LeakyReLU(alpha=0.2),
#                 method="subpixel_conv",
#                 name="layer_0"
#             ),
#             efflayers.UpsampleLayer(
#                 output_filters=self.num_filters,
#                 strides=(2, 2),
#                 apply_activ=True,
#                 activation=tf.keras.layers.LeakyReLU(alpha=0.2),
#                 method="subpixel_conv",
#                 name="layer_1"
#             ),
#             efflayers.UpsampleLayer(
#                 output_filters=self.num_filters,
#                 strides=(2, 2),
#                 apply_activ=True,
#                 activation=tf.keras.layers.LeakyReLU(alpha=0.2),
#                 method="subpixel_conv",
#                 name="layer_2"
#             ),
#             efflayers.UpsampleLayer(
#                 output_filters=3,
#                 strides=(2, 2),
#                 apply_activ=False,
#                 method="subpixel_conv",
#                 name="layer_3"
#             ),
#         ]
#         super(SynthesisTransform, self).build(input_shape)

#     def call(self, tensor):
#         with tf.variable_scope("synthesis_transform"):
#             for layer in self._layers:
#                 tensor = layer(tensor)
#                 _activation_summary(tensor, scope_name="ST_activations")
#         return tensor

class SynthesisTransform(tf.keras.layers.Layer):
    """The synthesis transform."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(SynthesisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            efflayers.UpsampleLayer(
                output_filters=self.num_filters,
                strides=(2, 2),
                apply_activ=True,
                activation=tfc.GDN(name="igdn_0", inverse=True),
                method="subpixel_conv",
                name="layer_0"
            ),
            efflayers.UpsampleLayer(
                output_filters=self.num_filters,
                strides=(2, 2),
                apply_activ=True,
                activation=tfc.GDN(name="igdn_1", inverse=True),
                method="subpixel_conv",
                name="layer_1"
            ),
            efflayers.UpsampleLayer(
                output_filters=self.num_filters,
                strides=(2, 2),
                apply_activ=True,
                activation=tfc.GDN(name="igdn_2", inverse=True),
                method="subpixel_conv",
                name="layer_2"
            ),
            efflayers.UpsampleLayer(
                output_filters=3,
                strides=(2, 2),
                apply_activ=False,
                method="subpixel_conv",
                name="layer_3"
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
