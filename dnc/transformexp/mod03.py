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

    def __init__(self, num_filters, bn=True, *args, **kwargs):
        self.num_filters = num_filters
        self.use_bn = bn
        super(AnalysisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        """
            Building the layers according to a modified EfficientNet architecture
        """
        self._layers = [
            efflayers.ConvolutionLayer(
                filters=32,
                kernel_size=[3, 3],
                strides=[2, 2],
                padding='same',
                use_bias=False,
                activation=tf.nn.swish,
                name="conv3x3_f32_downsample"  # Conv3x3 (downsample)
            ),
            effops.MBConvBlock(
                input_filters=32,
                output_filters=16,
                kernel_size=3,
                strides=[1, 1],
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                use_batchnorm=self.use_bn,
                name="mbconv1_fi32fo16_3x3_1",  # MBConv1(3x3) (block=1)
            ),
            effops.MBConvBlock(
                input_filters=16,
                output_filters=16,
                kernel_size=3,
                strides=[1, 1],
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                use_batchnorm=self.use_bn,
                name="mbconv1_fi16fo16_3x3_2",  # MBConv1(3x3)
            ),
            effops.MBConvBlock(
                input_filters=16,
                output_filters=24,
                kernel_size=3,
                strides=[2, 2],
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                use_batchnorm=self.use_bn,
                name="mbconv1_fi16fo24_3x3_1",  # MBConv1(3x3) (downsample) (block=2)
            ),
            effops.MBConvBlock(
                input_filters=24,
                output_filters=24,
                kernel_size=3,
                strides=[1, 1],
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                use_batchnorm=self.use_bn,
                name="mbconv1_fi24fo24_3x3_2",  # MBConv1(3x3)
            ),
            effops.MBConvBlock(
                input_filters=24,
                output_filters=40,
                kernel_size=5,
                strides=[2, 2],
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                use_batchnorm=self.use_bn,
                name="mbconv1_fi24fo40_5x5_1",  # MBConv1(5x5) (downsample) (block=3)
            ),
            effops.MBConvBlock(
                input_filters=40,
                output_filters=40,
                kernel_size=5,
                strides=[1, 1],
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                use_batchnorm=self.use_bn,
                name="mbconv1_fi40fo40_5x5_2",  # MBConv1(5x5)
            ),
            effops.MBConvBlock(
                input_filters=40,
                output_filters=80,
                kernel_size=3,
                strides=[2, 2],
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                use_batchnorm=self.use_bn,
                name="mbconv1_fi40fo80_3x3_1",  # MBConv1(3x3) (downsample) (block=4)
            ),
            effops.MBConvBlock(
                input_filters=80,
                output_filters=80,
                kernel_size=3,
                strides=[1, 1],
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                use_batchnorm=self.use_bn,
                name="mbconv1_fi80fo80_3x3_2",  # MBConv1(3x3)
            ),
            effops.MBConvBlock(
                input_filters=80,
                output_filters=112,
                kernel_size=5,
                strides=[1, 1],
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                use_batchnorm=self.use_bn,
                name="mbconv1_fi80fo112_5x5_1",  # MBConv1(5x5) (block=5)
            ),
            effops.MBConvBlock(
                input_filters=112,
                output_filters=112,
                kernel_size=5,
                strides=[1, 1],
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                use_batchnorm=self.use_bn,
                name="mbconv1_fi112fo112_5x5_2",  # MBConv1(5x5)
            ),
            effops.MBConvBlock(
                input_filters=112,
                output_filters=192,
                kernel_size=3,
                strides=[1, 1],
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                use_batchnorm=self.use_bn,
                name="mbconv1_fi112fo192_3x3_1",  # MBConv1(3x3) (block=6)
            ),
            efflayers.ConvolutionLayer(
                filters=192,
                kernel_size=(5, 5),
                strides=(1, 1),
                padding="same",
                use_bias=True,
                activation=None,
                norm_type=None,
                name="conv5x5_f192"  # Conv5x5
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

    def __init__(self, num_filters, bn=True, *args, **kwargs):
        self.num_filters = num_filters
        self.use_bn = bn
        super(SynthesisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            efflayers.ConvolutionLayer(
                filters=192,
                kernel_size=(5, 5),
                strides=(1, 1),
                padding="same",
                use_bias=True,
                activation=None,
                norm_type=None,
                name="conv5x5_f192"  # Conv5x5
            ),
            effops.MBUpsampleConvBlock(
                input_filters=192,
                output_filters=112,
                kernel_size=3,
                scale=1,
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                use_blur=False,
                use_batchnorm=self.use_bn,
                name="mbupsample1_fi192fo112_3x3_1",  # MBUpsample1(3x3) (block=1)(reverseblock=6)
            ),
            effops.MBUpsampleConvBlock(
                input_filters=112,
                output_filters=112,
                kernel_size=5,
                scale=1,
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                use_blur=False,
                use_batchnorm=self.use_bn,
                name="mbupsample1_fi112fo112_5x5_1",  # MBUpsample1(5x5)
            ),
            effops.MBUpsampleConvBlock(
                input_filters=112,
                output_filters=80,
                kernel_size=5,
                scale=1,
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                use_blur=False,
                use_batchnorm=self.use_bn,
                name="mbupsample1_fi112fo80_5x5_2",  # MBUpsample1(5x5) (block=2)(reverseblock=5)
            ),
            effops.MBUpsampleConvBlock(
                input_filters=80,
                output_filters=80,
                kernel_size=3,
                scale=1,
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                use_blur=False,
                use_batchnorm=self.use_bn,
                name="mbupsample1_fi80fo80_3x3_1",  # MBUpsample1(3x3)
            ),
            effops.MBUpsampleConvBlock(
                input_filters=80,
                output_filters=40,
                kernel_size=3,
                scale=2,
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                use_blur=False,
                use_batchnorm=self.use_bn,
                name="mbupsample1_fi80fo40_3x3_2",  # MBUpsample1(3x3) (upsample)(block=3)(reverseblock=4)
            ),
            effops.MBUpsampleConvBlock(
                input_filters=40,
                output_filters=40,
                kernel_size=5,
                scale=1,
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                use_blur=False,
                use_batchnorm=self.use_bn,
                name="mbupsample1_fi40fo40_5x5_1",  # MBUpsample1(5x5)
            ),
            effops.MBUpsampleConvBlock(
                input_filters=40,
                output_filters=24,
                kernel_size=5,
                scale=2,
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                use_blur=False,
                use_batchnorm=self.use_bn,
                name="mbupsample1_fi40fo24_5x5_2",  # MBUpsample1(5x5) (upsample)(block=4)(reverseblock=3)
            ),
            effops.MBUpsampleConvBlock(
                input_filters=24,
                output_filters=24,
                kernel_size=3,
                scale=1,
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                use_blur=False,
                use_batchnorm=self.use_bn,
                name="mbupsample1_fi24fo24_3x3_1",  # MBUpsample1(3x3)
            ),
            effops.MBUpsampleConvBlock(
                input_filters=24,
                output_filters=16,
                kernel_size=3,
                scale=2,
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                use_blur=True,
                use_batchnorm=self.use_bn,
                name="mbupsample1_fi24fo16_3x3_2",  # MBUpsample1(3x3) (upsample)(block=5)(reverseblock=2)
            ),
            effops.MBUpsampleConvBlock(
                input_filters=16,
                output_filters=16,
                kernel_size=3,
                scale=1,
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                use_blur=False,
                use_batchnorm=self.use_bn,
                name="mbupsample1_fi16fo16_3x3_1",  # MBUpsample1(3x3)
            ),
            effops.MBUpsampleConvBlock(
                input_filters=16,
                output_filters=32,
                kernel_size=3,
                scale=1,
                expand_ratio=1,
                se_ratio=0.25,
                id_skip=True,
                use_blur=False,
                use_batchnorm=self.use_bn,
                name="mbupsample1_fi16fo32_3x3_2",  # MBUpsample1(3x3) (block=6)(reverseblock=1)
            ),
            efflayers.SubpixelConvBlock(
                scale=2,
                leaky_alpha=0.2,
                name="subpixconv_upsample"  # (upsample)
            ),
            efflayers.ConvolutionLayer(
                filters=32,
                kernel_size=[3, 3],
                strides=[1, 1],
                padding='same',
                use_bias=True,
                activation=tf.nn.swish,
                name="conv3x3_f32"  # Conv3x3
            ),
            efflayers.ConvolutionLayer(
                filters=3,
                kernel_size=[3, 3],
                strides=[1, 1],
                padding='same',
                use_bias=True,
                activation=None,
                norm_type=None,
                name="conv3x3_out",
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
                192,
                (3, 3),
                name="layer_0",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=tf.nn.relu,
            ),
            tfc.SignalConv2D(
                192,
                (5, 5),
                name="layer_1",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=tf.nn.relu,
            ),
            tfc.SignalConv2D(
                320,
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
                320,
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
                192,
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
                192,
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
