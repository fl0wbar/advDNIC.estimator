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

from . import subtransforms


def round_filters(filters, global_params):
    """Round number of filters based on depth multiplier."""
    orig_f = filters
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    tf.logging.info("round_filter input={} output={}".format(orig_f, new_filters))
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class AnalysisStructure(tf.keras.layers.Layer):
    """
        A class that implements tf.keras.layer.Layer for MNAS-like Analysis Transform
        Reference: https://arxiv.org/abs/1807.11626
    """

    def __init__(self, blocks_args=None, global_params=None):
        """Initializes an `tf.keras.Layer` instance.

        Args:
          blocks_args: A list of BlockArgs to construct block modules.
          global_params: GlobalParams, a set of global parameters.

        Raises:
          ValueError: when blocks_args is not specified as a list.
        """
        super(AnalysisStructure, self).__init__()
        if not isinstance(blocks_args, list):
            raise ValueError("blocks_args should be a list.")
        self._global_params = global_params
        self._blocks_args = blocks_args
        self._relu_fn = global_params.relu_fn or tf.nn.swish

        self.endpoints = None

        self._build()

    def _get_conv_block(self, conv_type):
        conv_block_map = {
            0: subtransforms.MBSignalConvBlock,
            1: subtransforms.MBSignalConvBlockWithoutDepthwise,
            2: subtransforms.MBConvBlock,
            3: subtransforms.MBConvBlockWithoutDepthwise,
        }
        return conv_block_map[conv_type]

    def _build(self):
        """ Builds the layer """
        self._blocks = []
        # Builds blocks.
        for block_args in self._blocks_args:
            assert block_args.num_repeat > 0
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(
                    block_args.input_filters, self._global_params
                ),
                output_filters=round_filters(
                    block_args.output_filters, self._global_params
                ),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params),
            )

            # The first block needs to take care of stride and filter size increase.
            conv_block = self._get_conv_block(block_args.conv_type)
            self._blocks.append(conv_block(block_args, self._global_params))
            if block_args.num_repeat > 1:
                # pylint: disable=protected-access
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, strides=[1, 1]
                )
                # pylint: enable=protected-access
            for _ in xrange(block_args.num_repeat - 1):
                self._blocks.append(conv_block(block_args, self._global_params))

        batch_norm_momentum = self._global_params.batch_norm_momentum
        batch_norm_epsilon = self._global_params.batch_norm_epsilon
        if self._global_params.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1

        # Stem part.
        if self._global_params.stem_conv_type == "sigconv":
            self._conv_stem = tfc.SignalConv2D(
                filters=round_filters(32, self._global_params),
                kernel_support=(3, 3),
                kernel_initializer=subtransforms.custom_conv_kernel_initializer,
                strides_down=2,
                corr=True,
                padding="same_zeros",
                data_format=self._global_params.data_format,
                use_bias=False,
                # activation=tfc.GDN(data_format="channels_last"),
            )
        else:
            self._conv_stem = tf.layers.Conv2D(
                filters=round_filters(32, self._global_params),
                kernel_size=[3, 3],
                strides=[2, 2],
                kernel_initializer=subtransforms.custom_conv_kernel_initializer,
                padding="same",
                data_format=self._global_params.data_format,
                use_bias=False,
            )
        self._bn0 = subtransforms.batchnorm(
            axis=channel_axis, momentum=batch_norm_momentum, epsilon=batch_norm_epsilon
        )

    def call(self, inputs, training=True, features_only=None):
        """Implementation of call().

        Args:
          inputs: input tensors.
          training: boolean, whether the layer is constructed for training.
          features_only: build the base feature network only.

        Returns:
          output tensors.
        """
        with tf.variable_scope("analysis_transform"):
            outputs = None
            self.endpoints = {}
            # Calls Stem layers
            with tf.variable_scope("stem"):
                if self._global_params.stem_conv_type == "sigconv":
                    outputs = self._conv_stem(inputs)
                else:
                    if self._global_params.use_batchnorm:
                        outputs = self._relu_fn(
                            self._bn0(self._conv_stem(inputs), training=training)
                        )
                    else:
                        outputs = self._relu_fn(self._conv_stem(inputs))

            tf.logging.info("Built stem layers with output shape: %s" % outputs.shape)
            self.endpoints["stem"] = outputs

            # Calls blocks.
            reduction_idx = 0
            for idx, block in enumerate(self._blocks):
                is_reduction = False
                if (idx == len(self._blocks) - 1) or self._blocks[
                    idx + 1
                ].block_args().strides[0] > 1:
                    is_reduction = True
                    reduction_idx += 1

                with tf.variable_scope("blocks_%s" % idx):
                    drop_rate = self._global_params.drop_connect_rate
                    if drop_rate:
                        drop_rate *= float(idx) / len(self._blocks)
                        tf.logging.info(
                            "block_%s drop_connect_rate: %s" % (idx, drop_rate)
                        )
                    outputs = block.call(
                        outputs, training=training, drop_connect_rate=drop_rate
                    )
                    self.endpoints["block_%s" % idx] = outputs
                    if is_reduction:
                        self.endpoints["reduction_%s" % reduction_idx] = outputs
                    if block.endpoints:
                        for k, v in six.iteritems(block.endpoints):
                            self.endpoints["block_%s/%s" % (idx, k)] = v
                            if is_reduction:
                                self.endpoints[
                                    "reduction_%s/%s" % (reduction_idx, k)
                                ] = v
            self.endpoints["features"] = outputs

        return outputs
