# -*- coding: utf-8 -*-
""" Entropy Model Layers """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.stats
import tensorflow.compat.v1 as tf

from tensorflow.python.keras.engine import input_spec
from tensorflow_compression.python.ops import math_ops
from tensorflow_compression.python.ops import range_coding_ops


__all__ = ["GSMMConditional"]


class GSMMConditional(tf.keras.layers.Layer):
    """
    Conditional Gaussian entropy model. (here a variant of Gaussian Scale Mixture Model)

    The layer implements a conditionally Gaussian probability density model to
    estimate entropy of its input tensor, which is described in the paper:

    > "Variational image compression with a scale hyperprior"<br />
    > J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston<br />
    > https://arxiv.org/abs/1802.01436

    """

    _setattr_tracking = False

    def __init__(
        self,
        scale,
        scale_table,
        scale_bound=None,
        mean=None,
        indexes=None,
        tail_mass=2 ** -8,
        likelihood_bound=1e-9,
        range_coder_precision=16,
        **kwargs
    ):
        """
        Initializer

        Arguments:
            scale: `Tensor`, the scale parameters for the conditional distributions.
            scale_table: Iterable of positive floats. For range coding, the scale
                parameters in `scale` can't be used, because the probability tables need
                to be constructed statically. Only the values given in this table will
                actually be used for range coding. For each predicted scale, the next
                greater entry in the table is selected. It's optimal to choose the
                scales provided here in a logarithmic way.
            scale_bound: Float. Lower bound for scales. Any values in `scale` smaller
                than this value are set to this value to prevent non-positive scales. By
                default (or when set to `None`), uses the smallest value in
                `scale_table`. To disable, set to 0.
            mean: `Tensor`, the mean parameters for the conditional distributions. If
                            `None`, the mean is assumed to be zero.
            indexes: `Tensor` of type `int32` or `None`. Can be used to override the
                selection of scale table indexes based on the predicted values in
                `scale`. Only affects compression and decompression.
            tail_mass: Float, between 0 and 1. The bottleneck layer automatically
                determines the range of input values based on their frequency of
                occurrence. Values occurring in the tails of the distributions will not
                be encoded with range coding, but using a Golomb-like code. `tail_mass`
                determines the amount of probability mass in the tails which will be
                Golomb-coded. For example, the default value of `2 ** -8` means that on
                average, one 256th of all values will use the Golomb code.
            likelihood_bound: Float. If positive, the returned likelihood values are
                ensured to be greater than or equal to this value. This prevents very
                large gradients with a typical entropy loss (defaults to 1e-9).
            range_coder_precision: Integer, between 1 and 16. The precision of the
                range coder used for compression and decompression. This trades off
                computation speed with compression efficiency, where 16 is the slowest
                but most efficient setting. Choosing lower values may increase the
                average codelength slightly compared to the estimated entropies.
            **kwargs: Other keyword arguments passed to superclass (`Layer`).
        """
        super(GSMMConditional, self).__init(**kwargs)
        self._tail_mass = float(tail_mass)
        if not 0 < self._tail_mass < 1:
            raise ValueError(
                "`tail_mass` must be between 0 and 1, got {}".format(self._tail_mass)
            )
        self._likelihood_bound = float(likelihood_bound)
        self._range_coder_precision = int(range_coder_precision)

        self._scale = tf.convert_to_tensor(scale)
        input_shape = self.scale.shape
        self._scale_table = tuple(sorted(float(s) for s in scale_table))
        if any(s <= 0 for s in self._scale_table):
            raise ValueError("`scale_table` must be an iterable of positive numbers.")
        self._scale_bound = None if scale_bound is None else float(scale_bound)
        self._mean = None if mean is None else tf.convert_to_tensor(mean)
        if indexes is not None:
            self._indexes = tf.convert_to_tensor(indexes)
            if self.indexes.dtype != tf.int32:
                raise ValueError("`indexes` must have `int32` dtype.")
            input_shape = input_shape.merge_with(self.indexes.shape)
        if input_shape.ndims is None:
            raise ValueError(
                "Number of dimensions of `scale` or `indexes` must be known."
            )
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)

    @property
    def tail_mass(self):
        return self._tail_mass

    @property
    def likelihood_bound(self):
        return self._likelihood_bound

    @property
    def range_coder_precision(self):
        return self._range_coder_precision

    @property
    def scale(self):
        return self._scale

    @property
    def scale_table(self):
        return self._scale_table

    @property
    def scale_bound(self):
        return self._scale_bound

    @property
    def mean(self):
        return self._mean

    @property
    def indexes(self):
        return self._indexes

    def _quantize(self, inputs, mode):
        """Perturb or quantize a `Tensor` and optionally dequantize.

        Arguments:
          inputs: `Tensor`. The input values.
          mode: String. Can take on one of three values: `'noise'` (adds uniform
            noise), `'dequantize'` (quantizes and dequantizes), and `'symbols'`
            (quantizes and produces integer symbols for range coder).

        Returns:
          The quantized/perturbed `inputs`. The returned `Tensor` should have type
          `self.dtype` if mode is `'noise'`, `'dequantize'`; `tf.int32` if mode is
          `'symbols'`.
        """
        # Add noise or quantize (and optionally dequantize in one step).
        half = tf.constant(0.5, dtype=self.dtype)

        if mode == "noise":
            noise = tf.random.uniform(tf.shape(inputs), -half, half)
            return tf.math.add_n([inputs, noise])

        outputs = inputs
        if self.mean is not None:
            outputs -= self.mean
        outputs = tf.math.floor(outputs + half)

        if mode == "dequantize":
            if self.mean is not None:
                outputs += self.mean
            return outputs
        else:
            assert mode == "symbols", mode
            outputs = tf.cast(outputs, tf.int32)
            return outputs

    def _dequantize(self, inputs, mode):
        """Dequantize a `Tensor`.

        The opposite to `_quantize(inputs, mode='symbols')`.

        Arguments:
          inputs: `Tensor`. The range coder symbols.
          mode: String. Must be `'dequantize'`.

        Returns:
          The dequantized `inputs`. The returned `Tensor` should have type
          `self.dtype`.
        """
        assert mode == "dequantize"
        outputs = tf.cast(inputs, self.dtype)
        if self.mean is not None:
            outputs += self.mean
        return outputs

    def _likelihood(self, inputs):
        """Compute the likelihood of the inputs under the model.

        Arguments:
          inputs: `Tensor`. The input values.

        Returns:
          `Tensor` of same shape and type as `inputs`, giving the likelihoods
          evaluated at `inputs`.
        """
        values = inputs
        if self.mean is not None:
            values -= self.mean

        # This assumes that the standardized cumulative has the property
        # 1 - c(x) = c(-x), which means we can compute differences equivalently in
        # the left or right tail of the cumulative. The point is to only compute
        # differences in the left tail. This increases numerical stability: c(x) is
        # 1 for large x, 0 for small x. Subtracting two numbers close to 0 can be
        # done with much higher precision than subtracting two numbers close to 1.
        values = abs(values)
        upper = self._standardized_cumulative((0.5 - values) / self.scale)
        lower = self._standardized_cumulative((-0.5 - values) / self.scale)
        likelihood = upper - lower

        return likelihood

    def _prepare_indexes(self, shape=None):
        del shape  # unused
        if not self.built:
            self.build(self.input_spec.shape)
        return self.indexes

    def _pmf_to_cdf(self, pmf, tail_mass, pmf_length, max_length):
        """Helper function for computing the CDF from the PMF."""

        # Prevent tensors from bouncing back and forth between host and GPU.
        with tf.device("/cpu:0"):

            def loop_body(args):
                prob, length, tail = args
                prob = tf.concat([prob[:length], tail], axis=0)
                cdf = range_coding_ops.pmf_to_quantized_cdf(
                    prob, precision=self.range_coder_precision
                )
                return tf.pad(
                    cdf, [[0, max_length - length]], mode="CONSTANT", constant_values=0
                )

            return tf.map_fn(
                loop_body,
                (pmf, pmf_length, tail_mass),
                dtype=tf.int32,
                back_prop=False,
                name="pmf_to_cdf",
            )

    def _standardized_cumulative(self, inputs):
        """Evaluate the standardized cumulative density.

        Note: This function should be optimized to give the best possible numerical
        accuracy for negative input values.

        Arguments:
          inputs: `Tensor`. The values at which to evaluate the cumulative density.

        Returns:
          A `Tensor` of the same shape as `inputs`, containing the cumulative
          density evaluated at the given inputs.
        """
        half = tf.constant(0.5, dtype=self.dtype)
        const = tf.constant(-(2 ** -0.5), dtype=self.dtype)
        # Using the complementary error function maximizes numerical precision.
        return half * tf.math.erfc(const * inputs)

    def _standardized_quantile(self, quantile):
        """Evaluate the standardized quantile function.

        This returns the inverse of the standardized cumulative function for a
        scalar.

        Arguments:
          quantile: Float. The values at which to evaluate the quantile function.

        Returns:
          A float giving the inverse CDF value.
        """
        return scipy.stats.norm.ppf(quantile)

    def build(self, input_shape):
        """Builds the entropy model.

        This function precomputes the quantized CDF table based on the scale table.
        This can be done at graph construction time. Then, it creates the graph for
        computing the indexes into that table based on the scale tensor, and then
        uses this index tensor to determine the starting positions of the PMFs for
        each scale.

        Arguments:
          input_shape: Shape of the input tensor.

        Raises:
          ValueError: If `input_shape` doesn't specify number of input dimensions.
        """
        input_shape = tf.TensorShape(input_shape)
        input_shape.assert_is_compatible_with(self.input_spec.shape)

        scale_table = tf.constant(self.scale_table, dtype=self.dtype)

        # Lower bound scales. We need to do this here, and not in __init__, because
        # the dtype may not yet be known there.
        if self.scale_bound is None:
            self._scale = math_ops.lower_bound(self._scale, scale_table[0])
        elif self.scale_bound > 0:
            self._scale = math_ops.lower_bound(self._scale, self.scale_bound)

        multiplier = -self._standardized_quantile(self.tail_mass / 2)
        pmf_center = np.ceil(np.array(self.scale_table) * multiplier).astype(int)
        pmf_length = 2 * pmf_center + 1
        max_length = np.max(pmf_length)

        # This assumes that the standardized cumulative has the property
        # 1 - c(x) = c(-x), which means we can compute differences equivalently in
        # the left or right tail of the cumulative. The point is to only compute
        # differences in the left tail. This increases numerical stability: c(x) is
        # 1 for large x, 0 for small x. Subtracting two numbers close to 0 can be
        # done with much higher precision than subtracting two numbers close to 1.
        samples = abs(np.arange(max_length, dtype=int) - pmf_center[:, None])
        samples = tf.constant(samples, dtype=self.dtype)
        samples_scale = tf.expand_dims(scale_table, 1)
        upper = self._standardized_cumulative((0.5 - samples) / samples_scale)
        lower = self._standardized_cumulative((-0.5 - samples) / samples_scale)
        pmf = upper - lower

        # Compute out-of-range (tail) masses.
        tail_mass = 2 * lower[:, :1]

        def cdf_initializer(shape, dtype=None, partition_info=None):
            del partition_info  # unused
            assert tuple(shape) == (len(pmf_length), max_length + 2)
            assert dtype == tf.int32
            return self._pmf_to_cdf(
                pmf, tail_mass, tf.constant(pmf_length, dtype=tf.int32), max_length
            )

        quantized_cdf = self.add_variable(
            "quantized_cdf",
            shape=(len(pmf_length), max_length + 2),
            initializer=cdf_initializer,
            dtype=tf.int32,
            trainable=False,
        )
        cdf_length = self.add_variable(
            "cdf_length",
            shape=(len(pmf_length),),
            initializer=tf.initializers.constant(pmf_length + 2),
            dtype=tf.int32,
            trainable=False,
        )
        # Works around a weird TF issue with reading variables inside a loop.
        self._quantized_cdf = tf.identity(quantized_cdf)
        self._cdf_length = tf.identity(cdf_length)

        # Now, if they haven't been overridden, compute the indexes into the table
        # for each of the passed-in scales.
        if not hasattr(self, "_indexes"):
            # Prevent tensors from bouncing back and forth between host and GPU.
            with tf.device("/cpu:0"):
                fill = tf.constant(len(self.scale_table) - 1, dtype=tf.int32)
                initializer = tf.fill(tf.shape(self.scale), fill)

                def loop_body(indexes, scale):
                    return indexes - tf.cast(self.scale <= scale, tf.int32)

                self._indexes = tf.foldr(
                    loop_body,
                    scale_table[:-1],
                    initializer=initializer,
                    back_prop=False,
                    name="compute_indexes",
                )

        self._offset = tf.constant(-pmf_center, dtype=tf.int32)

        super(GSMMConditional, self).build(input_shape)

    def call(self, inputs, training):
        """Pass a tensor through the bottleneck.

        Arguments:
          inputs: The tensor to be passed through the bottleneck.
          training: Boolean. If `True`, returns a differentiable approximation of
            the inputs, and their likelihoods under the modeled probability
            densities. If `False`, returns the quantized inputs and their
            likelihoods under the corresponding probability mass function. These
            quantities can't be used for training, as they are not differentiable,
            but represent actual compression more closely.

        Returns:
          values: `Tensor` with the same shape as `inputs` containing the perturbed
            or quantized input values.
          likelihood: `Tensor` with the same shape as `inputs` containing the
            likelihood of `values` under the modeled probability distributions.

        Raises:
          ValueError: if `inputs` has an integral or inconsistent `DType`, or
            inconsistent number of channels.
        """
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        if inputs.dtype.is_integer:
            raise ValueError(
                "{} can't take integer inputs.".format(type(self).__name__)
            )

        outputs = self._quantize(inputs, "noise" if training else "dequantize")
        assert outputs.dtype == self.dtype
        likelihood = self._likelihood(outputs)
        if self.likelihood_bound > 0:
            likelihood_bound = tf.constant(self.likelihood_bound, dtype=self.dtype)
            likelihood = math_ops.lower_bound(likelihood, likelihood_bound)

        if not tf.executing_eagerly():
            outputs_shape, likelihood_shape = self.compute_output_shape(inputs.shape)
            outputs.set_shape(outputs_shape)
            likelihood.set_shape(likelihood_shape)

        return outputs, likelihood

    def compress(self, inputs):
        """Compress inputs and store their binary representations into strings.

        Arguments:
          inputs: `Tensor` with values to be compressed.

        Returns:
          compressed: String `Tensor` vector containing the compressed
            representation of each batch element of `inputs`.

        Raises:
          ValueError: if `inputs` has an integral or inconsistent `DType`, or
            inconsistent number of channels.
        """
        with tf.name_scope(self._name_scope()):
            inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
            if not self.built:
                # Check input assumptions set before layer building, e.g. input rank.
                input_spec.assert_input_compatibility(
                    self.input_spec, inputs, self.name
                )
                if self.dtype is None:
                    self._dtype = inputs.dtype.base_dtype.name
                self.build(inputs.shape)

            # Check input assumptions set after layer building, e.g. input shape.
            if not tf.executing_eagerly():
                input_spec.assert_input_compatibility(
                    self.input_spec, inputs, self.name
                )
                if inputs.dtype.is_integer:
                    raise ValueError(
                        "{} can't take integer inputs.".format(type(self).__name__)
                    )

            symbols = self._quantize(inputs, "symbols")
            assert symbols.dtype == tf.int32

            ndim = self.input_spec.ndim
            indexes = self._prepare_indexes(shape=tf.shape(symbols)[1:])
            broadcast_indexes = indexes.shape.ndims != ndim
            if broadcast_indexes:
                # We can't currently broadcast over anything else but the batch axis.
                assert indexes.shape.ndims == ndim - 1
                args = (symbols,)
            else:
                args = (symbols, indexes)

            def loop_body(args):
                string = range_coding_ops.unbounded_index_range_encode(
                    args[0],
                    indexes if broadcast_indexes else args[1],
                    self._quantized_cdf,
                    self._cdf_length,
                    self._offset,
                    precision=self.range_coder_precision,
                    overflow_width=4,
                    debug_level=0,
                )
                return string

            strings = tf.map_fn(
                loop_body, args, dtype=tf.string, back_prop=False, name="compress"
            )

            if not tf.executing_eagerly():
                strings.set_shape(inputs.shape[:1])

            return strings

    def decompress(self, strings, **kwargs):
        """Decompress values from their compressed string representations.

        Arguments:
          strings: A string `Tensor` vector containing the compressed data.
          **kwargs: Model-specific keyword arguments.

        Returns:
          The decompressed `Tensor`.
        """
        with tf.name_scope(self._name_scope()):
            strings = tf.convert_to_tensor(strings, dtype=tf.string)

            indexes = self._prepare_indexes(**kwargs)
            ndim = self.input_spec.ndim
            broadcast_indexes = indexes.shape.ndims != ndim
            if broadcast_indexes:
                # We can't currently broadcast over anything else but the batch axis.
                assert indexes.shape.ndims == ndim - 1
                args = (strings,)
            else:
                args = (strings, indexes)

            def loop_body(args):
                symbols = range_coding_ops.unbounded_index_range_decode(
                    args[0],
                    indexes if broadcast_indexes else args[1],
                    self._quantized_cdf,
                    self._cdf_length,
                    self._offset,
                    precision=self.range_coder_precision,
                    overflow_width=4,
                    debug_level=0,
                )
                return symbols

            symbols = tf.map_fn(
                loop_body, args, dtype=tf.int32, back_prop=False, name="decompress"
            )

            outputs = self._dequantize(symbols, "dequantize")
            assert outputs.dtype == self.dtype

            if not tf.executing_eagerly():
                outputs.set_shape(self.input_spec.shape)

            return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        return input_shape, input_shape
