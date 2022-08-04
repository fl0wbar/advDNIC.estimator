"""General utility functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import six
from six.moves import range  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
import os
import glob
import json
import logging
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.contrib.slim as slim
from dnc import common_layers


def read_png(filename):
    """Loads a PNG image file."""
    string = tf.read_file(filename)
    image = tf.image.decode_image(string, channels=3)
    image = tf.cast(image, tf.float32)
    image /= 255
    return image


def quantize_image(image):
    image = tf.round(image * 255)
    image = tf.saturate_cast(image, tf.uint8)
    return image


def write_png(filename, image):
    """Saves an image to a PNG file."""
    image = quantize_image(image)
    string = tf.image.encode_png(image)
    return tf.write_file(filename, string)


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


# TODO(jonycgn): Use tfc.PackedTensors once new binary packages have been built.
class PackedTensors(object):
    """Packed representation of compressed tensors."""

    def __init__(self, string=None):
        self._example = tf.train.Example()
        if string:
            self.string = string

    @property
    def model(self):
        """Model identifier."""
        buf = self._example.features.feature["MD"].bytes_list.value[0]
        return buf.decode("ascii")

    @model.setter
    def model(self, value):
        self._example.features.feature["MD"].bytes_list.value[:] = [
            value.encode("ascii")
        ]

    @model.deleter
    def model(self):
        del self._example.features.feature["MD"]

    @property
    def string(self):
        """A string representation of this object."""
        return self._example.SerializeToString()

    @string.setter
    def string(self, value):
        self._example.ParseFromString(value)

    def pack(self, tensors, arrays):
        """Packs Tensor values into this object."""
        if len(tensors) != len(arrays):
            raise ValueError("`tensors` and `arrays` must have same length.")
        i = 1
        for tensor, array in zip(tensors, arrays):
            feature = self._example.features.feature[chr(i)]
            feature.Clear()
            if array.ndim != 1:
                raise RuntimeError("Unexpected tensor rank: {}.".format(array.ndim))
            if tensor.dtype.is_integer:
                feature.int64_list.value[:] = array
            elif tensor.dtype == tf.string:
                feature.bytes_list.value[:] = array
            else:
                raise RuntimeError(
                    "Unexpected tensor dtype: '{}'.".format(tensor.dtype)
                )
            i += 1
        # Delete any remaining, previously set arrays.
        while chr(i) in self._example.features.feature:
            del self._example.features.feature[chr(i)]
            i += 1

    def unpack(self, tensors):
        """Unpacks Tensor values from this object."""
        arrays = []
        for i, tensor in enumerate(tensors):
            feature = self._example.features.feature[chr(i + 1)]
            np_dtype = tensor.dtype.as_numpy_dtype
            if tensor.dtype.is_integer:
                arrays.append(np.array(feature.int64_list.value, dtype=np_dtype))
            elif tensor.dtype == tf.string:
                arrays.append(np.array(feature.bytes_list.value, dtype=np_dtype))
            else:
                raise RuntimeError(
                    "Unexpected tensor dtype: '{}'.".format(tensor.dtype)
                )
        return arrays


def archive_ckpt(ckpt_eval, ckpt_objective, ckpt_path):
    """Archive a checkpoint if the metric is better."""
    ckpt_dir, ckpt_name = os.path.split(ckpt_path)

    saved_objective_path = os.path.join(ckpt_dir, "best_objective.txt")
    saved_objective = float("-inf")
    if tf.gfile.Exists(saved_objective_path):
        with tf.gfile.GFile(saved_objective_path, "r") as f:
            saved_objective = float(f.read())
    if saved_objective > ckpt_objective:
        tf.logging.info("Ckpt %s is worse than %s", ckpt_objective, saved_objective)
        return False

    filenames = tf.gfile.Glob(ckpt_path + ".*")
    if filenames is None:
        tf.logging.info("No files to copy for checkpoint %s", ckpt_path)
        return False

    # Clear the old folder.
    dst_dir = os.path.join(ckpt_dir, "archive")
    if tf.gfile.Exists(dst_dir):
        tf.gfile.DeleteRecursively(dst_dir)
    tf.gfile.MakeDirs(dst_dir)

    # Write checkpoints.
    for f in filenames:
        dest = os.path.join(dst_dir, os.path.basename(f))
        tf.gfile.Copy(f, dest, overwrite=True)
    ckpt_state = tf.train.generate_checkpoint_state_proto(
        dst_dir, model_checkpoint_path=ckpt_name, all_model_checkpoint_paths=[ckpt_name]
    )
    with tf.gfile.GFile(os.path.join(dst_dir, "checkpoint"), "w") as f:
        f.write(str(ckpt_state))
    with tf.gfile.GFile(os.path.join(dst_dir, "best_eval.txt"), "w") as f:
        f.write("%s" % ckpt_eval)

    # Update the best objective.
    with tf.gfile.GFile(saved_objective_path, "w") as f:
        f.write("%f" % ckpt_objective)

    tf.logging.info("Copying checkpoint %s to %s", ckpt_path, dst_dir)
    return True


def get_ema_vars():
    """Get all exponential moving average (ema) variables."""
    ema_vars = tf.trainable_variables() + tf.get_collection("moving_vars")
    for v in tf.global_variables():
        # We maintain mva for batch norm moving mean and variance as well.
        if "moving_mean" in v.name or "moving_variance" in v.name:
            ema_vars.append(v)
    return list(set(ema_vars))


def _get_streaming_variable(name, shape):
    return tf.compat.v1.get_variable(
        name=name,
        shape=shape,
        dtype=tf.float64,
        initializer=tf.compat.v1.initializers.zeros(),
        trainable=False,
        collections=[
            tf.compat.v1.GraphKeys.LOCAL_VARIABLES,
            tf.compat.v1.GraphKeys.METRIC_VARIABLES,
        ],
    )


def streaming_mean_tensor_float64(values, updates_collections=None, name=None):
    """
    A version of tf.metrics.mean_tensor that handles float64 values.
    Unlike tf.metrics.mean_tensor, current implementation does not support
    distributed processing and weights.

    Args:
        values: A `Tensor` of arbitrary dimensions.
        updates_collections: An optional list of collections that `update_op` should
          be added to.
        name: An optional variable_scope name.
    Returns:
        mean: A float64 `Tensor` representing the current mean, the value of `total`
          divided by `count`.
        update_op: An operation that increments the `total` and `count` variables
          appropriately and whose value matches `mean_value`.
    Raises:
        ValueError: If `updates_collections` is not a list or tuple.
        RuntimeError: If eager execution is enabled.
  """
    # Code below copied from the implementation of tf.metrics.mean_tensor.
    if tf.executing_eagerly():
        raise RuntimeError(
            "streaming_mean_tensor_float64 is not supported when "
            "eager execution is enabled."
        )
    if values.dtype != tf.float64:
        values = tf.cast(values, tf.float64)

    with tf.compat.v1.variable_scope(name, "streaming_mean_tensor_float64", (values,)):
        total = _get_streaming_variable(name="total_tensor", shape=values.get_shape())
        count = _get_streaming_variable(name="count_tensor", shape=values.get_shape())

        num_values = tf.ones_like(values)
        update_total_op = tf.compat.v1.assign_add(total, values)
        with tf.control_dependencies([values]):
            update_count_op = tf.compat.v1.assign_add(count, num_values)

        mean_t = tf.compat.v1.div_no_nan(total, count)
        update_op = tf.compat.v1.div_no_nan(
            update_total_op, tf.maximum(update_count_op, 0), name="update_op"
        )
        if updates_collections:
            tf.compat.v1.add_to_collections(updates_collections, update_op)

        return mean_t, update_op


def streaming_covariance(x, y=None, updates_collections=None, name=None):
    """
    A streaming an unbiased version of tfp.stats.covariance.
    Implementation based on
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online.
    Args:
        x: A 2D numeric `Tensor` holding samples.
        y: Optional Tensor with same dtype and shape as `x`. Default value: `None`
          (y is effectively set to x).
        updates_collections: An optional list of collections that `update_op` should
          be added to.
        name: An optional variable_scope name.
    Returns:
        covariance: A float64 `Tensor` holding the sample covariance matrix.
        update_op: An operation that updates the internal variables appropriately
          and whose value matches `covariance`.
    Raises:
        ValueError: If `updates_collections` is not a list or tuple.
        RuntimeError: If eager execution is enabled.
    """
    if tf.executing_eagerly():
        raise RuntimeError(
            "streaming_covariance is not supported when " "eager execution is enabled."
        )
    if x.dtype != tf.float64:
        x = tf.cast(x, tf.float64)
    if y is not None and y.dtype != tf.float64:
        y = tf.cast(y, tf.float64)

    if y is None:
        y = x
    x.shape.assert_has_rank(2)
    y.shape.assert_has_rank(2)

    x_event_shape = x.get_shape()[1:]
    y_event_shape = y.get_shape()[1:]

    with tf.compat.v1.variable_scope(name, "streaming_covariance", (x, y)):
        n = _get_streaming_variable(name="n", shape=[])
        meanx = _get_streaming_variable(name="meanx", shape=x_event_shape)
        meany = _get_streaming_variable(name="meany", shape=y_event_shape)
        cov_matrix = _get_streaming_variable(
            name="cov_matrix", shape=x_event_shape.as_list() + y_event_shape.as_list()
        )

    num_values = tf.cast(tf.shape(input=x)[0], dtype=tf.float64)
    dx = tf.reduce_mean(input_tensor=x, axis=0) - meanx
    dy = tf.reduce_mean(input_tensor=y, axis=0) - meany
    # (x_1 + ... + x_n + x_{n+1} + ... + x_{n+m}) / (n + m)
    # = (x_1 + ... + x_n) / n
    #   + m * ((x_{n+1} + ... + x_{n+m}) / m - (x_1 + ... + x_n) / n) / (n + m).
    meany_update_op = tf.compat.v1.assign_add(
        meany, (num_values / (n + num_values)) * dy
    )
    with tf.control_dependencies([meany_update_op]):
        cov_matrix_update_op = tf.compat.v1.assign_add(
            cov_matrix, tf.matmul(a=x - meanx, b=y - meany_update_op, transpose_a=True)
        )
    with tf.control_dependencies([cov_matrix_update_op]):
        meanx_update_op = tf.compat.v1.assign_add(
            meanx, (num_values / (n + num_values)) * dx
        )
    with tf.control_dependencies([meanx_update_op, meany_update_op]):
        update_n_op = tf.compat.v1.assign_add(n, num_values)

    result = tf.compat.v1.div_no_nan(cov_matrix, n - 1.0)
    update_op = tf.compat.v1.div_no_nan(
        cov_matrix_update_op, update_n_op - 1.0, name="covariance_update_op"
    )
    if updates_collections:
        tf.compat.v1.add_to_collections(updates_collections, update_op)

    return result, update_op


def add_scope(scope=None, scope_fn=None):
    """Return a decorator which add a TF name/variable scope to a function.
    Note that the function returned by the decorator accept an additional 'name'
    parameter, which can overwrite the name scope given when the function is
    created.
    Args:
        scope (str): name of the scope. If None, the function name is used.
        scope_fn (fct): Either tf.name_scope or tf.variable_scope
    Returns:
        fct: add_scope decorator
    """

    def decorator(f):
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            name = kwargs.pop("name", None)  # Python 2 hack for keyword only args
            with scope_fn(name or scope or f.__name__):
                return f(*args, **kwargs)

        return decorated

    return decorator


def add_var_scope(scope=None):
    return add_scope(scope, scope_fn=tf.compat.v1.variable_scope)


def add_name_scope(scope=None):
    return add_scope(scope, scope_fn=tf.name_scope)


def _add_variable_proxy_methods(var, proxy_tensor):
    """Proxy methods of underlying variable.
    This enables our custom getters to still work with, e.g., batch norm.
    Args:
        var: Variable to proxy
        proxy_tensor: Tensor that is identity of var
    """
    proxy_tensor.read_value = lambda: tf.identity(proxy_tensor)
    proxy_tensor.assign_sub = var.assign_sub
    proxy_tensor.assign = var.assign
    proxy_tensor.initialized_value = var.initialized_value


class Parallelism(object):
    """Helper class for creating sets of parallel function calls.
    The purpose of this class is to replace this code:
      e = []
      f = []
      for i in range(len(devices)):
        with tf.device(devices[i]):
          e_, f_ = func(a[i], b[i], c)
          e.append(e_)
          f.append(f_)
    with this code:
      e, f = expert_utils.Parallelism(devices)(func, a, b, c)
    """

    def __init__(
        self,
        device_names_or_functions,
        reuse=True,
        caching_devices=None,
        daisy_chain_variables=False,
        ps_devices=None,
    ):
        """Create a Parallelism.
        Args:
          device_names_or_functions: A list of length n, containing device names
            or device functions (see `tf.device`)
          reuse: True or None.  Whether to reuse variables created in the first
            replica in the subsequent replicas.
          caching_devices: Either `None`, or a list of length n containing device
            names.
          daisy_chain_variables: a boolean - if true, then copies variables in a
            daisy chain between devices.
          ps_devices: list<str>, list of devices for experts.
        Returns:
          a Parallelism.
        """
        assert device_names_or_functions
        self._devices = device_names_or_functions
        self._n = len(device_names_or_functions)
        self._reuse = reuse
        self._caching_devices = self._maybe_repeat(caching_devices)
        self._daisy_chain_variables = daisy_chain_variables
        self._ps_devices = ps_devices or [""]

    def __call__(self, fn, *args, **kwargs):
        """A parallel set of function calls (using the specified devices).
        Args:
          fn: a function or a list of n functions.
          *args: additional args.  Each arg should either be not a list, or a list
             of length n.
          **kwargs: additional keyword args.  Each arg should either be not a
             list, or a list of length n.
        Returns:
          either a single list of length n (if fn does not return a tuple), or a
          tuple of lists of length n (if fn returns a tuple).
        """
        # Construct lists or args and kwargs for each function.
        if args:
            my_args = transpose_list_of_lists([self._maybe_repeat(arg) for arg in args])
        else:
            my_args = [[] for _ in range(self.n)]
        my_kwargs = [{} for _ in range(self.n)]
        for k, v in six.iteritems(kwargs):
            vals = self._maybe_repeat(v)
            for i in range(self.n):
                my_kwargs[i][k] = vals[i]

        # Construct lists of functions.
        fns = self._maybe_repeat(fn)

        # Now make the parallel call.
        outputs = []
        cache = {}
        tensor_to_var = {}
        for i in range(self.n):

            def daisy_chain_getter(getter, name, *args, **kwargs):
                """Get a variable and cache in a daisy chain."""
                device_var_key = (self._devices[i], name)
                if device_var_key in cache:
                    # if we have the variable on the correct device, return it.
                    return cache[device_var_key]
                if name in cache:
                    # if we have it on a different device, copy it from the last device
                    last_device_v = cache[name]
                    var = tensor_to_var[last_device_v]
                    v = tf.identity(last_device_v)
                else:
                    var = getter(name, *args, **kwargs)
                    v = var.read_value()

                # keep track of the original variable
                tensor_to_var[v] = var
                _add_variable_proxy_methods(tensor_to_var[v], v)
                # update the cache
                cache[name] = v
                cache[device_var_key] = v
                return v

            # Variable scope will not reset caching_device on reused variables,
            # so we make a custom getter that uses identity to cache the variable.
            # pylint: disable=cell-var-from-loop
            def caching_getter(getter, name, *args, **kwargs):
                """Cache variables on device."""
                key = (self._caching_devices[i], name)
                if key in cache:
                    return cache[key]

                v = getter(name, *args, **kwargs)
                with tf.device(self._caching_devices[i]):
                    ret = v.read_value()
                _add_variable_proxy_methods(v, ret)
                cache[key] = ret
                return ret

            if self._daisy_chain_variables:
                custom_getter = daisy_chain_getter
            elif self._caching_devices[i]:
                custom_getter = caching_getter
            else:
                custom_getter = None
            # pylint: enable=cell-var-from-loop
            with tf.name_scope("parallel_%d" % i):
                with tf.variable_scope(
                    tf.get_variable_scope() if self._reuse else "parallel_%d" % i,
                    reuse=True if i > 0 and self._reuse else None,
                    caching_device=self._caching_devices[i],
                    custom_getter=custom_getter,
                ):
                    # TODO(noam, epot, avaswani)
                    # Allows for passing no device in case you want to default to the
                    # existing device. This is needed when we put all experts on a single
                    # device, for example in local_moe.
                    if self._devices[i] != DEFAULT_DEV_STRING:
                        with tf.device(self._devices[i]):
                            outputs.append(fns[i](*my_args[i], **my_kwargs[i]))
                    else:
                        outputs.append(fns[i](*my_args[i], **my_kwargs[i]))
        if isinstance(outputs[0], tuple):
            outputs = list(zip(*outputs))
            outputs = tuple([list(o) for o in outputs])
        return outputs

    @property
    def n(self):
        return self._n

    @property
    def devices(self):
        return self._devices

    @property
    def ps_devices(self):
        return self._ps_devices

    def _maybe_repeat(self, x):
        """Utility function for processing arguments that are singletons or lists.
    Args:
      x: either a list of self.n elements, or not a list.
    Returns:
      a list of self.n elements.
    """
        if isinstance(x, list):
            assert len(x) == self.n
            return x
        else:
            return [x] * self.n


def _rowwise_unsorted_segment_sum(values, indices, n):
    """UnsortedSegmentSum on each row.
    Args:
        values: a `Tensor` with shape `[batch_size, k]`.
        indices: an integer `Tensor` with shape `[batch_size, k]`.
        n: an integer.
    Returns:
        A `Tensor` with the same type as `values` and shape `[batch_size, n]`.
    """
    batch, k = tf.unstack(tf.shape(indices), num=2)
    indices_flat = tf.reshape(indices, [-1]) + tf.div(tf.range(batch * k), k) * n
    ret_flat = tf.unsorted_segment_sum(
        tf.reshape(values, [-1]), indices_flat, batch * n
    )
    return tf.reshape(ret_flat, [batch, n])


def _normal_distribution_cdf(x, stddev):
    """Evaluates the CDF of the normal distribution.
    Normal distribution with mean 0 and standard deviation stddev,
    evaluated at x=x.
    input and output `Tensor`s have matching shapes.
    Args:
        x: a `Tensor`
        stddev: a `Tensor` with the same shape as `x`.
    Returns:
        a `Tensor` with the same shape as `x`.
    """
    return 0.5 * (1.0 + tf.erf(x / (math.sqrt(2) * stddev + 1e-20)))


def _prob_in_top_k(clean_values, noisy_values, noise_stddev, noisy_top_values, k):
    """Helper function to NoisyTopKGating.
    Computes the probability that value is in top k, given different random noise.
    This gives us a way of backpropagating from a loss that balances the number
    of times each expert is in the top k experts per example.
    In the case of no noise, pass in None for noise_stddev, and the result will
    not be differentiable.
    Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        k: an integer.
    Returns:
        a `Tensor` of shape [batch, n].
    """
    batch = tf.shape(clean_values)[0]
    m = tf.shape(noisy_top_values)[1]
    top_values_flat = tf.reshape(noisy_top_values, [-1])
    # we want to compute the threshold that a particular value would have to
    # exceed in order to make the top k.  This computation differs depending
    # on whether the value is already in the top k.
    threshold_positions_if_in = tf.range(batch) * m + k
    threshold_if_in = tf.expand_dims(
        tf.gather(top_values_flat, threshold_positions_if_in), 1
    )
    is_in = tf.greater(noisy_values, threshold_if_in)
    if noise_stddev is None:
        return tf.to_float(is_in)
    threshold_positions_if_out = threshold_positions_if_in - 1
    threshold_if_out = tf.expand_dims(
        tf.gather(top_values_flat, threshold_positions_if_out), 1
    )
    # is each value currently in the top k.
    prob_if_in = _normal_distribution_cdf(clean_values - threshold_if_in, noise_stddev)
    prob_if_out = _normal_distribution_cdf(
        clean_values - threshold_if_out, noise_stddev
    )
    prob = tf.where(is_in, prob_if_in, prob_if_out)
    return prob


def cv_squared(x):
    """The squared coefficient of variation of a sample.
    Useful as a loss to encourage a positive distribution to be more uniform.
    Epsilons added for numerical stability.
    Returns 0 for an empty Tensor.
    Args:
        x: a `Tensor`.
    Returns:
        a `Scalar`.
    """
    epsilon = 1e-10
    float_size = tf.to_float(tf.size(x)) + epsilon
    mean = tf.reduce_sum(x) / float_size
    variance = tf.reduce_sum(tf.squared_difference(x, mean)) / float_size
    return variance / (tf.square(mean) + epsilon)


def _gates_to_load(gates):
    """Compute the true load per expert, given the gates.
    The load is the number of examples for which the corresponding gate is >0.
    Args:
        gates: a `Tensor` of shape [batch_size, n]
    Returns:
        a float32 `Tensor` of shape [n]
    """
    return tf.reduce_sum(tf.to_float(gates > 0), 0)


def _my_top_k(x, k):
    """GPU-compatible version of top-k that works for very small constant k.
    Calls argmax repeatedly.
    tf.nn.top_k is implemented for GPU, but the gradient, sparse_to_dense,
    seems not to be, so if we use tf.nn.top_k, then both the top_k and its
    gradient go on cpu.  Once this is not an issue, this function becomes
    obsolete and should be replaced by tf.nn.top_k.
    Args:
        x: a 2d Tensor.
        k: a small integer.
    Returns:
        values: a Tensor of shape [batch_size, k]
        indices: a int32 Tensor of shape [batch_size, k]
    """
    if k > 10:
        return tf.nn.top_k(x, k)
    values = []
    indices = []
    depth = tf.shape(x)[1]
    for i in range(k):
        values.append(tf.reduce_max(x, 1))
        argmax = tf.argmax(x, 1)
        indices.append(argmax)
        if i + 1 < k:
            x += tf.one_hot(argmax, depth, -1e9)
    return tf.stack(values, axis=1), tf.to_int32(tf.stack(indices, axis=1))


def vq_gating(x, num_experts, k, bneck, hparams=None, name="vq_gating"):
    """VQ gating.
    Args:
        x: input Tensor with shape [batch_size, input_size]
        num_experts: an integer
        k: an integer - number of experts per example
        bneck: a bottleneck object
        hparams: optional hparams
        name: an optional string
    Returns:
        gates: a Tensor with shape [batch_size, num_experts]
        load: a Tensor with shape [num_experts]
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        if hparams.use_scales:
            scales = tf.get_variable(
                "scales", [num_experts], tf.float32, initializer=tf.ones_initializer()
            )
            scales = tf.nn.softmax(scales)
            hparams.scales = scales
        input_size = x.get_shape().as_list()[-1]
        batch_size = common_layers.shape_list(x)[0]

        if k > 1:
            # first project into two dense layers, chop and discretize, and gate
            # TODO(avaswani): Maybe scale the embeddings flowing out of the experts.
            # We might want to do this to match the computation being done by topk
            x = tf.layers.dense(x, input_size * k)
            # x goes from [batch_size, input_size*k] to [batch_size*k, input_size]
            x = tf.reshape(x, [batch_size * k, input_size])
        inputs = tf.expand_dims(x, axis=1)
        inputs = tf.expand_dims(inputs, axis=1)
        # VQ hparams
        hparams.z_size = int(math.log(num_experts, 2))
        hparams.hidden_size = input_size
        hparams.top_k = k
        d = bneck.discrete_bottleneck(inputs)
        centroids = None
        exp_discrete = d["discrete"]
        embed_lookup = d["embed"]
        extra_loss = d["loss"]
        if hparams.residual_centroids:
            centroids = embed_lookup(exp_discrete)  # gives the centroids
        top_k_indices = tf.squeeze(exp_discrete, axis=1)
        tf.summary.histogram("discrete_counts", top_k_indices)
        # if k > 1, then we need to reshape top_k_indices from [batch_size*k, 1]
        # to [batch_size, k]
        if k > 1:
            top_k_indices = tf.reshape(top_k_indices, [batch_size, k])
        # get the top k gates
        top_k_gates = tf.ones([batch_size, k])
        # This will be a `Tensor` of shape `[batch_size, n]`, with zeros in the
        # positions corresponding to all but the top k experts per example.
        gates = _rowwise_unsorted_segment_sum(top_k_gates, top_k_indices, num_experts)
        # Compute count per expert from the gates.
        # gates has shape [batch_size, num_experts]
        # count per expert has shape [num_experts, 1]
        count_per_expert = tf.reduce_sum(gates, axis=0)
        if hparams.use_scales:
            scale_loss = tf.reduce_mean(tf.to_float(count_per_expert) * scales)
            extra_loss += scale_loss
        if common_layers.should_generate_summaries():
            tf.summary.histogram("vq_loss", extra_loss)
            tf.summary.historgram("scale_loss", scale_loss)
        return gates, extra_loss, centroids


def noisy_top_k_gating(
    x,
    num_experts,
    train,
    k=2,
    initializer=tf.zeros_initializer(),
    noisy_gating=True,
    noise_epsilon=1e-2,
    name=None,
):
    """Noisy top-k gating.
    See paper: https://arxiv.org/abs/1701.06538.
    Args:
        x: input Tensor with shape [batch_size, input_size]
        num_experts: an integer
        train: a boolean - we only add noise at training time.
        k: an integer - number of experts per example
        initializer: an initializer
        noisy_gating: a boolean
        noise_epsilon: a float
        name: an optional string
    Returns:
        gates: a Tensor with shape [batch_size, num_experts]
        load: a Tensor with shape [num_experts]
    """
    with tf.variable_scope(name, default_name="noisy_top_k_gating"):
        input_size = x.get_shape().as_list()[-1]
        w_gate = tf.get_variable(
            "w_gate", [input_size, num_experts], tf.float32, initializer
        )
        if noisy_gating:
            w_noise = tf.get_variable(
                "w_noise", [input_size, num_experts], tf.float32, initializer
            )
        clean_logits = tf.matmul(x, w_gate)
        if noisy_gating:
            raw_noise_stddev = tf.matmul(x, w_noise)
            noise_stddev = (tf.nn.softplus(raw_noise_stddev) + noise_epsilon) * (
                tf.to_float(train)
            )
            noisy_logits = clean_logits + (
                tf.random_normal(tf.shape(clean_logits)) * noise_stddev
            )
            logits = noisy_logits
            if common_layers.should_generate_summaries():
                tf.summary.histogram("noisy_logits", noisy_logits)
                tf.summary.histogram("noise_stddev", noise_stddev)
        else:
            logits = clean_logits
        top_logits, top_indices = _my_top_k(logits, min(k + 1, num_experts))
        # top k logits has shape [batch, k]
        top_k_logits = tf.slice(top_logits, [0, 0], [-1, k])
        top_k_indices = tf.slice(top_indices, [0, 0], [-1, k])
        top_k_gates = tf.nn.softmax(top_k_logits)
        # This will be a `Tensor` of shape `[batch_size, n]`, with zeros in the
        # positions corresponding to all but the top k experts per example.
        gates = _rowwise_unsorted_segment_sum(top_k_gates, top_k_indices, num_experts)
        if noisy_gating and k < num_experts:
            load = tf.reduce_sum(
                _prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits, k),
                0,
            )
        else:
            load = _gates_to_load(gates)
        if common_layers.should_generate_summaries():
            tf.summary.histogram("importance", tf.reduce_sum(gates, 0))
            tf.summary.histogram("load", load)
        return gates, load


class PadRemover(object):
    """Helper to remove padding from a tensor before sending to the experts.
  The padding is computed for one reference tensor containing the padding mask
  and then can be applied to any other tensor of shape [dim_origin,...].
  Ex:
      input = [
        [tok1, tok2],
        [tok3, tok4],
        [0, 0],
        [0, 0],
        [tok5, tok6],
        [0, 0],
      ]
      output = [
        [tok1, tok2],
        [tok3, tok4],
        [tok5, tok6],
      ]
  """

    def __init__(self, pad_mask):
        """Compute and store the location of the padding.
    Args:
      pad_mask (tf.Tensor): Reference padding tensor of shape
        [batch_size,length] or [dim_origin] (dim_origin=batch_size*length)
        containing non-zeros positive values to indicate padding location.
    """
        self.nonpad_ids = None
        self.dim_origin = None

        with tf.name_scope("pad_reduce/get_ids"):
            pad_mask = tf.reshape(pad_mask, [-1])  # Flatten the batch
            # nonpad_ids contains coordinates of zeros rows (as pad_mask is
            # float32, checking zero equality is done with |x| < epsilon, with
            # epsilon=1e-9 as standard, here pad_mask only contains positive values
            # so tf.abs would be redundant)
            self.nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))
            self.dim_origin = tf.shape(pad_mask)[:1]

    def remove(self, x):
        """Remove padding from the given tensor.
    Args:
      x (tf.Tensor): of shape [dim_origin,...]
    Returns:
      a tensor of shape [dim_compressed,...] with dim_compressed <= dim_origin
    """
        with tf.name_scope("pad_reduce/remove"):
            x_shape = x.get_shape().as_list()
            x = tf.gather_nd(x, indices=self.nonpad_ids)
            if not tf.executing_eagerly():
                # This is a hack but for some reason, gather_nd return a tensor of
                # undefined shape, so the shape is set up manually
                x.set_shape([None] + x_shape[1:])
        return x

    def restore(self, x):
        """Add padding back to the given tensor.
    Args:
      x (tf.Tensor): of shape [dim_compressed,...]
    Returns:
      a tensor of shape [dim_origin,...] with dim_compressed >= dim_origin. The
      dim is restored from the original reference tensor
    """
        with tf.name_scope("pad_reduce/restore"):
            x = tf.scatter_nd(
                indices=self.nonpad_ids,
                updates=x,
                shape=tf.concat([self.dim_origin, tf.shape(x)[1:]], axis=0),
            )
        return x
