"""
This file contains autoregressive model unit blocks
"""
import functools
import math
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.python.ops import variables
import dnc.niclib.layers as advlayers
from dnc.autoregressive.neuralops.ops.layers import resize, flow_to_image_tf
from dnc.autoregressive.layers import MaskedConv2D


dense_keras = tf.keras.layers.Dense
conv2d_keras = advlayers.SignalConvolutionLayer
# conv2d_keras = advlayers.ConvolutionLayer
# deconv2d_keras = advlayers.UpsampleLayer
deconv2d_keras = functools.partial(advlayers.SignalConvolutionLayer, transpose=True)
masked_conv2d_keras = MaskedConv2D


def passthrough(obj, value):
    return value


try:
    variables.Variable._build_initializer_expr = passthrough
except:  # older versions of TF don't have this
    pass


def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

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


def shape_list_2(x):
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


def int_shape(x):
    return list(map(int, x.get_shape()))


def static_shape_list(x):
    return x.get_shape().as_list()


def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    axis = len(x.get_shape()) - 1
    return tf.nn.elu(tf.concat([x, -x], axis))


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keep_dims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x - m2), axis))


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis, keep_dims=True)
    return x - m - tf.log(tf.reduce_sum(tf.exp(x - m), axis, keep_dims=True))


def discretized_mix_logistic_loss(x, l, sum_all=True):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been
    rescaled to [-1,1] interval """
    xs = shape_list(x)  # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = shape_list(l)  # predicted distribution, e.g. (B,32,32,100)
    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    means = l[:, :, :, :, :nr_mix]
    log_scales = tf.maximum(l[:, :, :, :, nr_mix : 2 * nr_mix], -7.0)
    coeffs = tf.nn.tanh(l[:, :, :, :, 2 * nr_mix : 3 * nr_mix])
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix])
    m2 = tf.reshape(
        means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :],
        [xs[0], xs[1], xs[2], 1, nr_mix],
    )
    m3 = tf.reshape(
        means[:, :, :, 2, :]
        + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :]
        + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :],
        [xs[0], xs[1], xs[2], 1, nr_mix],
    )
    means = tf.concat(
        [tf.reshape(means[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix]), m2, m3], 3
    )
    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = tf.nn.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2.0 * tf.nn.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme
    # cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999,
    # log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never
    # happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of
    # selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output,
    # it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value
    log_probs = tf.where(
        x < -0.999,
        log_cdf_plus,
        tf.where(
            x > 0.999,
            log_one_minus_cdf_min,
            tf.where(
                cdf_delta > 1e-5,
                tf.log(tf.maximum(cdf_delta, 1e-12)),
                log_pdf_mid - np.log(127.5),
            ),
        ),
    )

    log_probs = tf.reduce_sum(log_probs, 3) + log_prob_from_logits(logit_probs)
    if sum_all:
        return -tf.reduce_sum(log_sum_exp(log_probs))
    else:
        return -tf.reduce_sum(log_sum_exp(log_probs), [1, 2])


def discretized_mix_logistic_loss_per_chn(x, lr, lg, lb, sum_all=True):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been
    rescaled to [-1,1] interval """
    xs = shape_list(x)  # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = shape_list(lr)  # predicted distribution, e.g. (B,32,32,100)
    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 3)
    logit_probs = lr[:, :, :, :nr_mix]
    means = tf.concat(
        [
            lr[:, :, :, None, nr_mix : nr_mix * 2],
            lg[:, :, :, None, nr_mix : nr_mix * 2],
            lb[:, :, :, None, nr_mix : nr_mix * 2],
        ],
        axis=-2,
    )
    log_scales = tf.concat(
        [
            lr[:, :, :, None, nr_mix * 2 : nr_mix * 3],
            lg[:, :, :, None, nr_mix * 2 : nr_mix * 3],
            lb[:, :, :, None, nr_mix * 2 : nr_mix * 3],
        ],
        axis=-2,
    )
    log_scales = tf.maximum(log_scales, -7.0)

    x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix])
    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = tf.nn.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2.0 * tf.nn.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme
    # cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999,
    # log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never
    # happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of
    # selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output,
    # it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value
    log_probs = tf.where(
        x < -0.999,
        log_cdf_plus,
        tf.where(
            x > 0.999,
            log_one_minus_cdf_min,
            tf.where(
                cdf_delta > 1e-5,
                tf.log(tf.maximum(cdf_delta, 1e-12)),
                log_pdf_mid - np.log(127.5),
            ),
        ),
    )

    log_probs = tf.reduce_sum(log_probs, 3) + log_prob_from_logits(logit_probs)
    if sum_all:
        return -tf.reduce_sum(log_sum_exp(log_probs))
    else:
        return -tf.reduce_sum(log_sum_exp(log_probs), [1, 2])


def sample_from_discretized_mix_logistic(l, nr_mix):
    ls = shape_list(l)
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    sel = tf.one_hot(
        tf.argmax(
            logit_probs
            - tf.log(
                -tf.log(
                    tf.random_uniform(
                        logit_probs.get_shape(), minval=1e-5, maxval=1.0 - 1e-5
                    )
                )
            ),
            3,
        ),
        depth=nr_mix,
        dtype=tf.float32,
    )
    sel = tf.reshape(sel, xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = tf.reduce_sum(l[:, :, :, :, :nr_mix] * sel, 4)
    log_scales = tf.maximum(
        tf.reduce_sum(l[:, :, :, :, nr_mix : 2 * nr_mix] * sel, 4), -7.0
    )
    coeffs = tf.reduce_sum(tf.nn.tanh(l[:, :, :, :, 2 * nr_mix : 3 * nr_mix]) * sel, 4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = tf.random_uniform(means.get_shape(), minval=1e-5, maxval=1.0 - 1e-5)
    x = means + tf.exp(log_scales) * (tf.log(u) - tf.log(1.0 - u))
    x0 = tf.minimum(tf.maximum(x[:, :, :, 0], -1.0), 1.0)
    x1 = tf.minimum(tf.maximum(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.0), 1.0)
    x2 = tf.minimum(
        tf.maximum(
            x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, -1.0
        ),
        1.0,
    )
    return tf.concat(
        [
            tf.reshape(x0, xs[:-1] + [1]),
            tf.reshape(x1, xs[:-1] + [1]),
            tf.reshape(x2, xs[:-1] + [1]),
        ],
        3,
    )


def get_var_maybe_avg(var_name, ema, **kwargs):
    """ utility for retrieving polyak averaged params """
    v = tf.get_variable(var_name, **kwargs)
    if ema is not None:
        v = ema.average(v)
    return v


def get_vars_maybe_avg(var_names, ema, **kwargs):
    """ utility for retrieving polyak averaged params """
    vars = []
    for vn in var_names:
        vars.append(get_var_maybe_avg(vn, ema, **kwargs))
    return vars


def adam_updates(params, cost_or_grads, lr=0.001, mom1=0.9, mom2=0.999, eps=1e-8):
    """ Adam optimizer """
    updates = []
    if type(cost_or_grads) is not list:
        grads = tf.gradients(cost_or_grads, params)
    else:
        grads = cost_or_grads
    t = tf.Variable(1.0, "adam_t")
    for p, g in zip(params, grads):
        mg = tf.Variable(tf.zeros(p.get_shape()), p.name + "_adam_mg")
        if mom1 > 0:
            v = tf.Variable(tf.zeros(p.get_shape()), p.name + "_adam_v")
            v_t = mom1 * v + (1.0 - mom1) * g
            v_hat = v_t / (1.0 - tf.pow(mom1, t))
            updates.append(v.assign(v_t))
        else:
            v_hat = g
        mg_t = mom2 * mg + (1.0 - mom2) * tf.square(g)
        mg_hat = mg_t / (1.0 - tf.pow(mom2, t))
        g_t = v_hat / tf.sqrt(mg_hat + eps)
        p_t = p - lr * g_t
        updates.append(mg.assign(mg_t))
        updates.append(p.assign(p_t))
    updates.append(t.assign_add(1))
    return tf.group(*updates)


def get_name(layer_name, counters):
    """ utlity for keeping track of layer names """
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + "_" + str(counters[layer_name])
    counters[layer_name] += 1
    return name


@add_arg_scope
def dense(
    x,
    num_units,
    nonlinearity=None,
    init_scale=1.0,
    counters={},
    init=False,
    ema=None,
    **kwargs
):
    """ fully connected layer """
    name = get_name("dense", counters)
    with tf.variable_scope(name):
        if init:
            # data based initialization of parameters
            V = tf.get_variable(
                "V",
                [int(x.get_shape()[1]), num_units],
                tf.float32,
                tf.random_normal_initializer(0, 0.05),
                trainable=True,
            )
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0])
            x_init = tf.matmul(x, V_norm)
            m_init, v_init = tf.nn.moments(x_init, [0])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            g = tf.get_variable(
                "g", dtype=tf.float32, initializer=scale_init, trainable=True
            )
            b = tf.get_variable(
                "b", dtype=tf.float32, initializer=-m_init * scale_init, trainable=True
            )
            x_init = tf.reshape(scale_init, [1, num_units]) * (
                x_init - tf.reshape(m_init, [1, num_units])
            )
            if nonlinearity is not None:
                x_init = nonlinearity(x_init)
            return x_init

        else:
            V, g, b = get_vars_maybe_avg(["V", "g", "b"], ema)
            # tf.assert_variables_initialized([V, g, b])

            # use weight normalization (Salimans & Kingma, 2016)
            x = tf.matmul(x, V)
            scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
            x = tf.reshape(scaler, [1, num_units]) * x + tf.reshape(b, [1, num_units])

            # apply nonlinearity
            if nonlinearity is not None:
                x = nonlinearity(x)
            return x


@add_arg_scope
def conv2d(
    x,
    num_filters,
    filter_size=[3, 3],
    stride=[1, 1],
    pad="SAME",
    nonlinearity=None,
    init_scale=1.0,
    counters={},
    init=False,
    ema=None,
    **kwargs
):
    """ convolutional layer """
    name = get_name("conv2d", counters)
    with tf.variable_scope(name):
        if init:
            # data based initialization of parameters
            V = tf.get_variable(
                "V",
                filter_size + [int(x.get_shape()[-1]), num_filters],
                tf.float32,
                tf.random_normal_initializer(0, 0.05),
                trainable=True,
            )
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0, 1, 2])
            x_init = tf.nn.conv2d(x, V_norm, [1] + stride + [1], pad)
            m_init, v_init = tf.nn.moments(x_init, [0, 1, 2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-8)
            g = tf.get_variable(
                "g", dtype=tf.float32, initializer=scale_init, trainable=True
            )
            b = tf.get_variable(
                "b", dtype=tf.float32, initializer=-m_init * scale_init, trainable=True
            )
            x_init = tf.reshape(scale_init, [1, 1, 1, num_filters]) * (
                x_init - tf.reshape(m_init, [1, 1, 1, num_filters])
            )
            if nonlinearity is not None:
                x_init = nonlinearity(x_init)
            return x_init

        else:
            V, g, b = get_vars_maybe_avg(["V", "g", "b"], ema)
            # tf.assert_variables_initialized([V, g, b])

            # use weight normalization (Salimans & Kingma, 2016)
            W = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1, 2])

            # calculate convolutional layer output
            x = tf.nn.bias_add(tf.nn.conv2d(x, W, [1] + stride + [1], pad), b)

            # apply nonlinearity
            if nonlinearity is not None:
                x = nonlinearity(x)
            return x


@add_arg_scope
def maskconv2d(
    xtensor,
    num_filters,
    filter_size=[3, 3],
    stride=[1, 1],
    pad="SAME",
    mask_type="B",
    nonlinearity=None,
    init_scale=1.0,
    counters={},
    init=False,
    ema=None,
    **kwargs
):
    """ masked convolutional layer """
    name = get_name("maskconv2d", counters)
    with tf.variable_scope(name):
        maskW = np.ones(filter_size + [int(xtensor.get_shape()[-1]), num_filters])
        assert maskW.shape[0] == maskW.shape[1]
        filter_center = (filter_size[0]) / 2
        maskW[math.ceil(filter_center):] = 0
        maskW[math.floor(filter_center):, math.ceil(filter_center)] = 0
        mask_op = np.greater_equal if mask_type == "A" else np.greater
        for i in range(num_filters):
            for j in range(num_filters):
                if mask_op(i, j):
                    maskW[math.floor(filter_center), math.floor(filter_center), i::num_filters, j::num_filters] = 0

        maskW = tf.constant(maskW, dtype=tf.float32)

        if init:
            # data based initialization of parameters
            V = tf.get_variable(
                name="V",
                shape=filter_size + [int(xtensor.get_shape()[-1]), num_filters],
                dtype=tf.float32,
                initializer=tf.initializers.he_normal(seed=None),
                trainable=True,
            )
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0, 1, 2])
            # apply mask to the filter tensor in conv2d
            x_init = tf.nn.conv2d(xtensor, V_norm * maskW, [1] + stride + [1], pad)
            m_init, v_init = tf.nn.moments(x_init, [0, 1, 2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-8)
            g = tf.get_variable(
                "g", dtype=tf.float32, initializer=scale_init, trainable=True
            )
            b = tf.get_variable(
                "b", dtype=tf.float32, initializer=-m_init * scale_init, trainable=True
            )
            x_init = tf.reshape(scale_init, [1, 1, 1, num_filters]) * (
                x_init - tf.reshape(m_init, [1, 1, 1, num_filters])
            )
            if nonlinearity is not None:
                x_init = nonlinearity(x_init)
            return x_init

        else:
            # V, g, b = get_vars_maybe_avg(["V", "g", "b"], ema, dtype=tf.float32, initializer=tf.initializers.he_normal(seed=None))
            # tf.assert_variables_initialized([V, g, b])

            # data based initialization of parameters

            V = tf.get_variable(
                name="V",
                shape=filter_size + [int(xtensor.get_shape()[-1]), num_filters],
                dtype=tf.float32,
                initializer=tf.initializers.he_normal(seed=None),
                trainable=True,
            )
            m_init, v_init = tf.nn.moments(xtensor, [0, 1, 2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            g = tf.get_variable(
                "g", dtype=tf.float32, initializer=scale_init, trainable=True
            )
            tf.logging.info(">>> MASKED-CONV2D> g :NAME:%s, :SHAPE:%s" % (g.name, g.shape))
            b = tf.get_variable(
                "b", dtype=tf.float32, initializer=-m_init * scale_init, trainable=True
            )
            tf.logging.info(">>> MASKED-CONV2D> b :NAME:%s, :SHAPE:%s" % (b.name, b.shape))

            tf.assert_variables_initialized([V, g, b])

            # use weight normalization (Salimans & Kingma, 2016)
            W = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1, 2])
            # apply mask to the filter tensor in conv2d
            # calculate convolutional layer output
            xtensor = tf.nn.bias_add(tf.nn.conv2d(xtensor, W * maskW, [1] + stride + [1], pad), b)

            # apply nonlinearity
            if nonlinearity is not None:
                xtensor = nonlinearity(xtensor)
            return xtensor



@add_arg_scope
def deconv2d(
    x,
    num_filters,
    filter_size=[3, 3],
    stride=[1, 1],
    pad="SAME",
    nonlinearity=None,
    init_scale=1.0,
    counters={},
    init=False,
    ema=None,
    **kwargs
):
    """ transposed convolutional layer """
    name = get_name("deconv2d", counters)
    xs = shape_list(x)
    if pad == "SAME":
        target_shape = [xs[0], xs[1] * stride[0], xs[2] * stride[1], num_filters]
    else:
        target_shape = [
            xs[0],
            xs[1] * stride[0] + filter_size[0] - 1,
            xs[2] * stride[1] + filter_size[1] - 1,
            num_filters,
        ]
    with tf.variable_scope(name):
        if init:
            # data based initialization of parameters
            V = tf.get_variable(
                "V",
                filter_size + [num_filters, int(x.get_shape()[-1])],
                tf.float32,
                tf.random_normal_initializer(0, 0.05),
                trainable=True,
            )
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0, 1, 3])
            x_init = tf.nn.conv2d_transpose(
                x, V_norm, target_shape, [1] + stride + [1], padding=pad
            )
            m_init, v_init = tf.nn.moments(x_init, [0, 1, 2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-8)
            g = tf.get_variable(
                "g", dtype=tf.float32, initializer=scale_init, trainable=True
            )
            b = tf.get_variable(
                "b", dtype=tf.float32, initializer=-m_init * scale_init, trainable=True
            )
            x_init = tf.reshape(scale_init, [1, 1, 1, num_filters]) * (
                x_init - tf.reshape(m_init, [1, 1, 1, num_filters])
            )
            if nonlinearity is not None:
                x_init = nonlinearity(x_init)
            return x_init

        else:
            V, g, b = get_vars_maybe_avg(["V", "g", "b"], ema)
            # tf.assert_variables_initialized([V, g, b])

            # use weight normalization (Salimans & Kingma, 2016)
            W = tf.reshape(g, [1, 1, num_filters, 1]) * tf.nn.l2_normalize(V, [0, 1, 3])

            # calculate convolutional layer output
            x = tf.nn.conv2d_transpose(
                x, W, target_shape, [1] + stride + [1], padding=pad
            )
            x = tf.nn.bias_add(x, b)

            # apply nonlinearity
            if nonlinearity is not None:
                x = nonlinearity(x)
            return x


@add_arg_scope
def nin(x, num_units, **kwargs):
    """ a network in network layer (1x1 CONV) """
    with tf.variable_scope("nin"):
        s = shape_list(x)
        # x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
        # x = dense_keras(units=num_units)(x)
        x = conv2d_keras(
                filters=num_units,
                kernel_size=[1, 1],
                padding="same",
                strides=[1, 1],
                norm_type="None",
                self_attention=False,
                **kwargs
            )(x)
        # x = dense(x, num_units, **kwargs)
        return tf.reshape(x, s[:-1] + [num_units])


@add_arg_scope
def maskconv3x3(x, in_channels, out_channels, train: bool = True):
    """
        Masked 3x3 kernel 2d convolution for Autoregressive computation
    """
    def _conv_weight_variable(W, h, in_ch, out_ch, train: bool = True):
        d = 1.0 / np.sqrt(in_ch * W * h)
        initial = tf.random.truncated_normal([W, h, in_ch, out_ch], stddev=d)
        return tf.Variable(initial, trainable=train)

    def _conv_bias_variable(W, h, in_ch, out_ch, train: bool = True):
        d = 1.0 / np.sqrt(in_ch * W * h)
        initial = tf.constant([0.1 * d] * out_ch, dtype=tf.float32)
        return tf.Variable(initial, trainable=train)

    W = _conv_weight_variable(3, 3, in_channels, out_channels, train=train)
    o = np.zeros((in_channels, out_channels), np.float32)
    i = np.ones((in_channels, out_channels), np.float32)
    maskW = np.array(
        [[i, i, i],
        [i, i, o],
        [o, o, o]]
    )

    maskW = tf.constant(maskW, dtype=tf.float32)
    b = _conv_bias_variable(3, 3, in_channels, out_channels, train=train)
    conv = tf.nn.conv2d(input, W * maskW, strides=[1, 1, 1, 1], padding='SAME')
    out = conv + b
    return out

""" meta-layer consisting of multiple base layers """


@add_arg_scope
def gated_resnet(
    x,
    a=None,
    h=None,
    nonlinearity=concat_elu,
    conv=conv2d,
    init=False,
    counters={},
    ema=None,
    dropout_p=0.0,
    **kwargs
):
    with tf.variable_scope("GatedResBlock"):
        xs = shape_list(x)
        num_filters = xs[-1]

        c1 = conv(nonlinearity(x), num_filters)
        if a is not None:  # add short-cut connection if auxiliary input 'a' is given
            c1 += nin(nonlinearity(a), num_filters)
        c1 = nonlinearity(c1)
        if dropout_p > 0:
            c1 = tf.nn.dropout(c1, keep_prob=1.0 - dropout_p)
        c2 = conv(c1, num_filters * 2)

        # add projection of h vector if included: conditional generation
        if h is not None:
            with tf.variable_scope(get_name("conditional_weights", counters)):
                hw = get_var_maybe_avg(
                    "hw",
                    ema,
                    shape=[shape_list(h)[-1], 2 * num_filters],
                    dtype=tf.float32,
                    initializer=tf.random_normal_initializer(0, 0.05),
                    trainable=True,
                )
            if init:
                hw = hw.initialized_value()
            c2 += tf.reshape(tf.matmul(h, hw), [xs[0], 1, 1, 2 * num_filters])

        # Is this 3,2 or 2,3 ?
        a, b = tf.split(c2, 2, 3)
        c3 = a * tf.nn.sigmoid(b)
    return x + c3


"""
utilities for shifting the image around, efficient alternative to masking convolutions
"""


def down_shift(x, step=1):
    with tf.variable_scope("down_shift"):
        xs = shape_list(x)
        return tf.concat(
            [tf.zeros([xs[0], step, xs[2], xs[3]]), x[:, : xs[1] - step, :, :]], 1
        )


def right_shift(x, step=1):
    with tf.variable_scope("right_shift"):
        xs = shape_list(x)
        return tf.concat(
            [tf.zeros([xs[0], xs[1], step, xs[3]]), x[:, :, : xs[2] - step, :]], 2
        )


def left_shift(x, step=1):
    with tf.variable_scope("left_shift"):
        xs = shape_list(x)
        return tf.concat([x[:, :, step:, :], tf.zeros([xs[0], xs[1], step, xs[3]])], 2)


@add_arg_scope
def masked_conv2d(x, num_filters, filter_size=[3, 3], stride=[1, 1], **kwargs):
    with tf.variable_scope("masked_conv2d"):
        return masked_conv2d_keras(
            filters=num_filters,
            kernel_size=filter_size,
            strides=stride,
            padding="SAME",
            use_bias=True,
            **kwargs
        )(x)


@add_arg_scope
def base_conv2d(
    x, num_filters, filter_size=[3, 3], stride=[1, 1], **kwargs
):
    with tf.variable_scope("base_conv2d"):
        return conv2d_keras(
            filters=num_filters,
            kernel_size=filter_size,
            padding="same",
            strides=stride,
            norm_type="None",
            self_attention=False,
            **kwargs
        )(x)


@add_arg_scope
def down_shifted_conv2d(x, num_filters, filter_size=[2, 3], stride=[1, 1], **kwargs):
    with tf.variable_scope("down_shifted_conv2d"):
        x = tf.pad(
            x,
            [
                [0, 0],
                [filter_size[0] - 1, 0],
                [int((filter_size[1] - 1) / 2), int((filter_size[1] - 1) / 2)],
                [0, 0],
            ],
        )
        return conv2d_keras(
            filters=num_filters,
            kernel_size=filter_size,
            padding="valid",
            strides=stride,
            norm_type=None,
            self_attention=False,
            **kwargs
        )(x)


@add_arg_scope
def down_shifted_deconv2d(x, num_filters, filter_size=[2, 3], stride=[1, 1], **kwargs):
    with tf.variable_scope("down_shifted_deconv2d"):
        x = deconv2d_keras(
            filters=num_filters,
            kernel_size=filter_size,
            strides=stride,
            # method="conv2d_transpose",
            norm_type=None,
            self_attention=False,
            **kwargs
        )(x)
        xs = shape_list(x)
        tf.logging.info("############ down_shifted_deconv2d(x) : %s ###########" % (xs))
        # return x[
        #     :,
        #     : (xs[1] - filter_size[0] + 1),
        #     int((filter_size[1] - 1) / 2) : (xs[2] - int((filter_size[1] - 1) / 2)),
        #     :,
        # ]
        return x[:, :(xs[1] - filter_size[0] + 1), int((filter_size[1] - 1) / 2):(xs[2] - int((filter_size[1] - 1) / 2)), :]


@add_arg_scope
def down_right_shifted_conv2d(
    x, num_filters, filter_size=[3, 3], stride=[1, 1], **kwargs
):
    with tf.variable_scope("down_right_shifted_conv2d"):
        x = tf.pad(
            x, [[0, 0], [filter_size[0] - 1, 0], [filter_size[1] - 1, 0], [0, 0]]
        )
        return conv2d_keras(
            filters=num_filters,
            kernel_size=filter_size,
            padding="valid",
            strides=stride,
            norm_type=None,
            self_attention=False,
            **kwargs
        )(x)


@add_arg_scope
def down_right_shifted_deconv2d(
    x, num_filters, filter_size=[2, 2], stride=[1, 1], **kwargs
):
    with tf.variable_scope("down_right_shifted_deconv2d"):
        x = deconv2d_keras(
            filters=num_filters,
            kernel_size=filter_size,
            strides=stride,
            # method="conv2d_transpose",
            norm_type=None,
            self_attention=False,
            **kwargs
        )(x)
        xs = shape_list(x)
        return x[:, : (xs[1] - filter_size[0] + 1) :, : (xs[2] - filter_size[1] + 1), :]


def causal_shift_nin(x, num_filters, **kwargs):
    with tf.variable_scope("causal_shift_nin"):
        chns = shape_list(x)[-1]
        assert chns % 4 == 0
        left, upleft, up, upright = tf.split(x, 4, axis=-1)
        return nin(
            tf.concat(
                [
                    right_shift(left),
                    right_shift(down_shift(upleft)),
                    down_shift(up),
                    down_shift(left_shift(upleft)),
                ],
                axis=-1,
            ),
            num_filters,
            **kwargs
        )


from tensorflow.python.framework import function


@add_arg_scope
def mem_saving_causal_shift_nin(x, num_filters, init, counters, **kwargs):
    with tf.variable_scope("mem_saving_causal_shift_nin"):
        if init:
            return causal_shift_nin(
                x, num_filters, init=init, counters=counters, **kwargs
            )

        shps = shape_list(x)

        @function.Defun(tf.float32)
        def go(ix):
            tf.get_variable_scope().reuse_variables()
            ix.set_shape(shps)
            return causal_shift_nin(
                ix, num_filters, init=init, counters=counters, **kwargs
            )

        temp = go(x)
        temp.set_shape([shps[0], shps[1], shps[2], num_filters])
        return temp


import functools


@functools.lru_cache(maxsize=32)
def get_causal_mask(canvas_size, rate=1):
    causal_mask = np.zeros([canvas_size, canvas_size], dtype=np.float32)
    for i in range(canvas_size):
        causal_mask[i, :i] = 1.0
    causal_mask = tf.constant(causal_mask, dtype=tf.float32)

    if rate > 1:
        dim = int(np.sqrt(canvas_size))
        causal_mask = tf.reshape(causal_mask, [canvas_size, dim, dim, 1])
        causal_mask = -tf.nn.max_pool(
            -causal_mask, [1, rate, rate, 1], [1, rate, rate, 1], "SAME"
        )

    causal_mask = tf.reshape(causal_mask, [1, canvas_size, -1])
    return causal_mask


def causal_attention(key, mixin, query, downsample=1, use_pos_enc=False):
    with tf.variable_scope("causal_attention"):
        bs = shape_list_2(key)[0]
        nr_chns = shape_list_2(key)[-1]

        if downsample > 1:
            pool_shape = [1, downsample, downsample, 1]
            key = tf.nn.max_pool(key, pool_shape, pool_shape, "SAME")
            mixin = tf.nn.max_pool(mixin, pool_shape, pool_shape, "SAME")

        xs = shape_list_2(mixin)
        if use_pos_enc:
            pos1 = tf.range(0.0, xs[1]) / xs[1]
            pos2 = tf.range(0.0, xs[2]) / xs[1]
            mixin = tf.concat(
                [
                    mixin,
                    tf.tile(pos1[None, :, None, None], [xs[0], 1, xs[2], 1]),
                    tf.tile(pos2[None, None, :, None], [xs[0], xs[2], 1, 1]),
                ],
                axis=3,
            )

        mixin_chns = shape_list_2(mixin)[-1]
        canvas_size = int(np.prod(shape_list_2(key)[1:-1]))
        canvas_size_q = int(np.prod(shape_list_2(query)[1:-1]))
        causal_mask = get_causal_mask(canvas_size_q, downsample)

        dot = (
            tf.matmul(
                tf.reshape(query, [bs, canvas_size_q, nr_chns]),
                tf.reshape(key, [bs, canvas_size, nr_chns]),
                transpose_b=True,
            )
            - (1.0 - causal_mask) * 1e10
        )

        dot = dot - tf.reduce_max(dot, axis=-1, keep_dims=True)

        causal_exp_dot = tf.exp(dot / np.sqrt(nr_chns).astype(np.float32)) * causal_mask
        causal_probs = causal_exp_dot / (
            tf.reduce_sum(causal_exp_dot, axis=-1, keep_dims=True) + 1e-6
        )

        mixed = tf.matmul(
            causal_probs, tf.reshape(mixin, [bs, canvas_size, mixin_chns])
        )

        return tf.reshape(mixed, shape_list_2(query)[:-1] + [mixin_chns])


def non_cached_get_causal_mask(canvas_size, causal_unit):
    assert causal_unit == 1
    ones = tf.ones([canvas_size, canvas_size], dtype=tf.float32)
    lt = tf.matrix_band_part(ones, -1, 0) - tf.matrix_diag(
        tf.ones([canvas_size], dtype=tf.float32)
    )
    return lt[None, ...]


@add_arg_scope
def mem_saving_causal_attention(_key, _mixin, _query, causal_unit=1):
    with tf.variable_scope("mem_saving_causal_attention"):
        # @function.Defun(tf.float32, tf.float32, tf.float32)
        def go(key, mixin, query):
            key.set_shape(shape_list(_key))
            mixin.set_shape(shape_list(_mixin))
            query.set_shape(shape_list(_query))
            bs, nr_chns = shape_list(key)[0], shape_list(key)[-1]
            mixin_chns = shape_list(mixin)[-1]
            canvas_size = int(np.prod(shape_list(key)[1:-1]))
            causal_mask = non_cached_get_causal_mask(
                canvas_size, causal_unit=causal_unit
            )

            dot = (
                tf.matmul(
                    tf.reshape(query, [bs, canvas_size, nr_chns]),
                    tf.reshape(key, [bs, canvas_size, nr_chns]),
                    transpose_b=True,
                )
                - (1.0 - causal_mask) * 1e10
            )
            dot = dot - tf.reduce_max(dot, axis=-1, keep_dims=True)

            causal_exp_dot = (
                tf.exp(dot / np.sqrt(nr_chns).astype(np.float32)) * causal_mask
            )
            causal_probs = causal_exp_dot / (
                tf.reduce_sum(causal_exp_dot, axis=-1, keep_dims=True) + 1e-6
            )

            mixed = tf.matmul(
                causal_probs, tf.reshape(mixin, [bs, canvas_size, mixin_chns])
            )

            return tf.reshape(mixed, shape_list(mixin))

        temp = go(_key, _mixin, _query)
        temp.set_shape(shape_list(_mixin))
    return temp


def contextual_attention(
    foreground,
    background,
    mask=None,
    ksize=3,
    stride=1,
    rate=1,
    fuse_k=3,
    softmax_scale=10.0,
    fuse=True,
):
    """Contextual attention layer implementation

    Arguments:
        foreground {tf.Tensor} -- input feature to match
        background {tf.Tensor} -- input feature for match

    Keyword Arguments:
        mask {tf.Tensor} -- Input mask for background input, indicating patches not available. (default: {None})
        ksize {int} -- kernel size for contextual attention (default: {3})
        stride {int} -- stride for extracting patches from background (default: {1})
        rate {int} -- dilation for matching. (default: {1})
        fuse_k {int} -- use fused convolutional kernels (default: {3})
        softmax_scale {float} -- scaled softmax for attention (default: {10.})
        fuse {bool} -- use fused operations in computation (default: {True})
    """
    # get shapes of input tensors
    raw_fs = tf.shape(foreground)
    raw_int_fs = foreground.get_shape().as_list()
    raw_int_bs = background.get_shape().as_list()
    background_bs = tf.shape(background)
    # extract patches from background with stride and rate
    kernel = 2 * rate
    raw_w = tf.extract_image_patches(
        background,
        [1, kernel, kernel, 1],
        [1, rate * stride, rate * stride, 1],
        [1, 1, 1, 1],
        padding="SAME",
    )
    tf.logging.info(
        "////ContextualAttention (raw_w) SHAPE:%s, (raw_w_reshape):%s"
        % (raw_w.shape, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
    )
    raw_w = tf.reshape(raw_w, [background_bs, -1, kernel, kernel, raw_int_bs[3]])
    raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # downscaling foreground option: downscaling both foreground and
    # background for matching and use original background for reconstruction.
    foreground = resize(
        foreground, scale=1.0 / rate, func=tf.image.resize_nearest_neighbor
    )
    background = resize(
        background,
        to_shape=[int(raw_int_bs[1] / rate), int(raw_int_bs[2] / rate)],
        func=tf.image.resize_nearest_neighbor,
    )  # https://github.com/tensorflow/tensorflow/issues/11651
    if mask is not None:
        mask = resize(mask, scale=1.0 / rate, func=tf.image.resize_nearest_neighbor)
    fs = tf.shape(foreground)
    int_fs = foreground.get_shape().as_list()
    foreground_bs = tf.shape(foreground)[0]
    f_groups = tf.split(foreground, foreground_bs, axis=0)
    # from t(H*W*C) to w(b*k*k*c*h*w)
    bs = tf.shape(background)
    int_bs = background.get_shape().as_list()
    w = tf.extract_image_patches(
        background,
        [1, ksize, ksize, 1],
        [1, stride, stride, 1],
        [1, 1, 1, 1],
        padding="SAME",
    )
    w = tf.reshape(w, [foreground_bs, -1, ksize, ksize, int_fs[3]])
    w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # process mask
    if mask is None:
        mask = tf.zeros([1, bs[1], bs[2], 1])
    m = tf.extract_image_patches(
        mask, [1, ksize, ksize, 1], [1, stride, stride, 1], [1, 1, 1, 1], padding="SAME"
    )
    m = tf.reshape(m, [1, -1, ksize, ksize, 1])
    m = tf.transpose(m, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    m = m[0]
    mm = tf.cast(
        tf.equal(tf.reduce_mean(m, axis=[0, 1, 2], keep_dims=True), 0.0), tf.float32
    )
    w_groups = tf.split(w, bs[0], axis=0)
    raw_w_groups = tf.split(raw_w, bs[0], axis=0)
    y = []
    offsets = []
    k = fuse_k
    scale = softmax_scale
    fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
    for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
        # conv for compare
        wi = wi[0]
        wi_normed = wi / tf.maximum(
            tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0, 1, 2])), 1e-4
        )
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1, 1, 1, 1], padding="SAME")

        # conv implementation for fuse scores to encourage large patches
        if fuse:
            yi = tf.reshape(yi, [1, fs[1] * fs[2], bs[1] * bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1, 1, 1, 1], padding="SAME")
            yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, fs[1] * fs[2], bs[1] * bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1, 1, 1, 1], padding="SAME")
            yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
        yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1] * bs[2]])

        # softmax to match
        yi *= mm  # mask
        yi = tf.nn.softmax(yi * scale, 3)
        yi *= mm  # mask

        offset = tf.argmax(yi, axis=3, output_type=tf.int32)
        offset = tf.stack([offset // fs[2], offset % fs[2]], axis=-1)
        # deconv for patch pasting
        # 3.1 paste center
        wi_center = raw_wi[0]
        yi = (
            tf.nn.conv2d_transpose(
                yi,
                wi_center,
                tf.concat([[1], raw_fs[1:]], axis=0),
                strides=[1, rate, rate, 1],
            )
            / 4.0
        )
        y.append(yi)
        offsets.append(offset)
    y = tf.concat(y, axis=0)
    y.set_shape(raw_int_fs)
    offsets = tf.concat(offsets, axis=0)
    offsets.set_shape(int_bs[:3] + [2])
    # case1: visualize optical flow: minus current position
    h_add = tf.tile(tf.reshape(tf.range(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
    w_add = tf.tile(tf.reshape(tf.range(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
    offsets = offsets - tf.concat([h_add, w_add], axis=3)
    # to flow image
    flow = flow_to_image_tf(offsets)
    # # case2: visualize which pixels are attended
    # flow = highlight_flow_tf(offsets * tf.cast(mask, tf.int32))
    if rate != 1:
        flow = resize(flow, scale=rate, func=tf.image.resize_bilinear)
    return y, flow


def _base_noup_smallkey_spec(
    x,
    h=None,
    init=False,
    ema=None,
    dropout_p=0.5,
    nr_resnet=2,
    nr_filters=128,
    attn_rep=2,
    output_units=256,
    att_downsample=1,
    resnet_nonlinearity="concat_elu",
):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    with tf.variable_scope("pixelSNAIL_base_noup_smallkey_spec"):
        with arg_scope(
            [gated_resnet],
            counters=counters,
            init=init,
            ema=ema,
            dropout_p=dropout_p,
        ):

            # parse resnet nonlinearity argument
            if resnet_nonlinearity == "concat_elu":
                resnet_nonlinearity = concat_elu
            elif resnet_nonlinearity == "elu":
                resnet_nonlinearity = tf.nn.elu
            elif resnet_nonlinearity == "relu":
                resnet_nonlinearity = tf.nn.relu
            else:
                raise (
                    "resnet nonlinearity " + resnet_nonlinearity + " is not supported"
                )

            with arg_scope([gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

                # ////////// up pass through pixelCNN ////////
                xs = shape_list(x)
                xs_tensor = tf.convert_to_tensor(xs, dtype=tf.float32)
                background = tf.concat(
                    [
                        ((tf.range(xs_tensor[1], dtype=tf.float32) - xs_tensor[1] / 2) / xs_tensor[1])[
                            None, :, None, None
                        ]
                        + 0.0 * x,
                        ((tf.range(xs_tensor[2], dtype=tf.float32) - xs_tensor[2] / 2) / xs_tensor[2])[
                            None, None, :, None
                        ]
                        + 0.0 * x,
                    ],
                    axis=3,
                )
                # add channel of ones to distinguish image from padding later on
                # x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], axis=3)

                # ul_list = [
                #     down_shift(
                #         down_shifted_conv2d(
                #             x_pad, num_filters=nr_filters, filter_size=[1, 3]
                #         )
                #     )
                #     + right_shift(
                #         down_right_shifted_conv2d(
                #             x_pad, num_filters=nr_filters, filter_size=[2, 1]
                #         )
                #     )
                # ]  # stream
                ul_list = [
                    masked_conv2d(x, num_filters=nr_filters, filter_size=[3, 3])
                ]
                # for up and to the left

                for attn_rep in range(attn_rep):
                    for rep in range(nr_resnet):
                        ul_list.append(
                            gated_resnet(ul_list[-1], conv=down_right_shifted_conv2d)
                        )

                    ul = ul_list[-1]
                    raw_content = tf.concat([x, ul, background], axis=3)
                    q_size = 16
                    raw = nin(
                        gated_resnet(raw_content, conv=nin), nr_filters // 2 + q_size
                    )
                    key, mixin = raw[:, :, :, :q_size], raw[:, :, :, q_size:]
                    raw_q = tf.concat([ul, background], axis=3)
                    query = nin(gated_resnet(raw_q, conv=nin), q_size)
                    # tf.logging.info("//// Query SHAPE:%s " % (query.shape))
                    # mixed = tf.keras.layers.Attention(use_scale=False, causal=True)([query, mixin, key])
                    mixed = tf.keras.layers.AdditiveAttention(use_scale=False, causal=True)([query, mixin, key])
                    # tf.logging.info("//// AttentionOutput SHAPE:%s " % (mixed.shape))
                    ul_list.append(gated_resnet(ul, mixed, conv=nin))

                x_out = nin(tf.nn.elu(ul_list[-1]), output_units)

                return x_out


def _context_attn_spec(
    x,
    h=None,
    init=False,
    ema=None,
    dropout_p=0.5,
    nr_resnet=2,
    nr_filters=128,
    attn_rep=4,
    output_units=256,
    att_downsample=1,
    resnet_nonlinearity="concat_elu",
):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    with tf.variable_scope("pixelSNAILadv_context_attn_spec"):
        with arg_scope(
            [gated_resnet, dense, nin],
            counters=counters,
            init=init,
            ema=ema,
            dropout_p=dropout_p,
        ):

            # parse resnet nonlinearity argument
            if resnet_nonlinearity == "concat_elu":
                resnet_nonlinearity = concat_elu
            elif resnet_nonlinearity == "elu":
                resnet_nonlinearity = tf.nn.elu
            elif resnet_nonlinearity == "relu":
                resnet_nonlinearity = tf.nn.relu
            else:
                raise (
                    "resnet nonlinearity " + resnet_nonlinearity + " is not supported"
                )

            with arg_scope([gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

                # ////////// up pass through pixelCNN ////////
                xs = shape_list(x)
                background = tf.concat(
                    [
                        ((tf.range(xs[1], dtype=tf.float32) - xs[1] / 2) / xs[1])[
                            None, :, None, None
                        ]
                        + 0.0 * x,
                        ((tf.range(xs[2], dtype=tf.float32) - xs[2] / 2) / xs[2])[
                            None, None, :, None
                        ]
                        + 0.0 * x,
                    ],
                    axis=3,
                )
                # add channel of ones to distinguish image from padding later on
                x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], axis=3)

                ul_list = [
                    down_shift(
                        down_shifted_conv2d(
                            x_pad, num_filters=nr_filters, filter_size=[1, 3]
                        )
                    )
                    + right_shift(
                        down_right_shifted_conv2d(
                            x_pad, num_filters=nr_filters, filter_size=[2, 1]
                        )
                    )
                ]  # stream
                # for up and to the left

                for attn_rep in range(attn_rep):
                    for rep in range(nr_resnet):
                        ul_list.append(
                            gated_resnet(ul_list[-1], conv=down_right_shifted_conv2d)
                        )

                    ul = ul_list[-1]
                    raw_content = tf.concat([x, ul, background], axis=3)
                    attn_x, offset_flow = contextual_attention(
                        raw_content, raw_content, None, 3, 1, rate=2
                    )
                    ul_list.append(gated_resnet(ul, attn_x, conv=nin))

                x_out = nin(tf.nn.elu(ul_list[-1]), output_units)

                return x_out


def _base_spec(
    x,
    h=None,
    init=False,
    ema=None,
    dropout_p=0.5,
    nr_resnet=1,
    nr_filters=128,
    attn_rep=2,
    output_units=256,
    att_downsample=1,
    resnet_nonlinearity="concat_elu",
):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    with tf.variable_scope("pixelSNAILadv_base_spec_fcn"):
        with arg_scope(
            [gated_resnet], counters=counters, init=init, ema=ema, dropout_p=dropout_p
        ):

            # parse resnet nonlinearity argument
            if resnet_nonlinearity == "concat_elu":
                resnet_nonlinearity = concat_elu
            elif resnet_nonlinearity == "elu":
                resnet_nonlinearity = tf.nn.elu
            elif resnet_nonlinearity == "relu":
                resnet_nonlinearity = tf.nn.relu
            else:
                raise (
                    "resnet nonlinearity " + resnet_nonlinearity + " is not supported"
                )

            with arg_scope([gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

                # ////////// up pass through pixelCNN ////////
                xs = shape_list(x)
                tf.logging.info(
                    "PixelCNNppLayer-/X(input)/NAME:(%s)/SHAPE:(%s)" % (x.name, static_shape_list(x))
                )
                # add channel of ones to distinguish image from padding later on
                # x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], axis=3)

                # ul_list = [
                #     down_shift(
                #         down_shifted_conv2d(
                #             latent_x, num_filters=nr_filters, filter_size=[1, 3]
                #         )
                #     )
                #     + right_shift(
                #         down_right_shifted_conv2d(
                #             latent_x, num_filters=nr_filters, filter_size=[2, 1]
                #         )
                #     )
                # ]  # stream
                ul_list = [
                    masked_conv2d(x, num_filters=nr_filters, filter_size=[3, 3])
                ]  # stream
                # for up and to the left
                tf.logging.info(
                    ">>>> PixelCNNppLayer-/U(masked_conv2d)/NAME:(%s)/SHAPE:(%s)"
                    % (ul_list[-1].name, static_shape_list(ul_list[-1]))
                )

                for attn_rep in range(attn_rep):
                    for rep in range(nr_resnet):
                        ul_list.append(
                            gated_resnet(ul_list[-1], conv=base_conv2d)
                        )

                    ul = ul_list[-1]
                    raw_content = tf.concat([x, ul], axis=3)
                    ul_list.append(gated_resnet(raw_content, conv=nin))

                # x_out = base_conv2d(
                #     tf.nn.elu(ul_list[-1]),
                #     output_units,
                #     filter_size=[3, 3],
                #     stride=[1, 1],
                #     apply_pad=False,
                # )
                x_out = nin(tf.nn.elu(ul_list[-1]), output_units)

                tf.logging.info(
                    "PixelCNNppLayer-/X_out/NAME:(%s)/SHAPE:(%s)"
                    % (x_out.name, static_shape_list(x_out))
                )

                return x_out


def pxpp_spec(
    x,
    h=None,
    init=False,
    ema=None,
    dropout_p=0.5,
    nr_resnet=1,
    nr_filters=128,
    output_units=256,
    resnet_nonlinearity="concat_elu",
):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    with arg_scope(
        [conv2d, deconv2d, gated_resnet, dense],
        counters=counters,
        init=init,
        ema=ema,
        dropout_p=dropout_p,
    ):

        # parse resnet nonlinearity argument
        if resnet_nonlinearity == "concat_elu":
            resnet_nonlinearity = concat_elu
        elif resnet_nonlinearity == "elu":
            resnet_nonlinearity = tf.nn.elu
        elif resnet_nonlinearity == "relu":
            resnet_nonlinearity = tf.nn.relu
        else:
            raise ("resnet nonlinearity " + resnet_nonlinearity + " is not supported")

        with arg_scope([gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

            # ////////// up pass through pixelCNN ////////
            xs = shape_list(x)
            tf.logging.info(
                "PixelCNNppLayer-/X(input)/NAME:(%s)/SHAPE:(%s)" % (x.name, x.shape)
            )
            # add channel of ones to distinguish image from padding later on
            x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], 3)
            tf.logging.info(
                "PixelCNNppLayer-/X_pad/NAME:(%s)/SHAPE:(%s)"
                % (x_pad.name, x_pad.shape)
            )
            u_list = [
                down_shift(
                    down_shifted_conv2d(
                        x_pad, num_filters=nr_filters, filter_size=[2, 3]
                    )
                )
            ]  # stream for pixels above
            ul_list = [
                down_shift(
                    down_shifted_conv2d(
                        x_pad, num_filters=nr_filters, filter_size=[1, 3]
                    )
                )
                + right_shift(
                    down_right_shifted_conv2d(
                        x_pad, num_filters=nr_filters, filter_size=[2, 1]
                    )
                )
            ]  # stream for up and to the left

            for rep in range(nr_resnet):
                u_list.append(gated_resnet(u_list[-1], conv=down_shifted_conv2d))
                ul_list.append(
                    gated_resnet(
                        ul_list[-1], u_list[-1], conv=down_right_shifted_conv2d
                    )
                )

            u_list.append(
                down_shifted_conv2d(u_list[-1], num_filters=nr_filters, stride=[2, 2])
            )
            ul_list.append(
                down_right_shifted_conv2d(
                    ul_list[-1], num_filters=nr_filters, stride=[2, 2]
                )
            )

            for rep in range(nr_resnet):
                u_list.append(gated_resnet(u_list[-1], conv=down_shifted_conv2d))
                ul_list.append(
                    gated_resnet(
                        ul_list[-1], u_list[-1], conv=down_right_shifted_conv2d
                    )
                )

            u_list.append(
                down_shifted_conv2d(u_list[-1], num_filters=nr_filters, stride=[2, 2])
            )
            ul_list.append(
                down_right_shifted_conv2d(
                    ul_list[-1], num_filters=nr_filters, stride=[2, 2]
                )
            )

            for rep in range(nr_resnet):
                u_list.append(gated_resnet(u_list[-1], conv=down_shifted_conv2d))
                ul_list.append(
                    gated_resnet(
                        ul_list[-1], u_list[-1], conv=down_right_shifted_conv2d
                    )
                )

            # /////// down pass ////////
            u = u_list.pop()
            tf.logging.info(
                "PixelCNNppLayer-/U(downward_pass)/NAME:(%s)/SHAPE:(%s)"
                % (u.name, u.shape)
            )
            ul = ul_list.pop()
            tf.logging.info(
                "PixelCNNppLayer-/UL(downward_pass)/NAME:(%s)/SHAPE:(%s)"
                % (ul.name, ul.shape)
            )
            for rep in range(nr_resnet):
                u_pop = u_list.pop()
                tf.logging.info(
                    "PixelCNNppLayer-/U_LIST(pop)(downward_pass)/NAME:(%s)/SHAPE:(%s)"
                    % (u_pop.name, u_pop.shape)
                )
                tf.logging.info(
                    "PixelCNNppLayer-/U(downward_pass)/NAME:(%s)/SHAPE:(%s)"
                    % (u.name, u.shape)
                )
                u = gated_resnet(u, u_pop, conv=down_shifted_conv2d)

                ul_pop = ul_list.pop()

                tf.logging.info(
                    "PixelCNNppLayer-/UL_LIST(pop)(downward_pass)/NAME:(%s)/SHAPE:(%s)"
                    % (ul_pop.name, ul_pop.shape)
                )
                tf.logging.info(
                    "PixelCNNppLayer-/UL(downward_pass)/NAME:(%s)/SHAPE:(%s)"
                    % (ul.name, ul.shape)
                )
                ul = gated_resnet(
                    ul, tf.concat([u, ul_pop], 3), conv=down_right_shifted_conv2d
                )


            u = down_shifted_deconv2d(u, num_filters=nr_filters, stride=[2, 2])
            tf.logging.info(
                ">>>> PixelCNNppLayer-/U(deconv2d)/NAME:(%s)/SHAPE:(%s)"
                % (u.name, u.shape)
            )
            ul = down_right_shifted_deconv2d(ul, num_filters=nr_filters, stride=[2, 2])
            tf.logging.info(
                ">>>> PixelCNNppLayer-/UL(deconv2d)/NAME:(%s)/SHAPE:(%s)"
                % (ul.name, ul.shape)
            )

            for rep in range(nr_resnet + 1):
                u_pop = u_list.pop()
                tf.logging.info(
                    "PixelCNNppLayer-/U_LIST(pop)(downward_pass)/NAME:(%s)/SHAPE:(%s)"
                    % (u_pop.name, u_pop.shape)
                )
                tf.logging.info(
                    "PixelCNNppLayer-/U(downward_pass)/NAME:(%s)/SHAPE:(%s)"
                    % (u.name, u.shape)
                )
                u = gated_resnet(u, u_pop, conv=down_shifted_conv2d)

                ul_pop = ul_list.pop()

                tf.logging.info(
                    "PixelCNNppLayer-/UL_LIST(pop)(downward_pass)/NAME:(%s)/SHAPE:(%s)"
                    % (ul_pop.name, ul_pop.shape)
                )
                tf.logging.info(
                    "PixelCNNppLayer-/UL(downward_pass)/NAME:(%s)/SHAPE:(%s)"
                    % (ul.name, ul.shape)
                )
                ul = gated_resnet(
                    ul, tf.concat([u, ul_pop], 3), conv=down_right_shifted_conv2d
                )

            u = down_shifted_deconv2d(u, num_filters=nr_filters, stride=[2, 2])
            tf.logging.info(
                ">>>> PixelCNNppLayer-/U(deconv2d_2)/NAME:(%s)/SHAPE:(%s)"
                % (u.name, u.shape)
            )
            ul = down_right_shifted_deconv2d(ul, num_filters=nr_filters, stride=[2, 2])
            tf.logging.info(
                ">>>> PixelCNNppLayer-/UL(deconv2d_2)/NAME:(%s)/SHAPE:(%s)"
                % (ul.name, ul.shape)
            )

            for rep in range(nr_resnet + 1):
                u_pop = u_list.pop()
                tf.logging.info(
                    "PixelCNNppLayer-/U_LIST(pop)(downward_pass)/NAME:(%s)/SHAPE:(%s)"
                    % (u_pop.name, u_pop.shape)
                )
                tf.logging.info(
                    "PixelCNNppLayer-/U(downward_pass)/NAME:(%s)/SHAPE:(%s)"
                    % (u.name, u.shape)
                )
                u = gated_resnet(u, u_pop, conv=down_shifted_conv2d)

                ul_pop = ul_list.pop()

                tf.logging.info(
                    "PixelCNNppLayer-/UL_LIST(pop)(downward_pass)/NAME:(%s)/SHAPE:(%s)"
                    % (ul_pop.name, ul_pop.shape)
                )
                tf.logging.info(
                    "PixelCNNppLayer-/UL(downward_pass)/NAME:(%s)/SHAPE:(%s)"
                    % (ul.name, ul.shape)
                )
                ul = gated_resnet(
                    ul, tf.concat([u, ul_pop], 3), conv=down_right_shifted_conv2d
                )

            x_out = nin(tf.nn.elu(ul), output_units)
            tf.logging.info(
                "PixelCNNppLayer-/X_out/NAME:(%s)/SHAPE:(%s)"
                % (x_out.name, x_out.shape)
            )

            assert len(u_list) == 0
            assert len(ul_list) == 0

            return x_out
