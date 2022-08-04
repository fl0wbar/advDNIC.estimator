from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.eager import context


def build_learning_rate(
    initial_lr,
    global_step,
    steps_per_epoch=None,
    lr_decay_type="cosine_restarts",
    decay_factor=0.97,
    decay_epochs=2.4,
    total_steps=None,
    warmup_epochs=5,
):
    """Build learning rate."""
    if lr_decay_type == "exponential":
        assert steps_per_epoch is not None
        decay_steps = steps_per_epoch * decay_epochs
        lr = tf.train.exponential_decay(
            initial_lr, global_step, decay_steps, decay_factor, staircase=True
        )
    elif lr_decay_type == "cosine":
        assert total_steps is not None
        lr = (
            0.5
            * initial_lr
            * (1 + tf.cos(np.pi * tf.cast(global_step, tf.float32) / total_steps))
        )
    elif lr_decay_type == "cosine_restarts":
        lr = tf.train.cosine_decay_restarts(
            initial_lr, global_step, first_decay_steps=1000, t_mul=5.0, m_mul=0.75
        )
    elif lr_decay_type == "linear_cosine":
        lr = tf.train.linear_cosine_decay(initial_lr, global_step, decay_steps=10)
    elif lr_decay_type == "noisy_linear_cosine":
        lr = tf.train.noisy_linear_cosine_decay(initial_lr, global_step, decay_steps=10)
    elif lr_decay_type == "constant":
        lr = initial_lr
    else:
        assert False, "Unknown lr_decay_type : %s" % lr_decay_type

    if warmup_epochs:
        tf.logging.info("Learning rate warmup_epochs: %d" % warmup_epochs)
        warmup_steps = int(warmup_epochs * steps_per_epoch)
        warmup_lr = (
            initial_lr
            * tf.cast(global_step, tf.float32)
            / tf.cast(warmup_steps, tf.float32)
        )
        lr = tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)

    return lr


def cyclic_learning_rate(
    global_step,
    learning_rate=0.01,
    max_lr=0.1,
    step_size=20.0,
    gamma=0.99994,
    mode="triangular",
    name=None,
):
    """Applies cyclic learning rate (CLR).
     From the paper:
     Smith, Leslie N. "Cyclical learning
     rates for training neural networks." 2017.
     [https://arxiv.org/pdf/1506.01186.pdf]
      This method lets the learning rate cyclically
     vary between reasonable boundary values
     achieving improved classification accuracy and
     often in fewer iterations.
      This code varies the learning rate linearly between the
     minimum (learning_rate) and the maximum (max_lr).
      It returns the cyclic learning rate. It is computed as:
       ```python
       cycle = floor( 1 + global_step /
        ( 2 * step_size ) )
      x = abs( global_step / step_size – 2 * cycle + 1 )
      clr = learning_rate +
        ( max_lr – learning_rate ) * max( 0 , 1 - x )
       ```
      Polices:
        'triangular':
          Default, linearly increasing then linearly decreasing the
          learning rate at each cycle.
         'triangular2':
          The same as the triangular policy except the learning
          rate difference is cut in half at the end of each cycle.
          This means the learning rate difference drops after each cycle.
         'exp_range':
          The learning rate varies between the minimum and maximum
          boundaries and each boundary value declines by an exponential
          factor of: gamma^global_step.
       Example: 'triangular2' mode cyclic learning rate.
        '''python
        ...
        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=
          clr.cyclic_learning_rate(global_step=global_step, mode='triangular2'))
        train_op = optimizer.minimize(loss_op, global_step=global_step)
        ...
         with tf.Session() as sess:
            sess.run(init)
            for step in range(1, num_steps+1):
              assign_op = global_step.assign(step)
              sess.run(assign_op)
        ...
         '''
       Args:
        global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
          `global step` to use for the cyclic computation.  Must not be negative.
        learning_rate: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The initial learning rate which is the lower bound
          of the cycle (default = 0.1).
        max_lr:  A scalar. The maximum learning rate boundary.
        step_size: A scalar. The number of iterations in half a cycle.
          The paper suggests step_size = 2-8 x training iterations in epoch.
        gamma: constant in 'exp_range' mode:
          gamma**(global_step)
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
        name: String.  Optional name of the operation.  Defaults to
          'CyclicLearningRate'.
       Returns:
        A scalar `Tensor` of the same type as `learning_rate`.  The cyclic
        learning rate.
      Raises:
        ValueError: if `global_step` is not supplied.
      @compatibility(eager)
      When eager execution is enabled, this function returns
      a function which in turn returns the decayed learning
      rate Tensor. This can be useful for changing the learning
      rate value across different invocations of optimizer functions.
      @end_compatibility
  """
    if global_step is None:
        raise ValueError("global_step required for cyclic_learning_rate.")
    with ops.name_scope(
        name, "CyclicLearningRate", [learning_rate, global_step]
    ) as name:
        learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
        dtype = learning_rate.dtype
        global_step = math_ops.cast(global_step, dtype)
        step_size = math_ops.cast(step_size, dtype)

        def cyclic_lr():
            """Helper to recompute learning rate; most helpful in eager-mode."""
            # computing: cycle = floor( 1 + global_step / ( 2 * step_size ) )
            double_step = math_ops.multiply(2.0, step_size)
            global_div_double_step = math_ops.divide(global_step, double_step)
            cycle = math_ops.floor(math_ops.add(1.0, global_div_double_step))
            # computing: x = abs( global_step / step_size – 2 * cycle + 1 )
            double_cycle = math_ops.multiply(2.0, cycle)
            global_div_step = math_ops.divide(global_step, step_size)
            tmp = math_ops.subtract(global_div_step, double_cycle)
            x = math_ops.abs(math_ops.add(1.0, tmp))
            # computing: clr = learning_rate + ( max_lr – learning_rate ) * max( 0, 1 - x )
            a1 = math_ops.maximum(0.0, math_ops.subtract(1.0, x))
            a2 = math_ops.subtract(max_lr, learning_rate)
            clr = math_ops.multiply(a1, a2)
            if mode == "triangular2":
                clr = math_ops.divide(
                    clr,
                    math_ops.cast(
                        math_ops.pow(2, math_ops.cast(cycle - 1, tf.int32)), tf.float32
                    ),
                )
            if mode == "exp_range":
                clr = math_ops.multiply(math_ops.pow(gamma, global_step), clr)
            return math_ops.add(clr, learning_rate, name=name)

        if not context.executing_eagerly():
            cyclic_lr = cyclic_lr()
        return cyclic_lr
