from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf


class LAMBOptimizer(tf.train.Optimizer):
    """
    LAMBOptimizer optimizer.
    https://github.com/ymcui/LAMB_Optimizer_TF
    # IMPORTANT NOTE
    - This is NOT an official implementation.
    - LAMB optimizer is changed from arXiv v1 ~ v3.
    - We implement v3 version (which is the latest version on June, 2019.).
    - Our implementation is based on `AdamWeightDecayOptimizer` in BERT (provided by Google).

    # References
    - Large Batch Optimization for Deep Learning: Training BERT in 76 minutes. https://arxiv.org/abs/1904.00962v3
    - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. https://arxiv.org/abs/1810.04805
    # Parameters
    - There is nothing special, just the same as `AdamWeightDecayOptimizer`.
    """

    def __init__(
        self,
        learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=None,
        name="LAMBOptimizer",
    ):
        """Constructs a LAMBOptimizer."""
        super(LAMBOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/lamb_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer(),
            )
            v = tf.get_variable(
                name=param_name + "/lamb_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer(),
            )

            # Standard Adam update.
            next_m = tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad)
            next_v = tf.multiply(self.beta_2, v) + tf.multiply(
                1.0 - self.beta_2, tf.square(grad)
            )

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            ############## BELOW ARE THE SPECIFIC PARTS FOR LAMB ##############

            # Note: Here are two choices for scaling function \phi(z)
            # minmax:   \phi(z) = min(max(z, \gamma_l), \gamma_u)
            # identity: \phi(z) = z
            # The authors does not mention what is \gamma_l and \gamma_u
            # UPDATE: after asking authors, they provide me the code below.
            # ratio = array_ops.where(math_ops.greater(w_norm, 0), array_ops.where(
            #      math_ops.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

            r1 = tf.sqrt(tf.reduce_sum(tf.square(param)))
            r2 = tf.sqrt(tf.reduce_sum(tf.square(update)))

            r = tf.where(
                tf.greater(r1, 0.), tf.where(tf.greater(r2, 0.), r1 / r2, 1.0), 1.0
            )

            eta = self.learning_rate * r

            update_with_lr = eta * update

            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param), m.assign(next_m), v.assign(next_v)]
            )
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name
