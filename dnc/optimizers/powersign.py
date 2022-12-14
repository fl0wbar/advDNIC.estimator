from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


class PowerSignOptimizer(optimizer.Optimizer):
    """
        Optimizer that implements the PowerSign update.
        See [Bello et al., ICML2017],
        [Neural Optimizer Search with RL](https://arxiv.org/abs/1709.07417).
        Used best with linear_cosine_decay/noisy_linear_cosine_decay lr scheme
    """

    def __init__(
        self,
        learning_rate=0.1,
        base=math.e,
        beta=0.9,
        sign_decay_fn=None,
        use_locking=False,
        name="PowerSignOptimizer",
    ):
        """Constructs a new PowerSignOptimizer object.
        Initialization:
        ```
        m_0 <- 0 (Initialize initial 1st moment vector)
        t <- 0 (Initialize timestep)
        ```
        Update:
        ```
        t <- t + 1
        m_t <- beta1 * m_{t-1} + (1 - beta1) * g
        sign_decay <- sign_decay_fn(t)
        update <- base ** (sign_decay * sign(g) * sign(m)) * g
        variable <- variable - lr_t * update
        ```
        Example usage for PowerSign-cd (PowerSign with cosine sign decay)
        ```
        decay_steps = 1000
        linear_decay_fn = sign_decays.get_cosine_decay_fn(decay_steps)
        opt = PowerSignOptimizer(learning_rate=0.1, sign_decay_fn=linear_decay_fn)
        ```
        Args:
          learning_rate: learning_rate used when taking a step.
          base: base used in optimizer.
          beta: decay used for computing the moving average m.
          sign_decay_fn: decay function applied to the sign(g) sign(m) quantity.
              Takes global_step as an argument. See sign_decay.py for some examples.
          use_locking: If True, use locks for update operations.
          name: Optional name for the operations created iwhen applying gradients.
            Defaults to "PowerSignOptimizer".
        """
        super(PowerSignOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta = beta
        self._logbase = math.log(base)

        self._sign_decay_fn = sign_decay_fn

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta_t = None
        self._logbase_t = None

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        if self._sign_decay_fn is not None:
            self._sign_decay_t = ops.convert_to_tensor(
                self._sign_decay_fn(global_step), name="sign_decay"
            )
        return super(PowerSignOptimizer, self).apply_gradients(
            grads_and_vars, global_step=global_step, name=name
        )

    def _create_slots(self, var_list):
        # Create slots for the first moment.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta_t = ops.convert_to_tensor(self._beta, name="beta")
        self._logbase_t = ops.convert_to_tensor(self._logbase, name="logbase")
        if self._sign_decay_fn is None:
            self._sign_decay_t = ops.convert_to_tensor(1.0, name="sign_decay")

    def _apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        return training_ops.apply_power_sign(
            var,
            m,
            math_ops.cast(self._lr_t, var.dtype.base_dtype),
            math_ops.cast(self._logbase_t, var.dtype.base_dtype),
            math_ops.cast(self._sign_decay_t, var.dtype.base_dtype),
            math_ops.cast(self._beta_t, var.dtype.base_dtype),
            grad,
            use_locking=self._use_locking,
        ).op

    def _resource_apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        return training_ops.resource_apply_power_sign(
            var.handle,
            m.handle,
            math_ops.cast(self._lr_t, var.dtype.base_dtype),
            math_ops.cast(self._logbase_t, var.dtype.base_dtype),
            math_ops.cast(self._sign_decay_t, var.dtype.base_dtype),
            math_ops.cast(self._beta_t, var.dtype.base_dtype),
            grad,
            use_locking=self._use_locking,
        )

    def _apply_sparse(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)
        logbase_t = math_ops.cast(self._logbase_t, var.dtype.base_dtype)
        e_t = math_ops.cast(math.e, var.dtype.base_dtype)

        m = self.get_slot(var, "m")
        m_t = state_ops.assign(
            m, (m * beta_t) + (grad * (1 - beta_t)), use_locking=self._use_locking
        )

        sign_g = ops.IndexedSlices(
            math_ops.sign(grad.values), grad.indices, dense_shape=grad.dense_shape
        )
        sign_gm = ops.IndexedSlices(
            array_ops.gather(math_ops.sign(m_t), sign_g.indices) * sign_g.values,
            sign_g.indices,
            dense_shape=sign_g.dense_shape,
        )

        sign_decayed = math_ops.cast(self._sign_decay_t, var.dtype.base_dtype)
        multiplier_values = math_ops.pow(e_t, logbase_t * sign_decayed * sign_gm.values)
        multiplier = ops.IndexedSlices(
            multiplier_values, sign_gm.indices, dense_shape=sign_gm.dense_shape
        )

        final_update = ops.IndexedSlices(
            lr_t * multiplier.values * grad.values,
            multiplier.indices,
            dense_shape=multiplier.dense_shape,
        )

        var_update = state_ops.scatter_sub(
            var,
            final_update.indices,
            final_update.values,
            use_locking=self._use_locking,
        )

        return control_flow_ops.group(*[var_update, m_t])
