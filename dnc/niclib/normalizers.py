from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_compression as tfc

from . import subtransforms


class WeightNormalization(tf.keras.layers.Wrapper):
    """
        This wrapper reparameterizes a layer by decoupling the weight's
        magnitude and direction.
        This speeds up convergence by improving the
        conditioning of the optimization problem.
        Weight Normalization: A Simple Reparameterization to Accelerate
        Training of Deep Neural Networks: https://arxiv.org/abs/1602.07868
        Tim Salimans, Diederik P. Kingma (2016)
        WeightNormalization wrapper works for keras and tf layers.
        ```python
          net = WeightNormalization(
              tf.keras.layers.Conv2D(2, 2, activation='relu'),
              input_shape=(32, 32, 3),
              data_init=True)(x)
          net = WeightNormalization(
              tf.keras.layers.Conv2D(16, 5, activation='relu'),
              data_init=True)(net)
          net = WeightNormalization(
              tf.keras.layers.Dense(120, activation='relu'),
              data_init=True)(net)
          net = WeightNormalization(
              tf.keras.layers.Dense(n_classes),
              data_init=True)(net)
        ```
        Arguments:
          layer: a layer instance.
          data_init: If `True` use data dependent variable initialization
        Raises:
          ValueError: If not initialized with a `Layer` instance.
          ValueError: If `Layer` does not contain a `kernel` of weights
          NotImplementedError: If `data_init` is True and running graph execution
    """

    def __init__(self, layer, data_init=False, **kwargs):
        super(WeightNormalization, self).__init__(layer, **kwargs)
        self.data_init = data_init
        self._initialized = False
        self._track_trackable(layer, name="layer")

    def build(self, input_shape):
        """Build `Layer`"""
        input_shape = tf.TensorShape(input_shape).as_list()
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)

        if not self.layer.built:
            self.layer.build(input_shape)

            if not hasattr(self.layer, "kernel"):
                raise ValueError(
                    "`WeightNormalization` must wrap a layer that"
                    " contains a `kernel` for weights"
                )

            # The kernel's filter or unit dimension is -1
            self.layer_depth = int(self.layer.kernel.shape[-1])
            self.kernel_norm_axes = list(range(self.layer.kernel.shape.rank - 1))

            self.v = self.layer.kernel
            self.g = self.add_variable(
                name="g",
                shape=(self.layer_depth,),
                initializer=tf.keras.initializers.get("ones"),
                dtype=self.layer.kernel.dtype,
                trainable=True,
            )

        super(WeightNormalization, self).build()

    def call(self, inputs):
        """Call `Layer`"""
        if not self._initialized:
            self._initialize_weights(inputs)

        self._compute_weights()  # Recompute weights for each forward pass
        output = self.layer(inputs)
        return output

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())

    def _compute_weights(self):
        """Generate normalized weights.
        This method will update the value of self.layer.kernel with the
        normalized value, so that the layer is ready for call().
        """
        with tf.name_scope("compute_weights"):
            self.layer.kernel = (
                tf.nn.l2_normalize(self.v, axis=self.kernel_norm_axes) * self.g
            )

    def _initialize_weights(self, inputs):
        """Initialize weight g.
        The initial value of g could either from the initial value in v,
        or by the input value if self.data_init is True.
        """
        if self.data_init:
            self._data_dep_init(inputs)
        else:
            self._init_norm()
        self._initialized = True

    def _init_norm(self):
        """Set the weight g with the norm of the weight vector."""
        with tf.name_scope("init_norm"):
            flat = tf.reshape(self.v, [-1, self.layer_depth])
            self.g.assign(tf.reshape(tf.linalg.norm(flat, axis=0), (self.layer_depth,)))

    def _data_dep_init(self, inputs):
        """Data dependent initialization."""

        with tf.name_scope("data_dep_init"):
            # Generate data dependent init values
            existing_activation = self.layer.activation
            self.layer.activation = None
            x_init = self.layer(inputs)
            data_norm_axes = list(range(x_init.shape.rank - 1))
            m_init, v_init = tf.nn.moments(x_init, data_norm_axes)
            scale_init = 1.0 / tf.math.sqrt(v_init + 1e-10)

        # Assign data dependent init values
        self.g = self.g * scale_init
        if hasattr(self.layer, "bias"):
            self.layer.bias = -m_init * scale_init
        self.layer.activation = existing_activation

    def get_config(self):
        config = {"data_init": self.data_init}
        base_config = super(WeightNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def _l2normalize(v, eps=1e-12):
    """l2 normize the input vector."""
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_norm(weights, num_iters=1, update_collection=None, with_sigma=False):
    """
        Performs Spectral Normalization on a weight tensor.
        Specifically it divides the weight tensor by its largest singular value. This
        is intended to stabilize GAN training, by making the discriminator satisfy a
        local 1-Lipschitz constraint.
        Based on [Spectral Normalization for Generative Adversarial Networks][sn-gan]
        [sn-gan] https://openreview.net/pdf?id=B1QRgziT-
        Args:
            weights: The weight tensor which requires spectral normalization
            num_iters: Number of SN iterations.
            update_collection: The update collection for assigning persisted variable u.
                               If None, the function will update u during the forward
                               pass. Else if the update_collection equals 'NO_OPS', the
                               function will not update the u during the forward. This
                               is useful for the discriminator, since it does not update
                               u in the second pass.
                               Else, it will put the assignment in a collection
                               defined by the user. Then the user need to run the
                               assignment explicitly.
            with_sigma: For debugging purpose. If True, the function returns
                        the estimated singular value for the weight tensor.
        Returns:
            w_bar: The normalized weight tensor
            sigma: The estimated singular value for the weight tensor.
      """
    w_shape = weights.shape.as_list()
    w_mat = tf.reshape(weights, [-1, w_shape[-1]])  # [-1, output_channel]
    u = tf.get_variable(
        "u",
        [1, w_shape[-1]],
        initializer=tf.truncated_normal_initializer(),
        trainable=False,
    )
    u_ = u
    for _ in range(num_iters):
        v_ = _l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
        u_ = _l2normalize(tf.matmul(v_, w_mat))

    sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
    w_mat /= sigma
    if update_collection is None:
        with tf.control_dependencies([u.assign(u_)]):
            w_bar = tf.reshape(w_mat, w_shape)
    else:
        w_bar = tf.reshape(w_mat, w_shape)
        if update_collection != "NO_OPS":
            tf.add_to_collection(update_collection, u.assign(u_))
    if with_sigma:
        return w_bar, sigma
    else:
        return w_bar


class SpectralNormalization(tf.keras.layers.Wrapper):
    """
        This wrapper reparametererizes a layer's weights by
        applying Spectral Normalization on the weight tensor.

        Specifically it divides the weight tensor by its largest singular value. This
        is intended to stabilize GAN training, by making the discriminator satisfy a
        local 1-Lipschitz constraint.
        Based on [Spectral Normalization for Generative Adversarial Networks][sn-gan].
        [sn-gan]: https://openreview.net/forum?id=B1QRgziT-
        To reproduce an SN-GAN, apply this wrapper to every weight tensor of
        your discriminator. The last dimension of the weight tensor must be the number
        of output channels.

        SpectralNormalization wrapper works for keras and tf.layers.

        Arguments:
            layer: a layer instance.
        Raises:
            `Layer` instance with spectral normalized weights
            'sigma` The estimated singular value for the weight tensor.(for debugging)
            ValueError: If not initialized with a `Layer` instance.
            ValueError: If `Layer` does not contain a `kernel` of weights
    """

    def __init__(self, layer, npi: int = 1, update_collection=None, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)
        self._n_power_iterations = npi
        self._update_collection = update_collection
        self._track_trackable(layer, name="layer")

    def build(self, input_shape):
        """ Build `Layer` """
        input_shape = tf.TensorShape(input_shape).as_list()
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)

        if not self.layer.built:
            self.layer.build(input_shape)

            if not hasattr(self.layer, "kernel"):
                raise ValueError(
                    "`SpectralNormalization` must wrap a layer that"
                    " contains a `kernel` for weights"
                )

            self.w = self.layer.kernel
            self.w_shape = self.w.shape.as_list()
            self.u = self.add_variable(
                shape=tuple([1, self.w_shape[-1]]),
                initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                name="sn_u",
                trainable=False,
                dtype=tf.float32,
            )

        super(SpectralNormalization, self).build()

    def call(self, inputs):
        """Call `Layer`"""
        # Recompute weights for each forward pass
        self._compute_weights()
        output = self.layer(inputs)
        return output

    @staticmethod
    def _l2normalize(tensor, eps=1e-12):
        """l2 normalize the input vector."""
        return tensor / (tf.reduce_sum(tensor ** 2) ** 0.5 + eps)

    def _compute_weights(self):
        """
        Generate normalized weights.
        This method will update the value of self.layer.kernel with the
        normalized value, so that the layer is ready for call().
        """
        with tf.name_scope("sn_compute_weights"):
            w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
            _u = tf.identity(self.u)
            for _ in range(self._n_power_iterations):
                _v = self._l2normalize(tf.matmul(_u, w_reshaped, transpose_b=True))
                _u = self._l2normalize(tf.matmul(_v, w_reshaped))

            self.sigma = tf.squeeze(
                tf.matmul(tf.matmul(_v, w_reshaped), _u, transpose_b=True)
            )
            w_reshaped /= self.sigma
            if self._update_collection is None:
                with tf.control_dependencies([self.u.assign(_u)]):
                    self.layer.kernel = tf.reshape(w_reshaped, self.w_shape)
            else:
                self.layer.kernel = tf.reshape(w_reshaped, self.w_shape)
                if self._update_collection != "NO_OPS":
                    tf.add_to_collection(self._update_collection, self.u.assign(_u))

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())


class GroupNormalization(tf.keras.layers.Layer):
    """Group normalization layer.
    Group Normalization divides the channels into groups and computes
    within each group the mean and variance for normalization.
    Empirically, its accuracy is more stable than batch norm in a wide
    range of small batch sizes, if learning rate is adjusted linearly
    with batch sizes.
    Relation to Layer Normalization:
    If the number of groups is set to 1, then this operation becomes identical
    to Layer Normalization.
    Relation to Instance Normalization:
    If the number of groups is set to the
    input dimension (number of groups is equal
    to number of channels), then this operation becomes
    identical to Instance Normalization.
    Arguments
        groups: Integer, the number of groups for Group Normalization.
            Can be in the range [1, N] where N is the input dimension.
            The input dimension must be divisible by the number of groups.
        axis: Integer, the axis that should be normalized.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    Output shape
        Same shape as input.
    References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    def __init__(
        self,
        groups=2,
        axis=-1,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs
    ):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self._check_axis()

    def build(self, input_shape):

        self._check_if_input_shape_is_none(input_shape)
        self._set_number_of_groups_for_instance_norm(input_shape)
        self._check_size_of_dimensions(input_shape)
        self._create_input_spec(input_shape)

        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)
        self.built = True
        super(GroupNormalization, self).build(input_shape)

    def call(self, inputs):

        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.shape(inputs)

        reshaped_inputs, group_shape = self._reshape_into_groups(
            inputs, input_shape, tensor_input_shape
        )

        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)

        outputs = tf.reshape(normalized_inputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(
                self.gamma_initializer
            ),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(
                self.gamma_regularizer
            ),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):

        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        group_shape[self.axis] = input_shape[self.axis] // self.groups
        group_shape.insert(1, self.groups)
        group_shape = tf.stack(group_shape)
        reshaped_inputs = tf.reshape(inputs, group_shape)
        return reshaped_inputs, group_shape

    def _apply_normalization(self, reshaped_inputs, input_shape):

        group_shape = tf.keras.backend.int_shape(reshaped_inputs)
        group_reduction_axes = list(range(len(group_shape)))
        # Remember the ordering of the tensor is [batch, group , steps]. Jump
        # the first 2 to calculate the variance and the mean
        mean, variance = tf.nn.moments(
            reshaped_inputs, group_reduction_axes[2:], keepdims=True
        )

        gamma, beta = self._get_reshaped_weights(input_shape)
        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon,
        )
        return normalized_inputs

    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)

        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return gamma, beta

    def _check_if_input_shape_is_none(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(
                "Axis " + str(self.axis) + " of "
                "input tensor should have a defined dimension "
                "but the layer received an input with shape " + str(input_shape) + "."
            )

    def _set_number_of_groups_for_instance_norm(self, input_shape):
        dim = input_shape[self.axis]

        if self.groups == -1:
            self.groups = dim

    def _check_size_of_dimensions(self, input_shape):

        dim = input_shape[self.axis]
        if dim < self.groups:
            raise ValueError(
                "Number of groups (" + str(self.groups) + ") cannot be "
                "more than the number of channels (" + str(dim) + ")."
            )

        if dim % self.groups != 0:
            raise ValueError(
                "Number of groups (" + str(self.groups) + ") must be a "
                "multiple of the number of channels (" + str(dim) + ")."
            )

    def _check_axis(self):

        if self.axis == 0:
            raise ValueError(
                "You are trying to normalize your batch axis. Do you want to "
                "use tf.layer.batch_normalization instead"
            )

    def _create_input_spec(self, input_shape):

        dim = input_shape[self.axis]
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes={self.axis: dim}
        )

    def _add_gamma_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None

    def _add_beta_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(1, self.groups)
        return broadcast_shape


class InstanceNormalization(GroupNormalization):
    """Instance normalization layer.
    Instance Normalization is an specific case of ```GroupNormalization```since
    it normalizes all features of one channel. The Groupsize is equal to the
    channel size. Empirically, its accuracy is more stable than batch norm in a
    wide range of small batch sizes, if learning rate is adjusted linearly
    with batch sizes.
    Arguments
        axis: Integer, the axis that should be normalized.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    Output shape
        Same shape as input.
    References
        - [Instance Normalization: The Missing Ingredient for Fast Stylization]
        (https://arxiv.org/abs/1607.08022)
    """

    def __init__(self, **kwargs):
        if "groups" in kwargs:
            tf.logging.warning("The given value for groups will be overwritten.")

        kwargs["groups"] = -1
        super(InstanceNormalization, self).__init__(**kwargs)
