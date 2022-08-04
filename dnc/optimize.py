"""Optimization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import numpy as np
import tensorflow as tf
from .optimizers import (
    adafactor,
    adamwd,
    addsign,
    lamb,
    lars,
    lookahead,
    msadam,
    powersign,
    radam,
    ralamb,
    yellowfin,
)


def should_generate_summaries():
    """Is this an appropriate context to generate summaries.

    Returns:
      a boolean
    """
    name_scope = tf.contrib.framework.get_name_scope()
    if name_scope and "while/" in name_scope:
        # Summaries don't work well within tf.while_loop()
        return False
    if tf.get_variable_scope().reuse:
        # Avoid generating separate summaries for different data shards
        return False
    return True


def weight_decay_and_noise(
    loss, weight_decay_param, weight_noise_param, learning_rate, var_list=None
):
    """Apply weight decay and weight noise."""
    if var_list is None:
        var_list = tf.trainable_variables()

    decay_vars = [v for v in var_list]
    noise_vars = [v for v in var_list if "/body/" in v.name]

    weight_decay_loss = weight_decay(weight_decay_param, decay_vars)
    if weight_decay_param and should_generate_summaries():
        tf.summary.scalar("losses/weight_decay", weight_decay_loss)
    weight_noise_ops = weight_noise(weight_noise_param, learning_rate, noise_vars)

    with tf.control_dependencies(weight_noise_ops):
        loss = tf.identity(loss)

    loss += weight_decay_loss
    return loss


def weight_noise(noise_rate, learning_rate, var_list):
    """Apply weight noise to vars in var_list."""
    if not noise_rate:
        return [tf.no_op()]

    tf.logging.info(
        "Applying weight noise scaled by learning rate, " "noise_rate: %0.5f",
        noise_rate,
    )

    noise_ops = []

    for v in var_list:
        with tf.device(v.device):  # pylint: disable=protected-access
            scale = noise_rate * learning_rate * 0.001
            if should_generate_summaries():
                tf.summary.scalar("weight_noise_scale", scale)
            noise = tf.truncated_normal(v.shape) * scale
            noise_op = v.assign_add(noise)
            noise_ops.append(noise_op)

    return noise_ops


def weight_decay(decay_rate, var_list, skip_biases=True):
    """Apply weight decay to vars in var_list."""
    if not decay_rate:
        return 0.0

    tf.logging.info("Applying weight decay, decay_rate: %0.5f", decay_rate)

    weight_decays = []
    for v in var_list:
        # Weight decay.
        # This is a heuristic way to detect biases that works for main tf.layers.
        is_bias = len(v.shape.as_list()) == 1 and v.name.endswith("bias:0")
        if not (skip_biases and is_bias):
            with tf.device(v.device):
                v_loss = tf.nn.l2_loss(v)
            weight_decays.append(v_loss)

    return tf.add_n(weight_decays) * decay_rate


def get_parameter_overview(variables, limit=40):
    """Returns a string with variables names, their shapes, count, and types.
      To get all trainable parameters pass in `tf.trainable_variables()`.
      Args:
        variables: List of `tf.Variable`(s).
        limit: If not `None`, the maximum number of variables to include.
      Returns:
        A string with a table like in the example.
      +----------------+---------------+------------+---------+
      | Name           | Shape         | Size       | Type    |
      +----------------+---------------+------------+---------+
      | FC_1/weights:0 | (63612, 1024) | 65,138,688 | float32 |
      | FC_1/biases:0  |       (1024,) |      1,024 | float32 |
      | FC_2/weights:0 |    (1024, 32) |     32,768 | float32 |
      | FC_2/biases:0  |         (32,) |         32 | float32 |
      +----------------+---------------+------------+---------+
      Total: 65,172,512
    """
    # solution for error in calculating shapes for non-trainable variables
    def var_size(v):
        """Calculate variable sizes with None values in them"""
        v_shape = np.array(v.shape.as_list())
        v_shape = v_shape[v_shape != np.array(None)]
        v_size = int(np.prod(v_shape))
        return v_size

    max_name_len = max([len(v.name) for v in variables] + [len("Name")])
    max_shape_len = max([len(str(v.get_shape())) for v in variables] + [len("Shape")])
    max_size_len = max([len(str(var_size(v))) for v in variables] + [len("Size")])
    max_type_len = max(
        [len(v.dtype.base_dtype.name) for v in variables] + [len("Type")]
    )

    var_line_format = "| {: <{}s} | {: >{}s} | {: >{}s} | {: <{}s} |"
    sep_line_format = var_line_format.replace(" ", "-").replace("|", "+")

    header = var_line_format.replace(">", "<").format(
        "Name",
        max_name_len,
        "Shape",
        max_shape_len,
        "Size",
        max_size_len,
        "Type",
        max_type_len,
    )
    separator = sep_line_format.format(
        "", max_name_len, "", max_shape_len, "", max_size_len, "", max_type_len
    )

    lines = [separator, header, separator]

    total_weights = sum(var_size(v) for v in variables)

    # Create lines for up to 80 variables.
    for v in variables:
        if limit is not None and len(lines) >= limit:
            lines.append("[...]")
            break
        lines.append(
            var_line_format.format(
                v.name,
                max_name_len,
                str(v.get_shape()),
                max_shape_len,
                str(var_size(v)),
                max_size_len,
                v.dtype.base_dtype.name,
                max_type_len,
            )
        )

    lines.append(separator)
    lines.append("Total: {:,}".format(total_weights))

    return "\n".join(lines)


def log_parameter_overview(variables, msg):
    """Writes a table with variables name and shapes to INFO log.
        See get_parameter_overview for details.
      Args:
        variables: List of `tf.Variable`(s).
        msg: Message to be logged before the table.
    """
    table = get_parameter_overview(variables, limit=None)
    # The table can to large to fit into one log entry.
    lines = [msg] + table.split("\n")
    for i in range(0, len(lines), 80):
        tf.logging.info("\n%s", "\n".join(lines[i : i + 80]))


def log_variable_sizes(var_list=None, tag=None, verbose=False):
    """Log the sizes and shapes of variables, and the total size.

    Args:
      var_list: a list of variables; defaults to trainable_variables
      tag: a string; defaults to "Trainable Variables"
      verbose: bool, if True, log every weight; otherwise, log total size only.
    """
    if var_list is None:
        var_list = tf.trainable_variables()
    if tag is None:
        tag = "Trainable Variables"

    if not var_list:
        return

    name_to_var = {v.name: v for v in var_list}
    total_size = 0
    for v_name in sorted(list(name_to_var)):
        v = name_to_var[v_name]
        v_type = type(np.array(v.shape.as_list()))
        v_shape = np.array(v.shape.as_list())
        print("v.shape : ", v_shape)
        print("v_type :", v_type)
        v_shape = v_shape[v_shape != np.array(None)]
        v_size = int(np.prod(v_shape))
        if verbose:
            tf.logging.info(
                "Weight: %s\tshape: %s\tsize: %d",
                v.name[:-2].ljust(60),
                str(v.shape).ljust(20),
                v_size,
            )
        total_size += v_size
    tf.logging.info("%s Total size: %d", tag, total_size)


def summarize_variables(
    var_list=None,
    tag=None,
    save_stddev=True,
    save_mean=True,
    save_max=False,
    save_min=False,
):
    """Summarize the variables.

    Args:
      var_list: a list of variables; defaults to trainable_variables.
      tag: name scope of the summary; defaults to training_variables/.
      save_stddev: if True, enables scalar summary for std-dev of given variables
      save_mean: if True, enables scalar summary for mean value of given variables
      save_max: if True, enables scalar summary for max-value of given variables
      save_min: if True, enables scalar summary for min-value of given variables
    """
    if var_list is None:
        var_list = tf.trainable_variables()
    if tag is None:
        tag = "training_variables/"

    TOWER_NAME = "tower"
    name_to_var = {v.name: v for v in var_list}
    for v_name in list(name_to_var):
        v = name_to_var[v_name]
        v_name = re.sub("%s_[0-9]*/" % TOWER_NAME, "", v_name)
        tf.summary.histogram(tag + v_name, v)
        # if save_mean:
        #     mean_var = tf.reduce_mean(v)
        #     tf.summary.scalar(tag + v_name + "/stats/mean/", mean_var)
        #
        # if save_stddev:
        #     stddev_var = tf.math.reduce_std(v)
        #     tf.summary.scalar(tag + v_name + "/stats/stddev/", stddev_var)
        #
        # if save_max:
        #     tf.summary.scalar(tag + v_name + "/stats/max/", tf.reduce_max(v))
        #
        # if save_min:
        #     tf.summary.scalar(tag + v_name + "/stats/min/", tf.reduce_min(v))


def optimize(
    loss,
    learning_rate,
    scope_name,
    optimizer_name="adam",
    adam_beta1=0.85,
    adam_beta2=0.997,
    adam_epsilon=1e-6,
    momentum_polyak=0.9,
    momentum_nesterov=False,
    adafactor_beta1=0.0,
    adafactor_beta2=0.999,
    adafactor_factored=True,
    adafactor_decay_type="adam",
    adafactor_memory_exponent=0.8,
    adafactor_clipping_threshold=1.0,
    adafactor_multiply_by_parameter_scale=True,
    multistep_accumulate_steps=0,
    rmsprop_weightdecay=0.9,
    rmsprop_epsilon=0.001,
    weight_decay_rate=1e-6,
    weight_noise_rate=0.0,
    variables=None,
    summarize_vars=True,
    summarize_grads=True,
    clip_grad_norm=2.0,
    grad_noise_scale=0.0,
):
    """Minimize Loss"""
    loss = weight_decay_and_noise(
        loss, weight_decay_rate, weight_noise_rate, learning_rate
    )
    loss = tf.identity(loss, name="total_loss")
    if variables is None:
        variables = tf.trainable_variables()
    # Print trainable variables
    # log_variable_sizes(variables, verbose=summarize_vars)
    log_parameter_overview(variables, msg="Trainable Variables")
    # Print non-trainable variables.
    non_trainable_variables = list(set(tf.global_variables()) - set(variables))
    # log_variable_sizes(
    #     non_trainable_variables, tag="Non-trainable Variables", verbose=summarize_vars
    # )
    log_parameter_overview(non_trainable_variables, msg="Non-trainable Variables")
    if summarize_vars:
        with tf.name_scope(scope_name + "/"):
            summarize_variables(
                variables,
                tag="trainable_variables/",
                save_stddev=True,
                save_mean=True,
                save_max=False,
                save_min=False,
            )
            # Summarize non-trainable variables as well
            summarize_variables(
                non_trainable_variables,
                tag="non-trainable_variables/",
                save_stddev=False,
                save_mean=False,
                save_min=False,
                save_max=False,
            )

    # Build the optimizer
    if optimizer_name == "sgd":
        tf.logging.info("Using SGD optimizer")
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate, name=scope_name
        )
    elif optimizer_name == "momentum":
        tf.logging.info("Using Momentum optimizer")
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=momentum_polyak, name=scope_name
        )
    elif optimizer_name == "rmsprop":
        tf.logging.info("Using RMSProp optimizer")
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            rmsprop_weightdecay,
            momentum_polyak,
            rmsprop_epsilon,
            name=scope_name,
        )
    elif optimizer_name == "adam":
        tf.logging.info("Using ADAM optimizer")
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=adam_beta1,
            beta2=adam_beta2,
            epsilon=adam_epsilon,
            name=scope_name,
        )
    elif optimizer_name == "multistep_adam":
        tf.logging.info("Using Multistep-ADAM optimizer")
        optimizer = msadam.MultistepAdamOptimizer(
            learning_rate,
            beta1=adam_beta1,
            beta2=adam_beta2,
            epsilon=adam_epsilon,
            name=scope_name,
            n=multistep_accumulate_steps,
        )
    elif optimizer_name == "lazy_adam":
        tf.logging.info("Using LazyAdam optimizer")
        optimizer = tf.contrib.opt.LazyAdamOptimizer(
            learning_rate,
            beta1=adam_beta1,
            beta2=adam_beta2,
            epsilon=adam_epsilon,
            name=scope_name,
        )
    elif optimizer_name == "adam_w":
        tf.logging.info("Using ADAM-WeightDecay optimizer")
        optimizer = tf.contrib.opt.AdamWOptimizer(
            0.01 * learning_rate,
            learning_rate,
            beta1=adam_beta1,
            beta2=adam_beta2,
            epsilon=adam_epsilon,
            name=scope_name,
        )
    elif optimizer_name == "adamwd":
        tf.logging.info("Using ADAM-WeightDecay_correct optimizer")
        # It is recommended that you use this optimizer for fine tuning, since this
        # is how the model was trained (note that the Adam m/v variables are NOT
        # loaded from init_checkpoint.)
        optimizer = adamwd.AdamWeightDecayOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
            name=scope_name,
        )
    elif optimizer_name == "lamb":
        tf.logging.info("Using LAMB (LARS+ADAM-WeightDecay_correct) optimizer")
        optimizer = lamb.LAMBOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
            name=scope_name,
        )
    elif optimizer_name == "lookahead":
        tf.logging.info("Using LookAhead Optimizer")
        optimizer = lookahead.Lookahead(
            optimizer=tf.train.AdamOptimizer(
                learning_rate,
                beta1=adam_beta1,
                beta2=adam_beta2,
                epsilon=adam_epsilon,
                name="ADAMOptimizer",
            ),
            la_alpha=0.8,
            la_steps=5,
            name=scope_name,
        )
    elif optimizer_name == "radam":
        tf.logging.info("Using RADAM Optimizer")
        optimizer = radam.RAdamOptimizer(
            learning_rate=learning_rate,
            beta1=adam_beta1,
            beta2=adam_beta2,
            epsilon=adam_epsilon,
            weight_decay=0.0,
            amsgrad=False,
            total_steps=0,
            warmup_proportion=0.1,
            min_lr=0.0,
            name=scope_name,
        )
    elif optimizer_name == "ralamb":
        tf.logging.info("Using RLAMB Optimizer")
        optimizer = ralamb.RLAMBOptimizer(
            learning_rate=learning_rate,
            beta1=adam_beta1,
            beta2=adam_beta2,
            epsilon=adam_epsilon,
            weight_decay=0.0,
            eeta=0.001,
            trust_ratio_coeff=1.0,
            trust_ratio_clip=None,
            skip_list=["batch_normalization", "bias"],
            name=scope_name,
        )
    elif optimizer_name == "ralamb_la":
        tf.logging.info("Using RLAMB Optimizer with Lookahead")
        optimizer = lookahead.Lookahead(
            optimizer=ralamb.RLAMBOptimizer(
                learning_rate=learning_rate,
                beta1=adam_beta1,
                beta2=adam_beta2,
                epsilon=adam_epsilon,
                weight_decay=0.0,
                eeta=0.001,
                trust_ratio_coeff=1.0,
                trust_ratio_clip=None,
                skip_list=["bias"],
                name="RLAMBOptimizer",
            ),
            la_alpha=0.8,
            la_steps=5,
            name=scope_name,
        )
    elif optimizer_name == "yellow_fin":
        tf.logging.info("Using YellowFin optimizer")
        yf_start_lr = 0.00025
        yf_start_mom = 0.9
        optimizer = yellowfin.YellowFinOptimizer(
            yf_start_lr,
            momentum=yf_start_mom,
            use_nesterov=momentum_nesterov,
            name=scope_name,
        )
    elif optimizer_name == "adafactor":
        tf.logging.info("Using Adafactor optimizer")
        if adafactor_decay_type == "adam":
            decay_rate = adafactor.adafactor_decay_rate_adam(adafactor_beta2)
        elif adafactor_decay_type == "pow":
            decay_rate = adafactor.adafactor_decay_rate_pow(adafactor_memory_exponent)
        else:
            raise ValueError("Unknown adafactor_decay_type")
        optimizer = adafactor.AdafactorOptimizer(
            multiply_by_parameter_scale=adafactor_multiply_by_parameter_scale,
            learning_rate=learning_rate,
            decay_rate=decay_rate,
            beta1=adafactor_beta1,
            clipping_threshold=adafactor_clipping_threshold,
            factored=adafactor_factored,
            name=scope_name,
        )
    else:
        tf.logging.fatal("Unknown optimizer:", optimizer_name)

    opt_summaries = []
    if should_generate_summaries():
        with tf.name_scope(scope_name + "/"):
            if summarize_grads:
                tf.logging.info("Summarizing gradients")
                with tf.name_scope("grad_metrics"):
                    opt_summaries.extend(
                        ["gradients", "gradient_norm", "global_gradient_norm"]
                    )
            with tf.name_scope("opt_metrics"):
                tf.summary.scalar("learning_rate", learning_rate)
                opt_summaries.append("loss")

    if clip_grad_norm:
        tf.logging.info("Clipping gradients, norm: %0.5f", clip_grad_norm)
    if grad_noise_scale:
        tf.logging.info(
            "Adding noise to gradients, noise scale: %0.5f", grad_noise_scale
        )

    global_step = tf.train.get_or_create_global_step()

    train_op = tf.contrib.layers.optimize_loss(
        name=scope_name,
        loss=loss,
        global_step=global_step,
        learning_rate=learning_rate,
        clip_gradients=clip_grad_norm or None,
        gradient_noise_scale=grad_noise_scale or None,
        optimizer=optimizer,
        summaries=opt_summaries,
        colocate_gradients_with_ops=True,
        variables=variables,
    )
    if optimizer_name == "adamwd":
        # Normally the global step update is done inside of `apply_gradients`.
        # However, `AdamWeightDecayOptimizer` and `LAMBOptimizer` doesn't do this.
        # But if you use a different optimizer, you should probably take this line out.
        new_global_step = global_step + 1
        train_op = tf.group(train_op, [global_step.assign(new_global_step)])

    return train_op
