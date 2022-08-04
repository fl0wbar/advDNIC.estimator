"""General utility functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import datetime
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim


class HParams:
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, "w") as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def log_metrics_to_csv(epoch, metrics_val, log_path):
    """
        Saves evaluation metrics to a csv file in the log_path
    :param epoch: (int) epoch value during training
    :param metrics_val: (dict) dictionary of evaluation metrics
    :param log_path: (string) path to save the generated csv file
    """
    log_line = ""
    detail_log_line = ""
    log_line += " , ".join("{:05.3f}".format(v) for v in metrics_val.values())
    detail_log_line += " , ".join(
        "{}: {:05.3f}".format(k, v) for k, v in metrics_val.items()
    )
    # Write compression metrics to log
    with open(log_path + "/compression_metrics_num.csv", "a") as f:
        f.write("%d, %s\n" % (epoch, log_line))
    with open(log_path + "/compression_metrics_detail.csv", "a") as df:
        df.write("epoch: {:05f}, {}\n".format(epoch, detail_log_line))


def build_log_dir(args, arguments):
    """Set up a timestamped directory for results and logs for this training session"""
    if args.model_dir:
        log_path = args.model_dir  # (name + '_') if name else ''
    else:
        log_path = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join("results", log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    print('Logging results for this session in folder "%s".' % log_path)

    # Write command line arguments to file
    with open(log_path + "/args.txt", "w+") as f:
        f.write(" ".join(arguments))
    return log_path


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


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
    """A version of tf.metrics.mean_tensor that handles float64 values.
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
    """A streaming an unbiased version of tfp.stats.covariance.
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
