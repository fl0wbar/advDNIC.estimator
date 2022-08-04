import logging
import os
import sys
import math
from datetime import datetime
import random
import coloredlogs

import tensorflow.compat.v1 as tf
from tensorflow import logging
import tensorflow as tfmain

# import mlflow
# import mlflow.tensorflow

from model.input_func import input_fn, predict_input_fn
from model.utils import HParams
from model.utils import set_logger
from model.utils import build_log_dir
from model.structure import model_fn

# # Enable auto-logging to MLflow to capture TensorBoard metrics.
# mlflow.tensorflow.autolog()


def train(args):
    # create logging directory if it doesn't exist
    build_log_dir(args, sys.argv)
    json_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(
        json_path
    )
    hparams = HParams(json_path)

    # # mlflow.tracking.set_tracking_uri("http://localhost:5000/")
    # experiment_id = mlflow.create_experiment("advDNIC-generatorV1")
    # # Enable auto-logging to MLflow to capture TensorBoard metrics.
    # mlflow.tensorflow.autolog()

    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(args.random_seed)

    if args.verbose:
        coloredlogs.install(level=args.log_verbosity)
        tf.logging.set_verbosity(args.log_verbosity)
        # Set C++ Graph Execution level verbosity
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(
            tf.logging.__dict__[args.log_verbosity] / 10
        )

    # Set the logger
    set_logger(os.path.join(args.model_dir, "train.log"))

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    train_data_dir = args.train_data_dir
    eval_data_dir = args.eval_data_dir

    # Get the filenames from the train and dev sets
    train_filenames = [
        os.path.join(train_data_dir, f)
        for f in os.listdir(train_data_dir)
        if f.endswith(".png")
    ]
    eval_filenames = [
        os.path.join(eval_data_dir, f)
        for f in os.listdir(eval_data_dir)
        if f.endswith(".png")
    ]

    # Specify the sizes of the dataset we train on and evaluate on
    hparams.model_dir = args.model_dir
    hparams.train_data_dir = train_data_dir
    hparams.eval_data_dir = eval_data_dir
    hparams.train_size = len(train_filenames)
    hparams.eval_size = len(eval_filenames)
    hparams.batchsize = args.batchsize
    hparams.num_parallel_calls = args.num_parallel_calls

    steps_per_epoch = int(math.ceil(hparams.train_size / hparams.batchsize))
    train_steps = steps_per_epoch * args.epochs

    train_input_fn = lambda: input_fn(mode=tf.estimator.ModeKeys.TRAIN, params=hparams)
    eval_input_fn = lambda: input_fn(mode=tf.estimator.ModeKeys.EVAL, params=hparams)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = args.allow_growth
    if args.xla:
        session_config.graph_options.optimizer_options.global_jit_level = (
            tf.OptimizerOptions.ON_1
        )
        tfmain.config.optimizer.set_jit(enabled=True)

    run_config = tf.estimator.RunConfig(
        model_dir=args.model_dir,
        tf_random_seed=args.random_seed,
        save_summary_steps=args.save_summary_steps,
        save_checkpoints_steps=steps_per_epoch,
        session_config=session_config,
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, config=run_config, params=hparams
    )

    hooks = []
    if args.save_profiling_steps:
        hooks.append(
            tf.train.ProfilerHook(
                save_steps=args.save_profiling_steps, output_dir=args.model_dir
            )
        )
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=train_steps, hooks=hooks
    )

    # def _lower_rd_loss(best_eval_result, current_eval_result):
    #     metric = "loss"
    #     return best_eval_result[metric] > current_eval_result[metric]

    # serving_exporter = tf.estimator.BestExporter(
    #     name="estimatorBest",  # Saved models are exported under /export/estimateBest/
    #     serving_input_receiver_fn=predict_input_fn(),
    #     exports_to_keep=2,
    # )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=hparams.eval_size / hparams.batchsize,
        exporters=tf.estimator.LatestExporter(
            name="Servo", serving_input_receiver_fn=predict_input_fn
        )
        if predict_input_fn
        else None,
    )

    time_start = datetime.utcnow()
    print("=" * 80)
    print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
    print("=" * 80)

    # with mlflow.start_run(experiment_id=experiment_id):
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    time_end = datetime.utcnow()

    print("=" * 80)
    print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
    time_elapsed = time_end - time_start
    print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))
    print("=" * 80)
