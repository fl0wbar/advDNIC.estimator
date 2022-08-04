"""
    Defines the model
"""
import os
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops

# from model.generators.variant3 import generator
from model.generators.variant4 import generator
from model.summaries import visualize_yuv
from dnc.lr import build_learning_rate
from dnc.optimize import optimize
from dnc.utils import quantize_image


def make_tf_metric(
    metric, metrics_collections=None, updates_collections=None, name=None
):
    """
        Computes the (weighted) mean of the given metric (adapted from tf.metric.mean)
        This function makes a tf.Tensor behave as a metric by wrapping it in tf.metric.mean

    Args:
        :param metric: tensor value to be used as a metric (A Tensor of arbitrary dimensions).
        :param metrics_collections: An optional list of collections that the `metric` should be added to.
        :param updates_collections: An optional list of collections that `update_op` should be added to.
        :param name: An optional variable_scope name.

    Returns:
        :return: metric:  A Tensor representing the current metric, the value of total divided by count.
        :return: update_op: An operation that increments the total and count variables appropriately
                            and whose value matches metric_value
    """
    with tf.variable_scope("metric_ops"):
        metric_mean, metric_update_op = tf.metrics.mean(metric, name=name)

    if metrics_collections:
        ops.add_to_collections(metrics_collections, metric_mean)

    if updates_collections:
        ops.add_to_collections(updates_collections, metric_update_op)

    return metric_mean, metric_update_op


def model_fn(features, labels, mode, params):
    global_step = tf.train.get_or_create_global_step()
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    # Training mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        # get inputs for training
        print("/////////// features.keys : ", features.keys())
        x = features["source"]
        print("/////////// features['source'] -> x : ", x)
        print("/////////// labels.keys : ", labels.keys())
        gtx = labels["target"]
        print("/////////// labels['target'] -> gtx : ", gtx)

        genx, bpp, entropy_bottleneck = generator(
            params=params, inputs=x, is_training=is_training, predict=False
        )

        # for metric calculations bring both input and generated output to 0..255 range
        x_metric = gtx
        yuv_x_metric = tf.image.rgb_to_yuv(gtx)
        genx_metric = genx
        genx_metric = tf.clip_by_value(genx_metric, 0, 1)
        yuv_genx_metric = tf.image.rgb_to_yuv(genx_metric)

        # metrics in RGB colorspace
        # # Mean squared error across pixels. Multiply by 255^2 to correct for rescaling.
        train_mse = tf.reduce_mean(tf.squared_difference(x_metric, genx_metric)) * (
            255 ** 2
        )
        train_psnr = tf.squeeze(tf.image.psnr(genx_metric, x_metric, 1.0))
        train_ssim = tf.squeeze(tf.image.ssim(genx_metric, x_metric, 1.0))
        train_msssim = tf.squeeze(tf.image.ssim_multiscale(genx_metric, x_metric, 1.0))

        # metrics in YUV colorspace
        train_mse_yuv = tf.reduce_mean(
            tf.squared_difference(yuv_x_metric, yuv_genx_metric)
        )
        train_psnr_yuv = tf.squeeze(tf.image.psnr(yuv_x_metric, yuv_genx_metric, 1.0))
        train_ssim_yuv = tf.squeeze(tf.image.ssim(yuv_genx_metric, yuv_x_metric, 1.0))
        train_msssim_yuv = tf.squeeze(
            tf.image.ssim_multiscale(yuv_genx_metric, yuv_x_metric, 1.0)
        )

        # The rate-distortion cost.
        rd_loss = params.lmbda * train_mse + bpp

        # Minimize loss and auxiliary loss, and execute update op.
        steps_per_epoch = params.train_size / params.batchsize
        # Compute the current epoch and associated learning rate from global_step.
        # current_epoch = tf.cast(global_step, tf.float32) / steps_per_epoch
        scaled_lr = (params.base_learning_rate / 2.0) * (params.batchsize / 2.0)

        main_opt_lr = build_learning_rate(
            scaled_lr,
            global_step,
            steps_per_epoch,
            lr_decay_type=params.main_opt_lr_decay_type,
            warmup_epochs=False,
        )

        # main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        main_step = optimize(
            loss=rd_loss,
            learning_rate=main_opt_lr,
            optimizer_name=params.main_optimizer,
            scope_name="main_optimizer",
        )

        aux_opt_lr = build_learning_rate(
            scaled_lr * 10,
            global_step,
            steps_per_epoch,
            lr_decay_type=params.aux_opt_lr_decay_type,
            warmup_epochs=5,
        )

        # aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        aux_step = optimize(
            loss=entropy_bottleneck.losses[0],
            learning_rate=aux_opt_lr,
            optimizer_name=params.aux_optimizer,
            scope_name="aux_optimizer",
        )

        train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

        # Visualisation utils
        orig_image = quantize_image(gtx)
        recn_image = quantize_image(genx)
        orig_image_yuv = quantize_image(yuv_x_metric)
        recn_image_yuv = quantize_image(yuv_genx_metric)
        orig_R = tf.expand_dims(orig_image[:, :, :, 0], axis=3)
        orig_G = tf.expand_dims(orig_image[:, :, :, 1], axis=3)
        orig_B = tf.expand_dims(orig_image[:, :, :, 2], axis=3)
        orig_yuv_Y = tf.expand_dims(orig_image_yuv[:, :, :, 0], axis=3)
        orig_yuv_U = tf.expand_dims(orig_image_yuv[:, :, :, 1], axis=3)
        orig_yuv_V = tf.expand_dims(orig_image_yuv[:, :, :, 2], axis=3)
        recn_R = tf.expand_dims(recn_image[:, :, :, 0], axis=3)
        recn_G = tf.expand_dims(recn_image[:, :, :, 1], axis=3)
        recn_B = tf.expand_dims(recn_image[:, :, :, 2], axis=3)
        recn_yuv_Y = tf.expand_dims(recn_image_yuv[:, :, :, 0], axis=3)
        recn_yuv_U = tf.expand_dims(recn_image_yuv[:, :, :, 1], axis=3)
        recn_yuv_V = tf.expand_dims(recn_image_yuv[:, :, :, 2], axis=3)
        orig_RGB = tf.concat([orig_R, orig_G, orig_B], axis=2)
        orig_YUV = tf.concat([orig_yuv_Y, orig_yuv_U, orig_yuv_V], axis=2)
        recn_RGB = tf.concat([recn_R, recn_G, recn_B], axis=2)
        recn_YUV = tf.concat([recn_yuv_Y, recn_yuv_U, recn_yuv_V], axis=2)

        vis_og_YUV = visualize_yuv(orig_image_yuv, params)
        vis_rc_YUV = visualize_yuv(recn_image_yuv, params)

        # Summary
        with tf.name_scope("training_metrics/"):

            tf.summary.scalar("main_opt_lr", main_opt_lr)
            tf.summary.scalar("aux_opt_lr", aux_opt_lr)

            tf.summary.scalar("loss", rd_loss)
            tf.summary.scalar("bpp", bpp)
            tf.summary.scalar("mse", train_mse)
            tf.summary.scalar("psnr", tf.reduce_mean(train_psnr))
            tf.summary.scalar("ssim", tf.reduce_mean(train_ssim))
            tf.summary.scalar("msssim", tf.reduce_mean(train_msssim))
            tf.summary.scalar("mse_YUV", train_mse_yuv)
            tf.summary.scalar("psnr_YUV", tf.reduce_mean(train_psnr_yuv))
            tf.summary.scalar("ssim_YUV", tf.reduce_mean(train_ssim_yuv))
            tf.summary.scalar("msssim_YUV", tf.reduce_mean(train_msssim_yuv))
            tf.summary.image("original", orig_image, family="original")
            tf.summary.image("original_RGB", orig_RGB, family="original_RGB")
            # tf.summary.image("original_YUV", orig_YUV, family="original_YUV")
            tf.summary.image("reconstruction", recn_image, family="reconstruction")
            tf.summary.image("recon_RGB", recn_RGB, family="recn_RGB")
            # tf.summary.image("recon_YUV", recn_YUV, family="recn_YUV")
            tf.summary.image("original_YUV", vis_og_YUV, family="original_YUV")
            tf.summary.image("recon_YUV", vis_rc_YUV, family="recn_YUV")

        stats = tf.profiler.profile()
        print("Total parameters:", stats.total_parameters)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=rd_loss,
            train_op=train_op,
            eval_metric_ops={},
            predictions={},
        )

    # Evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
        # get inputs
        print("/////////// features.keys : ", features.keys())
        x = features["source"]
        print("/////////// features['source'] -> x : ", x)
        print("/////////// labels.keys : ", labels.keys())
        gtx = labels["target"]
        print("/////////// labels['target'] -> gtx : ", gtx)

        genx, bpp = generator(params=params, inputs=x, is_training=False, predict=False)

        summ_genx = genx
        # for metric calculations bring both input and generated output to 0..255 range
        x_metric = gtx * 255
        yuv_x_metric = tf.image.rgb_to_yuv(gtx)
        genx_metric = tf.clip_by_value(genx, 0, 1)
        yuv_genx_metric = tf.image.rgb_to_yuv(genx_metric)
        genx_metric = tf.round(genx_metric * 255)

        # metrics in RGB colorspace
        eval_mse = tf.reduce_mean(tf.squared_difference(x_metric, genx_metric))
        eval_psnr = tf.squeeze(tf.image.psnr(genx_metric, x_metric, 255))
        eval_ssim = tf.squeeze(tf.image.ssim(genx_metric, x_metric, 255))
        eval_msssim = tf.squeeze(tf.image.ssim_multiscale(genx_metric, x_metric, 255))

        # metrics in YUV colorspace
        eval_mse_yuv = tf.reduce_mean(
            tf.squared_difference(yuv_x_metric, yuv_genx_metric)
        )
        eval_psnr_yuv = tf.squeeze(tf.image.psnr(yuv_x_metric, yuv_genx_metric, 1.0))
        eval_ssim_yuv = tf.squeeze(tf.image.ssim(yuv_genx_metric, yuv_x_metric, 1.0))
        eval_msssim_yuv = tf.squeeze(
            tf.image.ssim_multiscale(yuv_genx_metric, yuv_x_metric, 1.0)
        )

        # The rate-distortion cost.
        rd_loss = params.lmbda * eval_mse + bpp

        # with tf.name_scope("estimator_eval/"):
        eval_metric_ops = {
            "estimator_eval/RD_Loss": make_tf_metric(rd_loss),
            "estimator_eval/MSE": make_tf_metric(eval_mse),
            "estimator_eval/PSNR": make_tf_metric(eval_psnr),
            "estimator_eval/SSIM": make_tf_metric(eval_ssim),
            "estimator_eval/MS_SSIM": make_tf_metric(eval_msssim),
            "estimator_eval/MSE_YUV": make_tf_metric(eval_mse_yuv),
            "estimator_eval/PSNR_YUV": make_tf_metric(eval_psnr_yuv),
            "estimator_eval/SSIM_YUV": make_tf_metric(eval_ssim_yuv),
            "estimator_eval/MS_SSIM_YUV": make_tf_metric(eval_msssim_yuv),
        }

        # Accumulate input and generated image batches for grid summaries
        current_step = tf.cast(tf.train.get_global_step(), tf.float32)
        print("///////////////// current EVAL step : ", current_step)

        # Visualisation utils
        orig_image = quantize_image(gtx)
        recn_image = quantize_image(summ_genx)
        orig_image_yuv = quantize_image(yuv_x_metric)
        recn_image_yuv = quantize_image(yuv_genx_metric)
        orig_R = tf.expand_dims(orig_image[:, :, :, 0], axis=3)
        orig_G = tf.expand_dims(orig_image[:, :, :, 1], axis=3)
        orig_B = tf.expand_dims(orig_image[:, :, :, 2], axis=3)
        orig_yuv_Y = tf.expand_dims(orig_image_yuv[:, :, :, 0], axis=3)
        orig_yuv_U = tf.expand_dims(orig_image_yuv[:, :, :, 1], axis=3)
        orig_yuv_V = tf.expand_dims(orig_image_yuv[:, :, :, 2], axis=3)
        recn_R = tf.expand_dims(recn_image[:, :, :, 0], axis=3)
        recn_G = tf.expand_dims(recn_image[:, :, :, 1], axis=3)
        recn_B = tf.expand_dims(recn_image[:, :, :, 2], axis=3)
        recn_yuv_Y = tf.expand_dims(recn_image_yuv[:, :, :, 0], axis=3)
        recn_yuv_U = tf.expand_dims(recn_image_yuv[:, :, :, 1], axis=3)
        recn_yuv_V = tf.expand_dims(recn_image_yuv[:, :, :, 2], axis=3)
        orig_RGB = tf.concat([orig_R, orig_G, orig_B], axis=2)
        orig_YUV = tf.concat([orig_yuv_Y, orig_yuv_U, orig_yuv_V], axis=2)
        recn_RGB = tf.concat([recn_R, recn_G, recn_B], axis=2)
        recn_YUV = tf.concat([recn_yuv_Y, recn_yuv_U, recn_yuv_V], axis=2)

        vis_og_YUV = visualize_yuv(orig_image_yuv, params)
        vis_rc_YUV = visualize_yuv(recn_image_yuv, params)

        # Summary
        eval_summary_ops = [
            tf.summary.scalar("eval_metrics/loss", rd_loss),
            tf.summary.scalar("eval_metrics/bpp", bpp),
            tf.summary.scalar("eval_metrics/mse", eval_mse),
            tf.summary.scalar("eval_metrics/psnr", tf.reduce_mean(eval_psnr)),
            tf.summary.scalar("eval_metrics/ssim", tf.reduce_mean(eval_ssim)),
            tf.summary.scalar("eval_metrics/msssim", tf.reduce_mean(eval_msssim)),
            tf.summary.scalar("eval_metrics/mse_YUV", eval_mse_yuv),
            tf.summary.scalar("eval_metrics/psnr_YUV", tf.reduce_mean(eval_psnr_yuv)),
            tf.summary.scalar("eval_metrics/ssim_YUV", tf.reduce_mean(eval_ssim_yuv)),
            tf.summary.scalar(
                "eval_metrics/msssim_YUV", tf.reduce_mean(eval_msssim_yuv)
            ),
            tf.summary.image("eval_metrics/original", orig_image),
            tf.summary.image("eval_metrics/RGB/original_RGB", orig_RGB),
            # tf.summary.image("eval_metrics/original_YUV", orig_YUV),
            tf.summary.image("eval_metrics/reconstruction", recn_image),
            tf.summary.image("eval_metrics/RGB/recon_RGB", recn_RGB),
            # tf.summary.image("eval_metrics/recon_YUV", recn_YUV),
            tf.summary.image("eval_metrics/YUV/original_YUV", vis_og_YUV),
            tf.summary.image("eval_metrics/YUV/recon_YUV", vis_rc_YUV),
        ]

        # output eval images
        eval_summary_hook = tf.train.SummarySaverHook(
            save_steps=100,
            output_dir=os.path.join(params.model_dir, "eval"),
            summary_op=eval_summary_ops,
        )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=rd_loss,
            eval_metric_ops=eval_metric_ops,
            evaluation_hooks=[eval_summary_hook],
        )

    if mode == tf.estimator.ModeKeys.PREDICT:
        # get inputs
        print("/////////// features.keys : ", features.keys())
        x = features["source"]
        print("/////////// features['source'] -> x : ", x)

        genx, genximg, bpp, encoding_tensors, generator_metadata = generator(
            params=params, inputs=x, is_training=False, predict=True
        )

        # Convert predicted_indices back into strings.
        # predictions = {
        #     "reconstructionbatch": genx,
        #     "reconstructionimg": genximg,
        #     "bpp": bpp,
        #     "encoded_string": encoding_tensors[0],
        #     "encoded_sidestring": encoding_tensors[1],
        #     "input_shape": encoding_tensors[2],
        #     "input_analysis_shape": encoding_tensors[3],
        #     "input_hyperanalysis_shape": encoding_tensors[4],
        # }

        predictions = {
            "reconstructionbatch": genx,
            "reconstructionimg": genximg,
            "bpp": bpp,
            "encoded_string": encoding_tensors[0],
            "encoded_sidestring": encoding_tensors[1],
            "input_shape": encoding_tensors[2],
            "input_analysis_shape": encoding_tensors[3],
            "input_hyperanalysis_shape": encoding_tensors[4],
        }

        export_outputs = {"predict": tf.estimator.export.PredictOutput(predictions)}
        # Provide an estimator spec for `ModeKeys.PREDICT` modes.
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs
        )
