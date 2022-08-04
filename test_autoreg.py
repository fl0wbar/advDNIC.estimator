from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from model.utils import HParams

from dnc.utils import *
# from model.generators.variant3 import generator
from model.generators.variant4 import generator

from mpl_toolkits import axes_grid1


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def compress(args):
    """Compresses an image."""

    # Load input image and add batch dimension.
    x = read_png(args.input_file)
    x = tf.expand_dims(x, 0)
    x.set_shape([1, None, None, 3])
    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)
    channel_axis = -1
    spatial_dims = [1, 2]
    x_shape = tf.shape(x)
    x_img = x[0, : x_shape[1], : x_shape[2], :]

    json_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(
        json_path
    )
    hparams = HParams(json_path)

    genx, genximg, eval_bpp, encoding_tensors, gen_metadata = generator(
        params=hparams, inputs=x, is_training=False, predict=True
    )

    maxhl_ent_loc = gen_metadata["maxhl_ent_loc"]
    maxhl_ent_val = gen_metadata["maxhl_ent_val"]
    maxl_ent_loc = gen_metadata["maxl_ent_loc"]
    maxl_ent_val = gen_metadata["maxl_ent_val"]
    hyperlatent_entropy_pc = gen_metadata["hyperlatent_entropy_pc"]
    latent_entropy_pc = gen_metadata["latent_entropy_pc"]
    hle_shape = gen_metadata["hle_shape"]
    hle_pc_shape = gen_metadata["hle_pc_shape"]
    le_shape = gen_metadata["le_shape"]
    le_pc_shape = gen_metadata["le_pc_shape"]
    z_hat_slice_maxhle = gen_metadata["z_hat_slice_maxhle"]
    y_hat_slice_maxle = gen_metadata["y_hat_slice_maxle"]
    hle_max_slice = gen_metadata["hle_max_slice"]
    le_max_slice = gen_metadata["le_max_slice"]
    sigma_slice = gen_metadata["sigma_slice"]
    mu_slice = gen_metadata["mu_slice"]

    # Bring both images back to 0..255 range.
    x *= 255
    yuv_x = tf.image.rgb_to_yuv(x)
    genx = tf.clip_by_value(genx, 0, 1)
    yuv_genx = tf.image.rgb_to_yuv(genx)
    genx = tf.round(genx * 255)

    pred_error = tf.squeeze(tf.math.subtract(x, genx))
    # metrics in RGB colorspace
    eval_mse = tf.reduce_mean(tf.squared_difference(x, genx))
    eval_psnr = tf.squeeze(tf.image.psnr(genx, x, 255))
    # metrics in YUV colorspace
    eval_mse_yuv = tf.reduce_mean(tf.squared_difference(yuv_x, yuv_genx))
    eval_psnr_yuv = tf.squeeze(tf.image.psnr(yuv_x, yuv_genx, 1.0))

    # The following ops are inherently optimized for cpu
    with tf.device("/cpu:0"):
        # metrics in RGB colorspace
        eval_ssim = tf.squeeze(tf.image.ssim(genx, x, 255))
        eval_msssim = tf.squeeze(tf.image.ssim_multiscale(genx, x, 255))
        # metrics in YUV colorspace
        eval_ssim_yuv = tf.squeeze(tf.image.ssim(yuv_genx, yuv_x, 1.0))
        eval_msssim_yuv = tf.squeeze(tf.image.ssim_multiscale(yuv_genx, yuv_x, 1.0))

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        # Load the pretrained model
        print("Loading weights from the pre-trained model")
        if hparams.use_adversarial_loss:
            checkpoint_path = os.path.join(args.model_dir, "generator")
        else:
            checkpoint_path = args.model_dir
        latest = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_path)
        ckpt_iteration = int(latest.split("-")[-1])
        print("Variables loaded from the checkpoint")
        print(
            "/// The latest tf.Checkpoint has been trained for {} iterations >>".format(
                ckpt_iteration
            )
        )
        # pprint.pprint(tf.train.list_variables(latest))
        tf.train.Saver().restore(sess, save_path=latest)

        arrays = sess.run(encoding_tensors)
        # Write a binary file with the shape information and the compressed string.
        packed = PackedTensors()
        packed.pack(encoding_tensors, arrays)
        with open(args.output_file, "wb") as f:
            f.write(packed.string)

        # If requested, transform the quantized image back and measure performance.
        if args.verbose:

            [
                maxhl_ent_loc,
                maxhl_ent_val,
                maxl_ent_loc,
                maxl_ent_val,
                hyperlatent_entropy_pc,
                latent_entropy_pc,
                hle_shape,
                hle_pc_shape,
                le_shape,
                le_pc_shape,
                z_hat_slice_maxhle,
                y_hat_slice_maxle,
                hle_max_slice,
                le_max_slice,
                sigma_slice,
                mu_slice,
                pred_error,
                x_img,
                genximg,
            ] = sess.run(
                [
                    maxhl_ent_loc,
                    maxhl_ent_val,
                    maxl_ent_loc,
                    maxl_ent_val,
                    hyperlatent_entropy_pc,
                    latent_entropy_pc,
                    hle_shape,
                    hle_pc_shape,
                    le_shape,
                    le_pc_shape,
                    z_hat_slice_maxhle,
                    y_hat_slice_maxle,
                    hle_max_slice,
                    le_max_slice,
                    sigma_slice,
                    mu_slice,
                    pred_error,
                    x_img,
                    genximg,
                ]
            )

            print("latent_entropy_pc : ", latent_entropy_pc)
            print("location of max latent_entropy_pc : ", np.argmax(latent_entropy_pc))
            print("hyperlatent_entropy_pc : ", hyperlatent_entropy_pc)
            print(
                "location of max hyperlatent_entropy_pc : ",
                np.argmax(hyperlatent_entropy_pc),
            )
            print("//// hle_pc_shape : ", hle_pc_shape)
            print("//// hle_shape : ", hle_shape)
            print("//// le_pc_shape : ", le_pc_shape)
            print("//// le_shape : ", le_shape)
            print("//// channel containing max hyper-latent entropy : ", maxhl_ent_loc)
            print("//// channel containing max latent entropy : ", maxl_ent_loc)

            print("//// value of maximum entropy hyperlatent : ", maxhl_ent_val)
            print("//// value of maximum entropy latent : ", maxl_ent_val)
            print("////// Latent-MaxEntropy : ", y_hat_slice_maxle)
            print("////// HyperLatent-MaxEntropy : ", z_hat_slice_maxhle)

            (
                eval_bpp,
                eval_mse_yuv,
                eval_psnr_yuv,
                eval_ssim_yuv,
                eval_msssim_yuv,
                eval_mse,
                eval_psnr,
                eval_ssim,
                eval_msssim,
                num_pixels,
            ) = sess.run(
                [
                    eval_bpp,
                    eval_mse_yuv,
                    eval_psnr_yuv,
                    eval_ssim_yuv,
                    eval_msssim_yuv,
                    eval_mse,
                    eval_psnr,
                    eval_ssim,
                    eval_msssim,
                    num_pixels,
                ]
            )

            # The actual bits per pixel including overhead.
            bpp = len(packed.string) * 8 / num_pixels

            print("Mean squared error: {:0.4f}".format(eval_mse))
            print("PSNR (dB): {:0.2f}".format(eval_psnr))
            print("Multiscale SSIM: {:0.4f}".format(eval_msssim))
            print(
                "Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - eval_msssim))
            )
            print("Information content in bpp: {:0.4f}".format(eval_bpp))
            print("Actual bits per pixel: {:0.4f}".format(bpp))

            with open(args.output_file + ".txt", "w") as txtf:
                txtf.write("Mean squared error: {:0.4f}\n".format(eval_mse))
                txtf.write("PSNR (dB): {:0.2f}\n".format(eval_psnr))
                txtf.write("Multiscale SSIM: {:0.4f}\n".format(eval_msssim))
                txtf.write(
                    "Multiscale SSIM (dB): {:0.2f}\n".format(
                        -10 * np.log10(1 - eval_msssim)
                    )
                )
                txtf.write("Information content in bpp: {:0.4f}\n".format(eval_bpp))
                txtf.write("Actual bits per pixel: {:0.4f}\n".format(bpp))

    if args.verbose:
        y_hat_maxle_norm = preprocessing.normalize(y_hat_slice_maxle)
        y_hat_maxle_norm = np.interp(
            y_hat_maxle_norm, (y_hat_maxle_norm.min(), y_hat_maxle_norm.max()), (-4, +4)
        )
        # z_hat_maxhle_norm = preprocessing.normalize(z_hat_slice_maxhle)
        fig = plt.figure(figsize=[16, 8])
        a = fig.add_subplot(2, 5, 1)
        latent_plot = plt.imshow(x_img)
        a.set_title("Input")
        a = fig.add_subplot(2, 5, 2)
        latent_plot = plt.imshow(genximg)
        a.set_title("Reconstruction")
        a = fig.add_subplot(2, 5, 3)
        latent_plot = plt.imshow(y_hat_slice_maxle, cmap="bwr_r")
        a.set_title("Latent")
        add_colorbar(latent_plot)
        a = fig.add_subplot(2, 5, 4)
        latent_plot = plt.imshow(y_hat_maxle_norm, cmap="bwr_r", vmin=-4, vmax=4)
        a.set_title("Latent Normalized")
        add_colorbar(latent_plot)
        a = fig.add_subplot(2, 5, 5)
        latent_plot = plt.imshow(z_hat_slice_maxhle, cmap="bwr_r")
        a.set_title("Hyper-Latent")
        add_colorbar(latent_plot)
        a = fig.add_subplot(2, 5, 6)
        latent_plot = plt.imshow(pred_error)
        a.set_title("Prediction Error")
        a = fig.add_subplot(2, 5, 7)
        latent_plot = plt.imshow(mu_slice, cmap="inferno")
        a.set_title("Predicted Means")
        add_colorbar(latent_plot)
        a = fig.add_subplot(2, 5, 8)
        latent_plot = plt.imshow(sigma_slice, cmap="inferno")
        a.set_title("Predicted Scale")
        add_colorbar(latent_plot)
        a = fig.add_subplot(2, 5, 9)
        latent_plot = plt.imshow(le_max_slice, cmap="inferno")
        a.set_title("Latent Entropy")
        add_colorbar(latent_plot)
        a = fig.add_subplot(2, 5, 10)
        latent_plot = plt.imshow(hle_max_slice, cmap="inferno")
        a.set_title("Hyper-Latent Entropy")
        add_colorbar(latent_plot)
        plt.tight_layout()
        plt.show()


def decompress(args):
    """Decompresses an image."""

    # Read the shape information and compressed string from the binary file.
    string = tf.placeholder(tf.string, [1])
    side_string = tf.placeholder(tf.string, [1])
    x_shape = tf.placeholder(tf.int32, [2])
    y_shape = tf.placeholder(tf.int32, [2])
    z_shape = tf.placeholder(tf.int32, [2])
    with open(args.input_file, "rb") as f:
        packed = PackedTensors(f.read())
    tensors = [string, side_string, x_shape, y_shape, z_shape]
    arrays = packed.unpack(tensors)

    json_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(
        json_path
    )
    hparams = HParams(json_path)

    # Decompress and transform the image back.
    z_shape = tf.concat([z_shape, [hparams.num_filters]], axis=0)

    decompression_vars = dict()
    decompression_vars["string"] = string
    decompression_vars["side_string"] = side_string
    decompression_vars["z_shape"] = z_shape
    decompression_vars["y_shape"] = y_shape

    x_hat, y_hat, z_hat, sigma = generator(
        params=hparams,
        inputs=None,
        is_training=False,
        predict=False,
        decompression=True,
        decompression_vars=decompression_vars,
    )

    # Remove batch dimension, and crop away any extraneous padding on the bottom
    # or right boundaries.
    x_hat = x_hat[0, : x_shape[0], : x_shape[1], :]

    # Write reconstructed image out as a PNG file.
    op = write_png(args.output_file, x_hat)

    # Load the latest model checkpoint, and perform the above actions.
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        latest = tf.train.latest_checkpoint(checkpoint_dir=args.model_dir)
        tf.train.Saver().restore(sess, save_path=latest)
        sess.run(op, feed_dict=dict(zip(tensors, arrays)))
