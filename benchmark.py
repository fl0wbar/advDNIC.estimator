import resource
from skimage.color import rgb2ycbcr
from PIL import Image
import tensorflow.compat.v1 as tf

from dnc.utils import *
from model.utils import HParams
from model.generators.variant1 import generator


class Benchmark(object):
    """
        A class for evaluation of compression model on a set of benchmarking images
    """

    def __init__(self, args, path, name):
        self.model_args = args
        self.path = path
        self.benchmarkname = name
        self.filenames, self.names = self.load_filenames_by_model(model="HR")

    def load_filenames_by_model(self, model, file_format="png"):
        """
            Loads all images that match '*_{model}.{file_format}'
            and returns sorted list of filenames and names
        """
        # Get files that match the pattern
        filenames = sorted(
            glob.glob(os.path.join(self.path, "*_" + model + "." + file_format))
        )
        # Extract name/prefix eg: '/.../baby_LR.png' -> 'baby'
        names = [os.path.basename(x).split("_")[0] for x in filenames]
        return filenames, names

    @staticmethod
    def deprocess(image):
        """ Deprocess image output by model (from -1 to 1 float to 0 to 255 uint8) """
        image = np.clip(255 * 0.5 * (image + 1.0), 0.0, 255.0).astype(np.uint8)
        return image

    @staticmethod
    def luminance(image):
        # Get luminance
        lum = rgb2ycbcr(image)[:, :, 0]
        return lum

    @staticmethod
    def save_image(image, path):
        if not os.path.exists(os.path.split(path)[0]):
            os.makedirs(os.path.split(path)[0])
        Image.fromarray(image).save(path, format="png")

    def benchmarkcompression(self):
        """
            Evaluate compression performance, returning the metrics and saving images
            and compressed files
        """

        json_path = os.path.join(self.model_args.model_dir, "params.json")
        assert os.path.isfile(
            json_path
        ), "No json configuration file found at {}".format(json_path)
        hparams = HParams(json_path)
        rgb_metric_log_line = ""
        yuv_metric_log_line = ""
        nimages = len(self.filenames)
        print("// Total benchmark dataset images : ", nimages)

        for i, filename in enumerate(self.filenames):
            # Create a tf.Graph for repeated evaluation
            benchmarkgraph = tf.Graph()
            with benchmarkgraph.as_default():
                x = read_png(filename)
                x = tf.expand_dims(x, 0)
                x.set_shape([1, None, None, 3])
                num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)
                x_shape = tf.shape(x)
                x_img = x[0, : x_shape[1], : x_shape[2], :]

                genx, genximg, eval_bpp, encoding_tensors, gen_metadata = generator(
                    params=hparams, inputs=x, is_training=False, predict=True
                )

                # Bring both images back to 0..255 range.
                x *= 255
                yuv_x = tf.image.rgb_to_yuv(x)
                genx = tf.clip_by_value(genx, 0, 1)
                yuv_genx = tf.image.rgb_to_yuv(genx)
                genx = tf.round(genx * 255)

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
                    eval_msssim_yuv = tf.squeeze(
                        tf.image.ssim_multiscale(yuv_genx, yuv_x, 1.0)
                    )

                # Define the initialization operation
                init_op = tf.global_variables_initializer()

                session_config = tf.ConfigProto()
                session_config.gpu_options.allow_growth = self.model_args.allow_growth

                with tf.Session(config=session_config) as sess:
                    # Load the latest model checkpoint, get the compressed string and the tensor
                    # shapes.
                    sess.run(init_op)
                    # Start the queue runners. If they are not started the program will hang
                    # see e.g. https://www.tensorflow.org/programmers_guide/reading_data
                    coord = tf.train.Coordinator()
                    threads = []
                    for qr in sess.graph.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                        threads.extend(
                            qr.create_threads(
                                sess, coord=coord, daemon=True, start=True
                            )
                        )

                    # Load the pretrained model
                    print("Loading weights from the pre-trained model")
                    if hparams.use_adversarial_loss:
                        checkpoint_path = os.path.join(
                            self.model_args.model_dir, "generator"
                        )
                    else:
                        checkpoint_path = self.model_args.model_dir
                    latest = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_path)
                    ckpt_iteration = int(latest.split("-")[-1])
                    print("Variables loaded from the checkpoint")
                    print(
                        "/// The latest Checkpoint was trained for {} iterations >>".format(
                            ckpt_iteration
                        )
                    )
                    # pprint.pprint(tf.train.list_variables(latest))
                    tf.train.Saver().restore(sess, save_path=latest)

                    arrays = sess.run(encoding_tensors)
                    # Write a binary file with the shape information and the compressed string.
                    packed = PackedTensors()
                    packed.pack(encoding_tensors, arrays)
                    # Write a binary file with the shape information and the compressed string.
                    genpathtxt = os.path.join(
                        self.model_args.model_dir,
                        self.benchmarkname,
                        self.names[i],
                        "{}_comp.dcf".format(ckpt_iteration),
                    )
                    if not os.path.exists(os.path.split(genpathtxt)[0]):
                        os.makedirs(os.path.split(genpathtxt)[0])
                    with open(genpathtxt, "wb") as f:
                        f.write(packed.string)

                    # Write reconstructed image out as a PNG file.
                    genpathimg = os.path.join(
                        self.model_args.model_dir,
                        self.benchmarkname,
                        self.names[i],
                        "{}_comp.png".format(ckpt_iteration),
                    )
                    genx_save_op = write_png(genpathimg, genximg)
                    if i < 2:
                        x_save_path = os.path.join(
                            self.model_args.model_dir,
                            self.benchmarkname,
                            self.names[i],
                            "{}_gt.png".format(ckpt_iteration),
                        )
                        x_save_op = write_png(x_save_path, x_img)
                        # This op saves the ground truth image as a png file
                        sess.run(x_save_op)

                    # This op saves the generated image as a png file
                    sess.run(genx_save_op)

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
                        "Multiscale SSIM (dB): {:0.2f}".format(
                            -10 * np.log10(1 - eval_msssim)
                        )
                    )
                    print("Information content in bpp: {:0.4f}".format(eval_bpp))
                    print("Actual bits per pixel: {:0.4f}".format(bpp))

                    txtpath = os.path.join(
                        self.model_args.model_dir,
                        self.benchmarkname,
                        self.names[i],
                        "{}_comp.txt".format(ckpt_iteration),
                    )
                    if not os.path.exists(os.path.split(txtpath)[0]):
                        os.makedirs(os.path.split(txtpath)[0])

                    with open(txtpath, "w") as txtf:
                        txtf.write("Mean squared error: {:0.4f}\n".format(eval_mse))
                        txtf.write("PSNR (dB): {:0.2f}\n".format(eval_psnr))
                        txtf.write("Multiscale SSIM: {:0.4f}\n".format(eval_msssim))
                        txtf.write(
                            "Multiscale SSIM (dB): {:0.2f}\n".format(
                                -10 * np.log10(1 - eval_msssim)
                            )
                        )
                        txtf.write(
                            "Information content in bpp: {:0.4f}\n".format(eval_bpp)
                        )
                        txtf.write("Actual bits per pixel: {:0.4f}\n".format(bpp))

                    rgb_metric_log_line += " {:0.7f}, {:0.7f}, {:0.7f}, {:0.7f}, {:0.7f}, {:0.7f}".format(
                        eval_bpp, bpp, eval_mse, eval_psnr, eval_ssim, eval_msssim
                    )
                    yuv_metric_log_line += " {:0.7f}, {:0.7f}, {:0.7f}, {:0.7f}".format(
                        eval_mse_yuv, eval_psnr_yuv, eval_ssim_yuv, eval_msssim_yuv
                    )
                    # Write compression metrics to log
                    log_path = os.path.join(
                        self.model_args.model_dir, self.benchmarkname
                    )
                    with open(log_path + "/compression_metrics.csv", "a") as f:
                        f.write(
                            "%d, %s, %s,\n"
                            % (ckpt_iteration, rgb_metric_log_line, yuv_metric_log_line)
                        )

            print(
                "Iteration ",
                i,
                " maxrss: ",
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            )
            # gc.collect()  # use Python default garbage collector for freeing the Graph
            tf.keras.backend.clear_session()
