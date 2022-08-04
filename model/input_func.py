import os
import tensorflow.compat.v1 as tf

from dnc.utils import read_png


# When using dataset.prefetch, use buffer_size=None to let it detect optimal buffer size
def input_fn(mode, params):
    data_dir = {
        tf.estimator.ModeKeys.TRAIN: params.train_data_dir,
        tf.estimator.ModeKeys.EVAL: params.eval_data_dir,
    }[mode]

    data_files = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".png")
    ]

    def _read_imgfiles(src_file, tgt_file):
        src_image = read_png(src_file)
        tgt_image = read_png(tgt_file)
        return src_image, tgt_image

    def _crop_images(src_image, tgt_image):
        src_image = tf.random_crop(src_image, [params.patchsize, params.patchsize, 3])
        tgt_image = src_image
        return src_image, tgt_image

    dataset = tf.data.Dataset.from_tensor_slices((data_files, data_files))
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(
            buffer_size=2 * len(data_files), reshuffle_each_iteration=True
        )
        dataset = dataset.repeat(None)
    dataset = dataset.map(_read_imgfiles, num_parallel_calls=params.num_parallel_calls)
    dataset = dataset.map(_crop_images, num_parallel_calls=params.num_parallel_calls)
    dataset = dataset.batch(params.batchsize)
    dataset = dataset.prefetch(buffer_size=None)
    dataset = dataset.map(
        map_func=lambda src_image, tgt_image: (
            {"source": src_image},
            {"target": tgt_image},
        ),
        num_parallel_calls=params.num_parallel_calls,
    )

    return dataset.make_one_shot_iterator().get_next()


def predict_input_fn():
    input_tensor = tf.placeholder(
        dtype=tf.float32, shape=[None, None, None, 3], name="input_tensor"
    )
    features = {"source": input_tensor}
    return tf.estimator.export.ServingInputReceiver(
        features=features,
        receiver_tensors={
            tf.saved_model.signature_constants.PREDICT_INPUTS: input_tensor
        },
    )
