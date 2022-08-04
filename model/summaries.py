"""Utility file for visualizing generated images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import io
from six.moves import range
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
import tensorflow.compat.v1 as tf


__all__ = ["image_grid", "image_reshaper", "python_image_grid"]


# TODO(joelshor): Make this a special case of `image_reshaper`.
def image_grid(input_tensor, grid_shape, image_shape=(32, 32), num_channels=3):
    """
    Arrange a minibatch of images into a grid to form a single image.

    Args:
        input_tensor: Tensor. Minibatch of images to format, either 4D
            ([batch size, height, width, num_channels]) or flattened
            ([batch size, height * width * num_channels]).
        grid_shape: Sequence of int. The shape of the image grid,
            formatted as [grid_height, grid_width].
        image_shape: Sequence of int. The shape of a single image,
            formatted as [image_height, image_width].
        num_channels: int. The number of channels in an image.
    Returns:
        Tensor representing a single image in which the input images have been
        arranged into a grid.
    Raises:
        ValueError: The grid shape and minibatch size don't match, or the image
            shape and number of channels are incompatible with the input tensor.
    """
    if grid_shape[0] * grid_shape[1] != int(input_tensor.shape[0]):
        raise ValueError(
            "Grid shape %s incompatible with minibatch size %i."
            % (grid_shape, int(input_tensor.shape[0]))
        )
    if len(input_tensor.shape) == 2:
        num_features = image_shape[0] * image_shape[1] * num_channels
        if int(input_tensor.shape[1]) != num_features:
            raise ValueError(
                "Image shape and number of channels incompatible with " "input tensor."
            )
    elif len(input_tensor.shape) == 4:
        if (
            int(input_tensor.shape[1]) != image_shape[0]
            or int(input_tensor.shape[2]) != image_shape[1]
            or int(input_tensor.shape[3]) != num_channels
        ):
            raise ValueError(
                "Image shape and number of channels incompatible with "
                "input tensor. %s vs %s"
                % (input_tensor.shape, (image_shape[0], image_shape[1], num_channels))
            )
    else:
        raise ValueError("Unrecognized input tensor format.")
    height, width = grid_shape[0] * image_shape[0], grid_shape[1] * image_shape[1]
    input_tensor = tf.reshape(
        input_tensor, tuple(grid_shape) + tuple(image_shape) + (num_channels,)
    )
    input_tensor = tf.transpose(a=input_tensor, perm=[0, 1, 3, 2, 4])
    input_tensor = tf.reshape(
        input_tensor, [grid_shape[0], width, image_shape[0], num_channels]
    )
    input_tensor = tf.transpose(a=input_tensor, perm=[0, 2, 1, 3])
    input_tensor = tf.reshape(input_tensor, [1, height, width, num_channels])
    return input_tensor


def python_image_grid(input_array, grid_shape):
    """
    This is a pure python version of tfgan.eval.image_grid.
    Args:
        input_array: ndarray. Minibatch of images to format. A 4D numpy array
            ([batch size, height, width, num_channels]).
        grid_shape: Sequence of int. The shape of the image grid,
            formatted as [grid_height, grid_width].
    Returns:
        Numpy array representing a single image in which the input images have been
        arranged into a grid.
    Raises:
        ValueError: The grid shape and minibatch size don't match.
        ValueError: The input array isn't 4D.
    """
    if grid_shape[0] * grid_shape[1] != int(input_array.shape[0]):
        raise ValueError(
            "Grid shape %s incompatible with minibatch size %i."
            % (grid_shape, int(input_array.shape[0]))
        )
    if len(input_array.shape) != 4:
        raise ValueError("Unrecognized input array format.")
    image_shape = input_array.shape[1:3]
    num_channels = input_array.shape[3]
    height, width = (grid_shape[0] * image_shape[0], grid_shape[1] * image_shape[1])
    input_array = np.reshape(
        input_array, tuple(grid_shape) + tuple(image_shape) + (num_channels,)
    )
    input_array = np.transpose(input_array, [0, 1, 3, 2, 4])
    input_array = np.reshape(
        input_array, [grid_shape[0], width, image_shape[0], num_channels]
    )
    input_array = np.transpose(input_array, [0, 2, 1, 3])
    input_array = np.reshape(input_array, [height, width, num_channels])
    return input_array


def _validate_images(images):
    for img in images:
        img.shape.assert_has_rank(3)
        img.shape.assert_is_fully_defined()
        if img.shape[-1] not in (1, 3):
            raise ValueError("image_reshaper only supports 1 or 3 channel images.")


def _assert_is_image(data):
    data.shape.assert_has_rank(4)
    data.shape[1:].assert_is_fully_defined()


# TODO(joelshor): Move the dimension logic from Python to Tensorflow.
def image_reshaper(images, num_cols=None):
    """
    A reshaped summary image.

    Returns an image that will contain all elements in the list and will be
    laid out in a nearly-square tiling pattern (e.g. 11 images will lead to a
    3x4 tiled image).

    Args:
        images: Image data to summarize. Can be an RGB or grayscale image, a list of
             such images, or a set of RGB images concatenated along the depth
             dimension. The shape of each image is assumed to be [batch_size,
             height, width, depth].
        num_cols: (Optional) If provided, this is the number of columns in the final
             output image grid. Otherwise, the number of columns is determined by
             the number of images.
    Returns:
        A summary image matching the input with automatic tiling if needed.
        Output shape is [1, height, width, channels].
    """
    if isinstance(images, tf.Tensor):
        images = tf.unstack(images)
    _validate_images(images)

    num_images = len(images)
    num_columns = num_cols if num_cols else int(math.ceil(math.sqrt(num_images)))
    num_rows = int(math.ceil(float(num_images) / num_columns))
    rows = [images[x : x + num_columns] for x in range(0, num_images, num_columns)]

    # Add empty image tiles if the last row is incomplete.
    num_short = num_rows * num_columns - num_images
    assert 0 <= num_short < num_columns
    if num_short > 0:
        rows[-1].extend([tf.zeros_like(images[-1])] * num_short)

    # Convert each row from a list of tensors to a single tensor.
    rows = [tf.concat(row, 1) for row in rows]

    # Stack rows vertically.
    img = tf.concat(rows, 0)

    return tf.expand_dims(img, 0)


def add_image_grid_summaries(real_image_batch, gen_image_batch, grid_size):
    """
    Add image summmaries for real and generated images

    Args:
        :param real_image_batch: (tf.Tensor) input image tf.Tensor for the model
        :param gen_image_batch: (tf.Tensor) generated image tf.Tensor from the model
        :param grid_size_batch: (tuple) The size of the image grid
    Raises:
        :raises: ValueError: If real and generated data aren't images.
    """
    _assert_is_image(real_image_batch)
    _assert_is_image(gen_image_batch)

    num_images = grid_size[1] * grid_size[2]
    real_image_shape = real_image_batch.shape.as_list()[1:3]
    gen_image_shape = gen_image_batch.shape.as_list()[1:3]
    real_image_channels = gen_image_batch.shape.as_list()[3]
    gen_image_channels = gen_image_batch.shape.as_list()[3]

    summary_op = [
        tf.compat.v1.summary.image(
            "original",
            image_grid(
                real_image_batch,
                grid_shape=(grid_size[0], grid_size[1]),
                image_shape=real_image_shape,
                num_channels=real_image_channels,
            ),
            max_outputs=1,
        ),
        tf.compat.v1.summary.image(
            "reconstruction",
            image_grid(
                gen_image_batch,
                grid_shape=(grid_size[0], grid_size[1]),
                image_shape=gen_image_shape,
                num_channels=gen_image_channels,
            ),
            max_outputs=1,
        ),
    ]

    return summary_op


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def visualize_yuv(images, params):
    """Visualize tensorflow YUV image tensor using OpenCV
    and matplotlib utilities

    Arguments:
        image {np.ndarray} -- Input numpy image tensor
    """
    # colormap for 'U' and 'V' of YUV colorspace
    lut_u = np.array([[[i, 255 - i, 0] for i in range(256)]], dtype=np.uint8)
    lut_v = np.array([[[0, 255 - i, i] for i in range(256)]], dtype=np.uint8)

    def _process_yuv(images):
        batch_imgs = []
        for img in images:
            y, u, v = cv2.split(img)
            # Convert back to BGR so we can apply the LUT and stack the images
            y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
            u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
            v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
            u_mapped = cv2.LUT(u, lut_u)
            v_mapped = cv2.LUT(v, lut_v)
            y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
            u_mapped = cv2.cvtColor(u_mapped, cv2.COLOR_BGR2RGB)
            v_mapped = cv2.cvtColor(v_mapped, cv2.COLOR_BGR2RGB)
            result = np.hstack([img, y, u_mapped, v_mapped])
            batch_imgs.append(result)
        return [batch_imgs]

    batch_imgs = tf.py_func(_process_yuv, [images], [tf.uint8], stateful=False, name='process_yuv')
    batch_imgs = tf.squeeze(tf.stack(batch_imgs))
    batch_imgs.set_shape([params.batchsize] + [params.patchsize, params.patchsize])

    return batch_imgs


def toLABplot(image, input_type="BGR"):
    """Visualize 'L*a*b' colorspace of an input image

    Arguments:
        image {np.ndarray} -- numpy image array

    Keyword Arguments:
        input_type {str} -- either of "RGB" or "BGR" (default: {"BGR"})

    Returns:
        matplotlib plot -- 3D plot of image array's LAB colorspace
    """
    conversion = cv2.COLOR_BGR2LAB if input_type == "BGR" else cv2.COLOR_RGB2LAB
    image_LAB = cv2.cvtColor(image, conversion)

    y, x, z = image_LAB.shape
    LAB_flat = np.reshape(image_LAB, [y * x, z])

    colors = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if input_type == "BGR" else image
    colors = np.reshape(colors, [y * x, z]) / 255.0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        xs=LAB_flat[:, 2], ys=LAB_flat[:, 1], zs=LAB_flat[:, 0], s=10, c=colors, lw=0
    )
    ax.set_xlabel("A")
    ax.set_ylabel("B")
    ax.set_zlabel("L")

    plt.show()

    return image_LAB

