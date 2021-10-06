# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ImageNet dataset with typical pre-processing."""

from typing import Optional

import tensorflow.compat.v2 as tf


def is_jpeg(image_bytes):
  return tf.equal(tf.strings.substr(image_bytes, 0, 4), b'\xff\xd8\xff\xe0')


decode_crop_jpeg = tf.io.decode_and_crop_jpeg


def decode_crop_nonjpeg(image_bytes, crop_window):
  return tf.image.crop_to_bounding_box(
      tf.io.decode_image(image_bytes, channels=3), crop_window[0],
      crop_window[1], crop_window[2], crop_window[3])


def clip_or_pad_to_fixed_size(input_tensor, size, constant_values=0):
  """Pads data to a fixed length at the first dimension.

  Args:
    input_tensor: `Tensor` with any dimension.
    size: `int` number for the first dimension of output Tensor.
    constant_values: `int` value assigned to the paddings.

  Returns:
    `Tensor` with the first dimension padded to `size`.
  """
  input_shape = input_tensor.get_shape().as_list()
  padding_shape = []

  # Computes the padding length on the first dimension, clip input tensor if it
  # is longer than `size`.
  input_length = tf.shape(input_tensor)[0]
  input_length = tf.clip_by_value(input_length, 0, size)
  input_tensor = input_tensor[:input_length]

  padding_length = tf.maximum(0, size - input_length)
  padding_shape.append(padding_length)

  # Copies shapes of the rest of input shape dimensions.
  for i in range(1, len(input_shape)):
    padding_shape.append(tf.shape(input_tensor)[i])

  # Pads input tensor to the fixed first dimension.
  paddings = tf.cast(constant_values * tf.ones(padding_shape),
                     input_tensor.dtype)
  padded_tensor = tf.concat([input_tensor, paddings], axis=0)
  output_shape = input_shape
  output_shape[0] = size
  padded_tensor.set_shape(output_shape)
  return padded_tensor


def preprocess_image(
    image_bytes: tf.Tensor,
    image_segmentation: tf.Tensor,
    spatial_crop: str,
    output_image_size: int,
    random_flip_left_right: bool,
) -> tf.Tensor:
  """Returns processed and resized images and segmentation (if not None)."""
  if spatial_crop == 'random':
    crop_window = get_random_crop_window(image_bytes)
    # Random horizontal flipping is optionally done in augmentations.preprocess.
  elif spatial_crop == 'center':
    crop_window = get_center_crop_window(image_bytes,
                                         output_image_size=output_image_size)
  else:
    raise ValueError(f'Unknown spatial crop mode: {spatial_crop}')

  image = tf.cond(
      is_jpeg(image_bytes),
      true_fn=lambda: decode_crop_jpeg(image_bytes, crop_window, channels=3),
      false_fn=lambda: decode_crop_nonjpeg(image_bytes, crop_window)
      )

  output_segmentation = {}
  if image_segmentation is not None:
    for name, mask in image_segmentation.items():
      output_segmentation[name] = tf.image.crop_to_bounding_box(
          mask, crop_window[0], crop_window[1], crop_window[2], crop_window[3])

  if random_flip_left_right:
    flip_sample = tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32)
    flip = tf.less(flip_sample, 0.5)
    image = tf.cond(flip,
                    lambda: tf.image.flip_left_right(image),
                    lambda: image)

    # pylint: disable=undefined-loop-variable
    for name, mask in output_segmentation.items():
      output_segmentation[name] = tf.cond(
          flip, lambda: tf.image.flip_left_right(mask), lambda: mask)
    # pylint: enable=undefined-loop-variable

  assert image.dtype == tf.uint8
  image = tf.image.resize(
      image, [output_image_size, output_image_size],
      tf.image.ResizeMethod.BICUBIC)
  # Clamp overshoots outside the range [0.0, 255.0] caused by interpolation
  image = tf.clip_by_value(image / 255., 0., 1.)

  for name, mask in output_segmentation.items():
    output_segmentation[name] = tf.image.resize(
        mask, [output_image_size, output_image_size],
        tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return image, output_segmentation


def get_random_crop_window(image_bytes: tf.Tensor) -> tf.Tensor:
  """Makes a random crop."""
  img_size = tf.cond(
      is_jpeg(image_bytes),
      true_fn=lambda: tf.image.extract_jpeg_shape(image_bytes),
      false_fn=lambda: tf.shape(tf.image.decode_image(image_bytes, channels=3)))
  area = tf.cast(img_size[1] * img_size[0], tf.float32)
  target_area = tf.random.uniform([], 0.08, 1.0, dtype=tf.float32) * area

  log_ratio = (tf.math.log(3 / 4), tf.math.log(4 / 3))
  aspect_ratio = tf.math.exp(
      tf.random.uniform([], *log_ratio, dtype=tf.float32))

  w = tf.cast(tf.round(tf.sqrt(target_area * aspect_ratio)), tf.int32)
  h = tf.cast(tf.round(tf.sqrt(target_area / aspect_ratio)), tf.int32)

  w = tf.minimum(w, img_size[1])
  h = tf.minimum(h, img_size[0])

  offset_w = tf.random.uniform((),
                               minval=0,
                               maxval=img_size[1] - w + 1,
                               dtype=tf.int32)
  offset_h = tf.random.uniform((),
                               minval=0,
                               maxval=img_size[0] - h + 1,
                               dtype=tf.int32)

  crop_window = tf.stack([offset_h, offset_w, h, w])
  return crop_window


def extract_nonjpeg_shape(image_bytes):
  return tf.shape(tf.image.decode_image(image_bytes, channels=3))


def get_center_crop_window(
    image_bytes: tf.Tensor,
    jpeg_shape: Optional[tf.Tensor] = None,
    output_image_size: int = 224,
) -> tf.Tensor:
  """Crops to center of image with padding then scales."""

  if jpeg_shape is None:
    jpeg_shape = tf.cond(
        is_jpeg(image_bytes),
        true_fn=lambda: tf.image.extract_jpeg_shape(image_bytes),
        false_fn=lambda: extract_nonjpeg_shape(image_bytes))

  image_height = jpeg_shape[0]
  image_width = jpeg_shape[1]

  padded_center_crop_size = tf.cast(
      ((output_image_size / (output_image_size + 32)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)), tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([
      offset_height, offset_width, padded_center_crop_size,
      padded_center_crop_size
  ])
  return crop_window
