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

import abc
import enum
from typing import Any, Generator, Mapping, Sequence
from absl import logging

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from detcon.utils import tf_image_ops

Batch = Mapping[str, np.ndarray]
JaxBatch = Mapping[str, jnp.ndarray]
TFBatch = Mapping[str, tf.Tensor]


class DatasetMode(enum.Enum):
  """Loading modes for the dataset."""
  PRETRAIN = 1  # Generates two augmented views (random crop + augmentations).
  LINEAR_TRAIN = 2  # Generates a single random crop.
  EVAL = 3  # Generates a single center crop.
  SEGMENT = 4  # Generates a single random crop with accompanying masks


class Split(enum.Enum):
  """Imagenet dataset split."""
  TRAIN = 1
  VALID = 2
  TEST = 3

  @classmethod
  def from_string(cls, name: str) -> 'Split':
    return {
        'TRAIN': Split.TRAIN,
        'VALID': Split.VALID,
        'VALIDATION': Split.VALID,
        'TEST': Split.TEST
    }[name.upper()]


class DatasetAdapter(metaclass=abc.ABCMeta):
  """Adapter for a dataset."""

  def __init__(self,
               dataset_directory: str,
               dataset_name: str,
               enable_double_transpose: bool,
               allow_caching: bool,
               preprocessing_config: Mapping[str, Mapping[str, Any]],
               use_tfds: bool):
    self._dataset_directory = dataset_directory
    self._dataset_name = dataset_name
    self._preprocessing_config = preprocessing_config
    self._use_double_transpose = (
        enable_double_transpose and jax.local_devices()[0].platform == 'tpu')
    self._allow_caching = allow_caching
    self._use_tfds = use_tfds
    if not self._use_tfds:
      assert not self._allow_caching
    self._use_fh = False

  @abc.abstractmethod
  def num_examples(self, split: Split) -> int:
    """Returns the number of examples for a given split."""

  @abc.abstractproperty
  def num_classes(self) -> int:
    """Returns the number of classes, used for the classifier."""

  @property
  def num_train_examples(self) -> int:
    return self.num_examples(Split.TRAIN)

  def normalize_images(self, images: jnp.ndarray) -> jnp.ndarray:
    return images

  @abc.abstractmethod
  def _transpose_for_h2d(self, batch: TFBatch) -> TFBatch:
    """Transposes images for a batch of data."""
    # We use the double-transpose-trick to improve performance for TPUs. Note
    # that this (typically) requires a matching HWCN->NHWC transpose in your
    # model code. The compiler cannot make this optimization for us since our
    # data pipeline and model are compiled separately.

  def load(
      self,
      split: Split,
      *,
      dataset_mode: DatasetMode,
      batch_dims: Sequence[int]) -> Generator[Batch, None, None]:
    """A generator that returns Batches.

    Args:
      split: The split to load.
      dataset_mode: How to preprocess the data.
      batch_dims: The number of batch dimensions.

    Yields:
      Batches containing keys:
      - (view1, view2, labels, masks) if preprocess_mode is PRETRAIN;
      - (images, labels) if preprocess_mode is EVAL or LINEAR_TRAIN;
      - (images, labels, masks) if preprocess_mode is SEGMENT.
    """
    if (dataset_mode is DatasetMode.EVAL and
        self.num_examples(split) % np.prod(batch_dims) != 0):
      raise ValueError(f'Test/valid must be divisible by {np.prod(batch_dims)}')
    ds = self._wrap(self._load(split), dataset_mode, batch_dims)
    logging.info('Constructed dataset:')
    logging.info(ds)
    yield from tfds.as_numpy(ds)

  @abc.abstractmethod
  def _load(self, split: Split) -> tf.data.Dataset:
    """Returns a TF dataset for the correct split."""

  @abc.abstractmethod
  def _preprocess_pretrain(self, example: TFBatch) -> TFBatch:
    pass

  @abc.abstractmethod
  def _preprocess_linear_train(self, example: TFBatch) -> TFBatch:
    pass

  @abc.abstractmethod
  def _preprocess_segment(self, example: TFBatch) -> TFBatch:
    pass

  @abc.abstractmethod
  def _preprocess_eval(self, example: TFBatch) -> TFBatch:
    pass

  def _wrap(
      self,
      ds: tf.data.Dataset,
      dataset_mode: DatasetMode,
      batch_dims: Sequence[int]) -> tf.data.Dataset:
    """Wraps a TF dataset with the correct preprocessing."""

    total_batch_size = np.prod(batch_dims)

    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = 48
    options.experimental_threading.max_intra_op_parallelism = 1
    ds = ds.with_options(options)

    if dataset_mode is not DatasetMode.EVAL:
      options.experimental_deterministic = False
      if jax.process_count() > 1 and self._allow_caching:
        # Only cache if we are reading a subset of the dataset.
        ds = ds.cache()
      ds = ds.repeat()
      ds = ds.shuffle(buffer_size=10 * total_batch_size, seed=0)

    if dataset_mode is DatasetMode.PRETRAIN:
      ds = ds.map(
          self._preprocess_pretrain,
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif dataset_mode is DatasetMode.LINEAR_TRAIN:
      ds = ds.map(
          self._preprocess_linear_train,
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif dataset_mode is DatasetMode.SEGMENT:
      ds = ds.map(
          self._preprocess_segment,
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
      ds = ds.map(
          self._preprocess_eval,
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    for i, batch_size in enumerate(reversed(batch_dims)):
      ds = ds.batch(batch_size)
      if i == 0 and self._use_double_transpose:
        ds = ds.map(self._transpose_for_h2d)

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds

  @abc.abstractmethod
  def maybe_transpose_on_device(self, batch: JaxBatch) -> JaxBatch:
    """Transpose images for TPU training.."""
    pass

  def _tf_transpose_helper(
      self, batch: TFBatch, transpose_order: Sequence[int]) -> TFBatch:
    new_batch = dict(batch)
    if 'images' in batch:
      new_batch['images'] = tf.transpose(batch['images'], transpose_order)
    else:
      new_batch['view1'] = tf.transpose(batch['view1'], transpose_order)
      new_batch['view2'] = tf.transpose(batch['view2'], transpose_order)
    return new_batch

  def _jax_transpose_helper(
      self, batch: JaxBatch, transpose_order: Sequence[int]) -> JaxBatch:
    new_batch = dict(batch)
    if 'images' in batch:
      new_batch['images'] = jnp.transpose(batch['images'], transpose_order)
    else:
      new_batch['view1'] = jnp.transpose(batch['view1'], transpose_order)
      new_batch['view2'] = jnp.transpose(batch['view2'], transpose_order)
    return new_batch


class ImageDatasetAdapter(DatasetAdapter):
  """Adapter for a dataset containing single images."""

  def _transpose_for_h2d(self, batch: TFBatch) -> TFBatch:
    """Transposes images for a batch of data."""
    # NHWC -> HWCN
    return self._tf_transpose_helper(batch, transpose_order=(1, 2, 3, 0))

  def _get_segmentations(self, example: TFBatch) -> TFBatch:
    """Load segmentations from example."""
    image_segmentations = {}
    if self._use_fh:
      image_bytes = example['image']
      # Required because some images in COCO are PNGs
      is_jpeg = tf.equal(tf.strings.substr(image_bytes, 0, 4),
                         b'\xff\xd8\xff\xe0')
      image_shape = tf.cond(
          is_jpeg,
          true_fn=lambda: tf.image.extract_jpeg_shape(image_bytes),
          false_fn=lambda: tf.shape(  # pylint: disable=g-long-lambda
              tf.image.decode_image(image_bytes, channels=3)))
      n_fh_masks = 1
      fh_shape = tf.concat([[n_fh_masks], image_shape[:2]], axis=0)
      fh_masks = example['felzenszwalb_segmentations']
      fh_masks = tf.reshape(fh_masks, fh_shape)
      fh_masks = tf.transpose(fh_masks, perm=(1, 2, 0))
      fh_masks.set_shape([None, None, n_fh_masks])
      image_segmentations['fh'] = fh_masks

    if 'groundtruth_instance_masks' in example:
      max_gt_masks = 16
      gt_masks = example['groundtruth_instance_masks']
      gt_masks = tf_image_ops.clip_or_pad_to_fixed_size(
          gt_masks, max_gt_masks, 0)
      gt_masks = tf.transpose(gt_masks, [1, 2, 0])  # [H, W, n], n = #masks
      image_segmentations['gt'] = gt_masks

    return image_segmentations

  def _preprocess_pretrain(self, example: TFBatch) -> TFBatch:
    pretrain_config = self._preprocessing_config['pretrain']
    label = tf.cast(example['label'], tf.int32)

    if not self._use_tfds:
      image_segmentations = self._get_segmentations(example)

    view1, masks1 = tf_image_ops.preprocess_image(
        example['image'], image_segmentations, **pretrain_config)
    view2, masks2 = tf_image_ops.preprocess_image(
        example['image'], image_segmentations, **pretrain_config)

    pretrain_example = {'view1': view1, 'view2': view2, 'labels': label}

    for name, mask in masks1.items():
      pretrain_example[name + '_segmentations1'] = mask
    for name, mask in masks2.items():
      pretrain_example[name + '_segmentations2'] = mask

    return pretrain_example

  def _preprocess_segment(self, example: TFBatch) -> TFBatch:
    pretrain_config = self._preprocessing_config['linear_train']
    label = tf.cast(example['label'], tf.int32)

    if not self._use_tfds:
      image_segmentations = self._get_segmentations(example)
    else:
      image_segmentations = None

    image, masks = tf_image_ops.preprocess_image(
        example['image'], image_segmentations, **pretrain_config)

    pretrain_example = {'images': image, 'labels': label}
    for name, mask in masks.items():
      pretrain_example[name] = mask

    return pretrain_example

  def _preprocess_linear_train(self, example: TFBatch) -> TFBatch:
    preprocess_config = self._preprocessing_config['linear_train']
    image, _ = tf_image_ops.preprocess_image(
        image_bytes=example['image'],
        image_segmentation=None,
        **preprocess_config)
    label = tf.cast(example['label'], tf.int32)
    return {'images': image, 'labels': label}

  def _preprocess_eval(self, example: TFBatch) -> TFBatch:
    preprocess_config = self._preprocessing_config['eval']
    image, _ = tf_image_ops.preprocess_image(
        image_bytes=example['image'],
        image_segmentation=None,
        **preprocess_config)
    label = tf.cast(example['label'], tf.int32)
    return {'images': image, 'labels': label}

  def maybe_transpose_on_device(self, batch: JaxBatch) -> JaxBatch:
    """Transpose images for TPU training.."""
    if not self._use_double_transpose:
      return batch
    # HWCN -> NHWC
    return self._jax_transpose_helper(batch, transpose_order=(3, 0, 1, 2))
