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

"""Wrapper for image datasets with typical pre-processing."""

from typing import Any, Mapping, Text, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from detcon.datasets import dataset_adapter
from detcon.datasets import imagenet_with_fh

Split = dataset_adapter.Split
Batch = dataset_adapter.Batch
Split = dataset_adapter.Split
DatasetMode = dataset_adapter.DatasetMode

_DATASET_METADATA = {
    'imagenet2012:5.*.*': {
        'num_examples': {
            Split.TRAIN: 1281167,
            Split.VALID: 0,
            Split.TEST: 50000,
        },
        'num_classes': 1000,
        'color_stats': {
            'mean_rgb': (0.485, 0.456, 0.406),
            'std_rgb': (0.229, 0.224, 0.225)
        }
    },
    'coco': {
        'num_examples': {
            Split.TRAIN: 118287,
            Split.VALID: 5000,
            Split.TEST: 20228,
        },
        'num_classes': 80,
        'color_stats': {
            'mean_rgb': (0.485, 0.456, 0.406),
            'std_rgb': (0.229, 0.224, 0.225)
        },
    },
    'jft': {
        'num_examples': {
            Split.TRAIN: 302982257,
            Split.VALID: 641676,
            Split.TEST: 990221,
        },
        'num_classes': 18291,
        'color_stats': {
            'mean_rgb': (0.485, 0.456, 0.406),
            'std_rgb': (0.229, 0.224, 0.225)
        },
    }
}

_DATASET_METADATA['coco_fh'] = _DATASET_METADATA['coco']


class ImageDataset(dataset_adapter.ImageDatasetAdapter):
  """Adapter for a dataset."""

  def __init__(self,
               dataset_directory: Text,
               dataset_name: Text,
               enable_double_transpose: bool,
               allow_caching: bool,
               preprocessing_config: Mapping[Text, Mapping[Text, Any]],
               use_tfds: bool):
    assert dataset_name in _DATASET_METADATA
    super().__init__(
        dataset_directory=dataset_directory,
        dataset_name=dataset_name,
        enable_double_transpose=enable_double_transpose,
        preprocessing_config=preprocessing_config,
        allow_caching=allow_caching,
        use_tfds=use_tfds)
    self._color_stats = _DATASET_METADATA[self._dataset_name]['color_stats']
    metadata = _DATASET_METADATA[self._dataset_name]
    self._num_examples = metadata['num_examples']
    self._use_fh = 'fh' in dataset_name or 'coco' not in dataset_name

  def _to_tfds_split(self, split: Split) -> tfds.Split:
    """Returns the TFDS split appropriately sharded."""
    return {
        Split.TRAIN: tfds.Split.TRAIN,
        Split.VALID: tfds.Split.VALIDATION,
        Split.TEST: tfds.Split.TEST
    }[split]

  def num_examples(self, split: Split) -> int:
    return self._num_examples[split]

  @property
  def num_classes(self) -> int:
    return _DATASET_METADATA[self._dataset_name]['num_classes']

  def _shard(self,
             split: Split,
             shard_index: int,
             num_shards: int) -> Tuple[int, int]:
    """Returns [start, end) for the given shard index."""
    assert shard_index < num_shards
    arange = np.arange(self.num_examples(split))
    shard_range = np.array_split(arange, num_shards)[shard_index]
    start, end = shard_range[0], (shard_range[-1] + 1)
    return start, end

  def normalize_images(self, images: jnp.ndarray) -> jnp.ndarray:
    if self._dataset_name.lower().startswith('imagenet'):
      mean_rgb = self._color_stats['mean_rgb']
      stddev_rgb = self._color_stats['std_rgb']
      normed_images = images - jnp.array(mean_rgb).reshape((1, 1, 1, 3))
      stddev = jnp.array(stddev_rgb).reshape((1, 1, 1, 3))
      normed_images = normed_images / stddev
    else:
      normed_images = images
    return normed_images

  def _load(self, split: Split) -> tf.data.Dataset:
    """Loads the given split of the dataset."""
    if self._dataset_name.lower().startswith('imagenet'):
      if self._use_tfds:
        start, end = self._shard(split, jax.process_index(),
                                 jax.process_count())
        tfds_split = tfds.core.ReadInstruction(
            self._to_tfds_split(split), from_=start, to=end, unit='abs')
        ds = tfds.load(
            self._dataset_name,
            split=tfds_split,
            decoders={'image': tfds.decode.SkipDecoding()})
      else:
        ds = imagenet_with_fh.load_dataset(
            self._dataset_directory, split,
            jax.process_index(), jax.process_count())
    else:
      raise ValueError
    return ds
