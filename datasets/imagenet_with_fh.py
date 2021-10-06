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

"""ImageNet with associated Felzenzwalb-Huttenlocher masks for each image."""

import os
import numpy as np
import tensorflow.compat.v2 as tf
from detcon.datasets import dataset_adapter


def _parse_example_proto(value):
  """Parse an Imagenet record from value."""
  keys_to_features = {
      "image/encoded":
          tf.io.FixedLenFeature((), tf.string, default_value=""),
      "image/class/label":
          tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
      "image/felzenszwalb_segmentations":
          tf.io.VarLenFeature(tf.string),
  }

  parsed = tf.io.parse_single_example(value, keys_to_features)
  fs_segmentation = parsed["image/felzenszwalb_segmentations"]
  fs_segmentation = tf.sparse.to_dense(fs_segmentation, default_value="")
  fs_segmentation = tf.io.decode_raw(fs_segmentation, tf.uint8)

  return {
      "image": parsed["image/encoded"],
      # ImageNet labels start at 1 but we want to start at 0.
      "label": tf.cast(parsed["image/class/label"], tf.int64) - 1,
      "felzenszwalb_segmentations": fs_segmentation,
  }


def check_train_dataset_directory(dataset_directory):
  basename_glob_expression = "train-*-of-02048"
  paths = tf.io.gfile.glob(
      os.path.join(dataset_directory, basename_glob_expression))
  assert len(paths) >= 2048, ("Could not find 2048 TFRecord shards in data"
                              "directory.")


def load_dataset(data_dir, split, shard_index=0, num_shards=1):
  """Load dataset that reads from ImageNet-FH TFRecords."""
  assert shard_index < num_shards
  if split == dataset_adapter.Split.TRAIN:
    shard_range = np.array_split(np.arange(2048), num_shards)[shard_index]
    start, end = shard_range[0], (shard_range[-1] + 1)
    files = [f"train-{i:05}-of-02048" for i in range(start, end)]
    paths = [os.path.join(data_dir, file) for file in files]
  elif split == dataset_adapter.Split.TEST:
    basename_glob_expression = "validation-*-of-00256"
    paths = tf.io.gfile.glob(
        os.path.join(data_dir, basename_glob_expression))
  else:
    raise ValueError(f"No such split exists: {split}.")

  assert paths, os.path.join(data_dir, basename_glob_expression)

  ds = tf.data.Dataset.from_tensor_slices(paths)
  ds = ds.interleave(
      tf.data.TFRecordDataset,
      cycle_length=tf.data.experimental.AUTOTUNE,
      block_length=1,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )
  ds = ds.map(
      _parse_example_proto,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )
  ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  options = tf.data.Options()
  options.experimental_optimization.parallel_batch = True
  options.experimental_optimization.map_parallelization = True
  options.experimental_deterministic = False
  ds = ds.with_options(options)

  return ds
