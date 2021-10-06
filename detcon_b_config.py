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

"""Config file for pre-training with the DetCon-B experiment."""

from jaxline import base_config
from ml_collections import config_dict


def get_config(num_epochs: int = 100, train_batch_size: int = 4096):
  """Return config object, containing all hyperparameters for training."""
  config = base_config.get_base_config()
  config.eval_modes = ()

  dataset_name = 'imagenet2012:5.*.*'
  evaluation_subset = 'test'
  fh_mask_to_use = 0
  train_images_per_epoch = 1281167
  warmup_epochs = int(num_epochs // 100)
  warmup_steps = (warmup_epochs * train_images_per_epoch) // train_batch_size
  max_steps = (num_epochs * train_images_per_epoch) // train_batch_size

  config.encoder_type = 'ResNet50'
  model_config = dict(
      mlp_hidden_size=4096,
      projection_size=256,
      encoder='ResNet50',
      encoder_width_multiplier=1,
      encoder_use_v2=False,
      fh_mask_to_use=fh_mask_to_use,
      norm_config={
          'decay_rate': .9,  # BN specific
          'eps': 1e-5,
          # Accumulate batchnorm statistics across devices.
          # This should be equal to the `axis_name` argument passed
          # to jax.pmap.
          'cross_replica_axis': 'i',  # BN specific
          'create_scale': True,
          'create_offset': True,
      })

  # Experiment config.
  config.experiment_kwargs = config_dict.ConfigDict(
      dict(
          config=dict(
              mock_out_train_dataset=False,
              training_mode='self-supervised',
              random_seed=0,
              model=model_config,
              optimizer=dict(
                  name='lars',
                  base_learning_rate=0.2,
                  scale_by_batch=True,
                  warmup_epochs=10,
                  warmup_steps=warmup_steps,
                  lars_kwargs={
                      'weight_decay': 1.5e-6,
                      'eta': 1e-3,
                      'momentum': .9,
                  }),
              training=dict(
                  batch_size=train_batch_size,
                  images_per_epoch=train_images_per_epoch,
                  base_target_ema=.996,
                  max_steps=max_steps,
                  num_epochs=num_epochs,
                  nce_loss_temperature=0.1,
              ),
              data=dict(
                  loader_kwargs=dict(
                      dataset_directory=None,
                      dataset_name=dataset_name,
                      enable_double_transpose=True,
                      allow_caching=False,
                      use_tfds=False,
                      preprocessing_config=dict(
                          pretrain=dict(
                              spatial_crop='random',
                              output_image_size=224,
                              random_flip_left_right=True,),
                          eval=dict(
                              spatial_crop='center',
                              output_image_size=224,
                              random_flip_left_right=True,),
                      ),
                  ),
                  training_subset='train',
                  evaluation_subset=evaluation_subset,
              ),
              evaluation=dict(
                  batch_size=100,
              ),
          ),))

  # Training loop config.
  config.training_steps = max_steps
  config.log_train_data_interval = 60
  config.log_tensors_interval = 60
  config.save_checkpoint_interval = 300
  config.eval_specific_checkpoint_dir = ''

  # Prevents accidentally setting keys that aren't recognized (e.g. in tests).
  config.lock()

  return config
