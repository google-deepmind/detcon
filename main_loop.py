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

"""Training loop for the primary DetCon-B pretraining experiment."""

import time
from typing import Type

from absl import app
from absl import flags
from absl import logging
import chex
import jax
import ml_collections
import numpy as np

from detcon import detcon_b
from detcon import detcon_b_config

flags.DEFINE_string('worker_mode', 'train', 'The mode, train or eval')
flags.DEFINE_string('worker_tpu_driver', '', 'The tpu driver to use')
flags.DEFINE_integer('pretrain_epochs', 1000, 'Number of pre-training epochs')
flags.DEFINE_integer('batch_size', 4096, 'Total batch size')
flags.DEFINE_integer('log_tensors_interval', 60, 'Print stats every n seconds')
flags.DEFINE_string('dataset_directory', '/tmp/imagenet-fh-train',
                    'Local directory with generated FH-ImageNet dataset')
flags.DEFINE_bool('is_test', False, 'Run the sanity test (pretrain for 1 step)')

FLAGS = flags.FLAGS


Experiment = Type[detcon_b.PretrainExperiment]


def train_loop(experiment_class: Experiment, config: ml_collections.ConfigDict):
  """The main training loop.

  Args:
    experiment_class: the constructor for the experiment (either byol_experiment
    or eval_experiment).
    config: the experiment config.
  """
  experiment = experiment_class(**config.experiment_kwargs, mode='train')
  logging.info('Setup pre-training experiment class!')
  rng = jax.random.PRNGKey(0)
  step = 0

  host_id = jax.host_id()
  last_logging = time.time()

  local_device_count = jax.local_device_count()
  while step < config['training_steps']:
    step_rng, rng = tuple(jax.random.split(rng))
    # Broadcast the random seeds across the devices
    step_rng_device = jax.random.split(step_rng, num=jax.device_count())
    step_rng_device = step_rng_device[
        host_id * local_device_count:(host_id + 1) * local_device_count]
    step_device = np.broadcast_to(step, [local_device_count])
    logging.info('Setup RNGs!')

    # Perform a training step and get scalars to log.
    scalars = experiment.step(global_step=step_device, rng=step_rng_device,
                              writer=None)
    logging.info('Finished pre-training step!')

    # Logging metrics
    current_time = time.time()
    if current_time - last_logging > FLAGS.log_tensors_interval:
      logging.info('Step %d: %s', step, scalars)
      last_logging = current_time
    step += 1
  logging.info('Step %d: %s', step, scalars)


def main(_):
  if FLAGS.is_test:
    fake_pmap = chex.fake_pmap()
    fake_pmap.start()
  if FLAGS.worker_tpu_driver:
    jax.config.update('jax_xla_backend', 'tpu_driver')
    jax.config.update('jax_backend_target', FLAGS.worker_tpu_driver)
    logging.info('Backend: %s %r', FLAGS.worker_tpu_driver, jax.devices())

  experiment_class = detcon_b.PretrainExperiment
  config = detcon_b_config.get_config(FLAGS.pretrain_epochs, FLAGS.batch_size)
  loader_kwargs = config['experiment_kwargs']['config']['data']['loader_kwargs']
  loader_kwargs['dataset_directory'] = FLAGS.dataset_directory

  if FLAGS.is_test:
    config['experiment_kwargs']['config']['mock_out_train_dataset'] = True
    config['training_steps'] = 1
    config['log_train_data_interval'] = None
    config['save_checkpoint_interval'] = None
    config['experiment_kwargs']['config']['training']['max_steps'] = 1
    config['experiment_kwargs']['config']['training']['batch_size'] = 4

  train_loop(experiment_class, config)
  if FLAGS.is_test:
    fake_pmap.stop()
  logging.info('Finished running training loop!')


if __name__ == '__main__':
  app.run(main)
