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

"""Common experiment components for pre-training."""

import abc
from typing import Any, Dict, Generator, Mapping, NamedTuple, Optional, Text, Tuple

from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
from jaxline import experiment
from jaxline import utils
import ml_collections
import numpy as np
import optax

from detcon.datasets import image_dataset
from detcon.datasets import imagenet_with_fh
from detcon.utils import helpers as byol_helpers
from detcon.utils import optimizers as lars_optimizer
from detcon.utils import schedules

# Type declarations.
LogsDict = Dict[Text, jnp.ndarray]


class ExperimentState(NamedTuple):
  """Byol's model and optimization parameters and state."""
  online_params: hk.Params
  target_params: hk.Params
  online_state: hk.State
  target_state: hk.State
  opt_state: lars_optimizer.LarsState


class BaseExperiment(experiment.AbstractExperiment):
  """Common training and evaluation component definition."""

  # Holds a map from object properties that will be checkpointed to their name
  # within a checkpoint. Currently it is assume that these are all sharded
  # device arrays.
  CHECKPOINT_ATTRS = {'_exp_state': 'exp_state'}

  def __init__(self, mode: Text, config: ml_collections.ConfigDict):
    """Constructs the experiment.

    Args:
      mode: A string, equivalent to FLAGS.mode when running normally.
      config: Experiment configuration.
    """
    super().__init__(mode)
    self.mode = mode
    self.dataset_adapter = image_dataset.ImageDataset(
        **config.data.loader_kwargs)

    dataset_directory = config.data.loader_kwargs.dataset_directory
    if not config.mock_out_train_dataset:
      imagenet_with_fh.check_train_dataset_directory(dataset_directory)

    config.training.images_per_epoch = self.dataset_adapter.num_train_examples
    self.config = config

    # Checkpointed experiment state.
    self._exp_state = None

    # Input pipelines.
    self._train_input = None
    self._eval_input = None

    # build the transformed ops
    self.forward = hk.without_apply_rng(hk.transform_with_state(self._forward))
    # training can handle multiple devices, thus the pmap
    self.update_pmap = jax.pmap(self._update_fn, axis_name='i',
                                donate_argnums=(0))
    # evaluation can only handle single device
    self.eval_batch_jit = jax.jit(self._eval_batch)

  @abc.abstractmethod
  def _forward(
      self,
      inputs: image_dataset.Batch,
      is_training: bool,
  ) -> Mapping[Text, jnp.ndarray]:
    """Forward application of the network.

    Args:
      inputs: A batch of data, i.e. a dictionary, with either two keys,
        (`images` and `labels`) or three keys (`view1`, `view2`, `labels`).
      is_training: Training or evaluating the model? When True, inputs must
        contain keys `view1` and `view2`. When False, inputs must contain key
        `images`.

    Returns:
      All outputs of the model, i.e. a dictionary for either the two views, or
      the image.
    """
    pass

  @abc.abstractmethod
  def _update_fn(
      self,
      exp_state: ExperimentState,
      global_step: jnp.ndarray,
      rng: jnp.ndarray,
      inputs: image_dataset.Batch,
  ) -> Tuple[ExperimentState, LogsDict]:
    """Update online and target parameters.

    Args:
      exp_state: current experiment state.
      global_step: current training step.
      rng: current random number generator
      inputs: inputs, containing two batches of crops from the same images,
        view1 and view2 and labels

    Returns:
      Tuple containing the updated exp state after processing the inputs, and
      various logs.
    """
    pass

  @abc.abstractmethod
  def _make_initial_state(
      self,
      rng: jnp.ndarray,
      dummy_input: image_dataset.Batch,
  ) -> ExperimentState:
    """ExperimentState initialization.

    Args:
      rng: random number generator used to initialize parameters. If working in
        a multi device setup, this need to be a ShardedArray.
      dummy_input: a dummy image, used to compute intermediate outputs shapes.

    Returns:
      Initial experiment state.
    """
    pass

  def lr_schedule(self, step: jnp.ndarray) -> jnp.ndarray:
    """Cosine schedule wrapper."""
    batch_size = self.config.training.batch_size

    if self.config.optimizer.scale_by_batch:
      bs = batch_size
    else:
      bs = 256  # cancels out batch scaling

    return schedules.learning_schedule(
        step,
        batch_size=bs,
        total_steps=self.config.training.max_steps,
        base_learning_rate=self.config.optimizer.base_learning_rate,
        warmup_steps=self.config.optimizer.warmup_steps)

  def _optimizer(self, learning_rate: float) -> optax.GradientTransformation:
    """Build optimizer from config."""

    if self.config.optimizer.name == 'lars':
      return lars_optimizer.lars(
          learning_rate,
          weight_decay_filter=lars_optimizer.exclude_bias_and_norm,
          lars_adaptation_filter=lars_optimizer.exclude_bias_and_norm,
          **self.config.optimizer.lars_kwargs)
    elif self.config.optimizer.name == 'adam':
      return optax.chain(
          optax.scale_by_adam(**self.config.optimizer.adam_kwargs),
          lars_optimizer.add_weight_decay(
              self.config.optimizer.adam_weight_decay,
              filter_fn=lars_optimizer.exclude_bias_and_norm),
          optax.scale(-learning_rate))

  def _classifier_loss(
      self,
      logits: jnp.ndarray,
      labels: jnp.ndarray) -> Tuple[jnp.ndarray, LogsDict]:
    """Computes the classification loss and corresponding logs.

    Classification loss (with gradient flows stopped from flowing into the
    ResNet). This is used to provide an evaluation of the representation
    quality during training.

    Args:
      logits: the classifier logits.
      labels: the labels.

    Returns:
      The classifier loss, and a dict of logs.
    """

    # adapted from nfnets code
    def _one_hot(value, num_classes):
      """One-hot encoding potentially over a sequence of labels."""
      y = jax.nn.one_hot(value, num_classes)

      if self.config.data.loader_kwargs.dataset_name == 'jft':
        y = jnp.sum(y, -2)
        y = y / jnp.sum(y, -1, keepdims=True)  # Average one-hots
      return y

    one_hot_labels = _one_hot(labels, self.dataset_adapter.num_classes)

    classif_loss = byol_helpers.softmax_cross_entropy(logits=logits,
                                                      labels=one_hot_labels)
    classif_loss = jnp.mean(classif_loss)

    logs = dict(
        classif_loss=classif_loss,
    )
    return classif_loss, logs

  def step(self, *, global_step: jnp.ndarray, rng: jnp.ndarray,
           writer: Optional[Any]) -> Mapping[Text, np.ndarray]:
    """See base class."""
    if self._train_input is None:
      self._initialize_train()

    inputs = next(self._train_input)

    self._exp_state, scalars = self.update_pmap(
        self._exp_state,
        global_step=global_step,
        rng=rng,
        inputs=inputs,
    )

    return utils.get_first(scalars)

  def _initialize_train(self):
    """Initialize train.

    This includes initializing the input pipeline and experiment state.
    """
    self._train_input = utils.py_prefetch(self._build_train_input)

    # Check we haven't already restored params
    if self._exp_state is None:
      logging.info(
          'Initializing parameters rather than restoring from checkpoint.')

      # initialize params and optimizer state
      inputs = next(self._train_input)
      init_exp = jax.pmap(self._make_initial_state, axis_name='i')

      # Init uses the same RNG key on all hosts+devices to ensure everyone
      # computes the same initial state and parameters.
      init_rng = jax.random.PRNGKey(self.config.random_seed)
      init_rng = utils.bcast_local_devices(init_rng)

      self._exp_state = init_exp(rng=init_rng, dummy_input=inputs)

  def _fake_data_generator(self, image_shape: Tuple[int, int, int, int]):
    mask1 = np.random.uniform(low=0, high=7, size=image_shape + (1,))
    mask2 = np.random.uniform(low=0, high=7, size=image_shape + (1,))
    while True:
      yield {
          'view1': jnp.ones(image_shape + (3,)) * 0.5,
          'view2': jnp.ones(image_shape + (3,)) * 0.3,
          'fh_segmentations1': jnp.array(np.round(mask1), dtype=jnp.uint8),
          'fh_segmentations2': jnp.array(np.round(mask2), dtype=jnp.uint8),
          'labels': jnp.ones([image_shape[0], image_shape[1], 1],
                             dtype=jnp.int64),
      }

  def _build_train_input(self) -> Generator[image_dataset.Batch, None, None]:
    """See base class."""
    num_devices = jax.device_count()
    global_batch_size = self.config.training.batch_size
    per_device_batch_size, ragged = divmod(global_batch_size, num_devices)

    if ragged:
      raise ValueError(
          f'Global batch size {global_batch_size} must be divisible by '
          f'num devices {num_devices}')

    if self.config.mock_out_train_dataset:
      img_config = self.config.data.loader_kwargs.preprocessing_config.pretrain
      return self._fake_data_generator(
          image_shape=(
              jax.local_device_count(), per_device_batch_size,
              img_config.output_image_size, img_config.output_image_size),
          )
    else:
      return self.dataset_adapter.load(
          image_dataset.Split.from_string(self.config.data.training_subset),
          dataset_mode=image_dataset.DatasetMode.PRETRAIN,
          batch_dims=[jax.local_device_count(), per_device_batch_size])

  def _eval_batch(
      self,
      params: hk.Params,
      state: hk.State,
      batch: image_dataset.Batch,
  ) -> Mapping[Text, jnp.ndarray]:
    """Evaluates a batch.

    Args:
      params: Parameters of the model to evaluate. Typically the online
        parameters.
      state: State of the model to evaluate. Typically the online state.
      batch: Batch of data to evaluate (must contain keys images and labels).

    Returns:
      Unreduced evaluation loss on the batch.
    """
    batch = self.dataset_adapter.maybe_transpose_on_device(batch)

    outputs, _ = self.forward.apply(params, state, batch, is_training=False)
    logits = outputs['logits']
    labels = hk.one_hot(batch['labels'], self.dataset_adapter.num_classes)
    loss = byol_helpers.softmax_cross_entropy(logits, labels, reduction=None)
    # NOTE: Returned values will be summed and finally divided by num_samples.
    return {
        'eval_loss': loss,
    }

  def _build_eval_input(self) -> Generator[image_dataset.Batch, None, None]:
    """See base class."""
    split = image_dataset.Split.from_string(
        self.config.data.evaluation_subset)
    return self.dataset_adapter.load(
        split,
        dataset_mode=image_dataset.DatasetMode.EVAL,
        batch_dims=[self.config.evaluation.batch_size])

  def _eval_epoch(self):
    """Evaluates an epoch."""
    num_samples = 0.
    summed_scalars = None

    params = utils.get_first(self._exp_state.online_params)
    state = utils.get_first(self._exp_state.online_state)

    for inputs in self._build_eval_input():
      num_samples += inputs['labels'].shape[0]
      scalars = self.eval_batch_jit(params, state, inputs)

      # Accumulate the sum of scalars for each step.
      scalars = jax.tree_map(lambda x: jnp.sum(x, axis=0), scalars)
      if summed_scalars is None:
        summed_scalars = scalars
      else:
        summed_scalars = jax.tree_multimap(jnp.add, summed_scalars, scalars)

    mean_scalars = jax.tree_map(lambda x: x / num_samples, summed_scalars)
    return mean_scalars

  def evaluate(self, global_step, **unused_args):
    """Thin wrapper around _eval_epoch."""

    global_step = np.array(utils.get_first(global_step))
    scalars = jax.device_get(self._eval_epoch())

    logging.info('[Step %d] Eval scalars: %s', global_step, scalars)
    return scalars
