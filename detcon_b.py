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

"""DetCon-B implementation using Jaxline."""

import sys
from typing import Any, Mapping, Text, Tuple

from absl import flags
import haiku as hk
import jax
import jax.numpy as jnp
from jaxline import platform
import ml_collections
import numpy as np
import optax

from detcon import pretrain_common
from detcon.datasets import image_dataset
from detcon.utils import augmentations
from detcon.utils import helpers as byol_helpers
from detcon.utils import losses
from detcon.utils import networks
from detcon.utils import schedules


def featurewise_std(x: jnp.ndarray) -> jnp.ndarray:
  """Computes the featurewise standard deviation."""
  return jnp.mean(jnp.std(x, axis=0))


class PretrainExperiment(pretrain_common.BaseExperiment):
  """DetCon-B's training component definition."""

  def __init__(self, mode: Text, config: ml_collections.ConfigDict):
    """Constructs the experiment.

    Args:
      mode: A string, equivalent to FLAGS.mode when running normally.
      config: Experiment configuration.
    """
    super().__init__(mode, config)

    assert config.training_mode in ['self-supervised', 'supervised', 'both']

    if mode == 'train':
      self.forward = hk.transform_with_state(self._forward)

  def create_binary_mask(
      self,
      batch_size,
      num_pixels,
      masks,
      max_mask_id=256,
      downsample=(1, 32, 32, 1)):
    """Generates binary masks from the Felzenszwalb masks.

    From a FH mask of shape [batch_size, H,W] (values in range
    [0,max_mask_id], produces corresponding (downsampled) binary masks of
    shape [batch_size, max_mask_id, H*W/downsample].
    Args:
      batch_size: batch size of the masks
      num_pixels: Number of points on the spatial grid
      masks: Felzenszwalb masks
      max_mask_id: # unique masks in Felzenszwalb segmentation
      downsample: rate at which masks must be downsampled.
    Returns:
      binary_mask: Binary mask with specification above
    """
    fh_mask_to_use = self.config.model.fh_mask_to_use
    mask = masks[..., fh_mask_to_use:(fh_mask_to_use+1)]

    mask_ids = jnp.arange(max_mask_id).reshape(1, 1, 1, max_mask_id)
    binary_mask = jnp.equal(mask_ids, mask).astype('float32')

    binary_mask = hk.avg_pool(binary_mask, downsample, downsample, 'VALID')
    binary_mask = binary_mask.reshape(batch_size, num_pixels, max_mask_id)
    binary_mask = jnp.argmax(binary_mask, axis=-1)
    binary_mask = jnp.eye(max_mask_id)[binary_mask]
    binary_mask = jnp.transpose(binary_mask, [0, 2, 1])
    return binary_mask

  def sample_masks(self, binary_mask, batch_size, n_random_vectors=16):
    """Samples which binary masks to use in the loss."""
    mask_exists = jnp.greater(binary_mask.sum(-1), 1e-3)
    sel_masks = mask_exists.astype('float32') + 0.00000000001
    sel_masks = sel_masks / sel_masks.sum(1, keepdims=True)
    sel_masks = jnp.log(sel_masks)

    mask_ids = jax.random.categorical(
        hk.next_rng_key(), sel_masks, axis=-1,
        shape=tuple([n_random_vectors, batch_size]))
    mask_ids = jnp.transpose(mask_ids, [1, 0])

    smpl_masks = jnp.stack(
        [binary_mask[b][mask_ids[b]] for b in range(batch_size)])
    return smpl_masks, mask_ids

  def run_detcon_b_forward_on_view(
      self,
      view_encoder: Any,
      projector: Any,
      predictor: Any,
      classifier: Any,
      is_training: bool,
      images: jnp.ndarray,
      masks: jnp.ndarray,
      suffix: Text = '',
      ):
    outputs = {}
    images = self.dataset_adapter.normalize_images(images)

    embedding_local = view_encoder(images, is_training=is_training)
    embedding = jnp.mean(embedding_local, axis=[1, 2])

    bs, emb_h, emb_w, emb_d = embedding_local.shape
    emb_a = emb_h * emb_w

    if masks is not None and self.config.training_mode != 'supervised':
      binary_mask = self.create_binary_mask(bs, emb_a, masks)
      smpl_masks, mask_ids = self.sample_masks(binary_mask, bs)
      smpl_masks_area = smpl_masks.sum(axis=-1, keepdims=True)
      smpl_masks = smpl_masks / jnp.maximum(smpl_masks_area, 1.)

      embedding_local = embedding_local.reshape([bs, emb_a, emb_d])
      smpl_embedding = jnp.matmul(smpl_masks, embedding_local)

      proj_out = projector(smpl_embedding, is_training)
      pred_out = predictor(proj_out, is_training)

      outputs['projection' + suffix] = proj_out
      outputs['prediction' + suffix] = pred_out
      outputs['mask_ids' + suffix] = mask_ids

    # Note the stop_gradient: label information is not leaked into the
    # main network.
    if self.config.training_mode == 'self-supervised':
      embedding = jax.lax.stop_gradient(embedding)

    classif_out = classifier(embedding)
    outputs['logits' + suffix] = classif_out
    return outputs

  def _forward(
      self,
      inputs: image_dataset.Batch,
      is_training: bool,
  ) -> Mapping[Text, jnp.ndarray]:
    """Forward application of byol's architecture.

    Args:
      inputs: A batch of data, i.e. a dictionary, with either two keys,
        (`images` and `labels`) or three keys (`view1`, `view2`, `labels`).
      is_training: Training or evaluating the model? When True, inputs must
        contain keys `view1` and `view2`. When False, inputs must contain key
        `images`.

    Returns:
      All outputs of the model, i.e. a dictionary with projection, prediction
      and logits keys, for either the two views, or the image.
    """
    mlp_kwargs = dict(
        hidden_size=self.config.model.mlp_hidden_size,
        bn_config=self.config.model.norm_config,
        output_size=self.config.model.projection_size)

    encoder = getattr(networks, self.config.model.encoder)
    view_encoder = encoder(
        num_classes=None,  # Don't build the final linear layer
        resnet_v2=self.config.model.encoder_use_v2,
        bn_config=self.config.model.norm_config,
        width_multiplier=self.config.model.encoder_width_multiplier,
        final_mean_pool=False)

    projector = networks.MLP(name='projector', **mlp_kwargs)
    predictor = networks.MLP(name='predictor', **mlp_kwargs)
    classifier = hk.Linear(
        output_size=self.dataset_adapter.num_classes, name='classifier')

    if is_training:
      outputs_view1 = self.run_detcon_b_forward_on_view(
          view_encoder, projector, predictor, classifier, is_training,
          inputs['view1'], inputs['fh_segmentations1'], '_view1')
      outputs_view2 = self.run_detcon_b_forward_on_view(
          view_encoder, projector, predictor, classifier, is_training,
          inputs['view2'], inputs['fh_segmentations2'], '_view2')
      return {**outputs_view1, **outputs_view2}
    else:
      return self.run_detcon_b_forward_on_view(
          view_encoder, projector, predictor, classifier, is_training,
          inputs['images'], None, '')

  def loss_fn(
      self,
      online_params: hk.Params,
      target_params: hk.Params,
      online_state: hk.State,
      target_state: hk.Params,
      rng: jnp.ndarray,
      inputs: image_dataset.Batch,
  ) -> Tuple[jnp.ndarray, Tuple[Mapping[Text, hk.State],
                                pretrain_common.LogsDict]]:
    """Compute BYOL's loss function.

    Args:
      online_params: parameters of the online network (the loss is later
        differentiated with respect to the online parameters).
      target_params: parameters of the target network.
      online_state: internal state of online network.
      target_state: internal state of target network.
      rng: random number generator state.
      inputs: inputs, containing two batches of crops from the same images,
        view1 and view2 and labels

    Returns:
      BYOL's loss, a mapping containing the online and target networks updated
      states after processing inputs, and various logs.
    """
    inputs = self.dataset_adapter.maybe_transpose_on_device(inputs)

    inputs = augmentations.postprocess(inputs, rng)

    online_network_out, online_state = self.forward.apply(
        params=online_params,
        state=online_state,
        rng=rng,
        inputs=inputs,
        is_training=True)
    target_network_out, target_state = self.forward.apply(
        params=target_params,
        state=target_state,
        rng=rng,
        inputs=inputs,
        is_training=True)

    # Representation loss

    # The stop_gradient is not necessary as we explicitly take the gradient with
    # respect to online parameters only in `optax.apply_updates`. We leave it to
    # indicate that gradients are not backpropagated through the target network.

    repr_loss = 0.0
    if self.config.training_mode != 'supervised':
      repr_loss = losses.byol_nce_detcon(
          online_network_out['prediction_view1'],
          online_network_out['prediction_view2'],
          jax.lax.stop_gradient(target_network_out['projection_view1']),
          jax.lax.stop_gradient(target_network_out['projection_view2']),
          online_network_out['mask_ids_view1'],
          online_network_out['mask_ids_view2'],
          target_network_out['mask_ids_view1'],
          target_network_out['mask_ids_view2'],
          temperature=self.config.training.nce_loss_temperature)

    classif_loss, logs = self._classifier_loss(
        logits=online_network_out['logits_view1'], labels=inputs['labels'])
    loss = repr_loss + classif_loss

    if self.config.training_mode != 'supervised':
      logs.update(
          dict(
              loss=loss,
              repr_loss=repr_loss,
              proj_mean=online_network_out['projection_view1'].mean(),
              proj_std=featurewise_std(
                  online_network_out['projection_view1']),
              normalized_proj_std=featurewise_std(
                  byol_helpers.l2_normalize(
                      online_network_out['projection_view1'], axis=-1)),
              pred_mean=online_network_out['prediction_view1'].mean(),
              pred_std=featurewise_std(
                  online_network_out['prediction_view1']),
              normalized_pred_std=featurewise_std(
                  byol_helpers.l2_normalize(
                      online_network_out['prediction_view1'], axis=-1),)))
    else:
      logs.update(dict(loss=loss, repr_loss=repr_loss))

    return loss, (dict(online_state=online_state,
                       target_state=target_state), logs)

  def _update_fn(
      self,
      byol_state: pretrain_common.ExperimentState,
      global_step: jnp.ndarray,
      rng: jnp.ndarray,
      inputs: image_dataset.Batch,
  ) -> Tuple[pretrain_common.ExperimentState, pretrain_common.LogsDict]:
    """Update online and target parameters.

    Args:
      byol_state: current BYOL state.
      global_step: current training step.
      rng: current random number generator
      inputs: inputs, containing two batches of crops from the same images,
        view1 and view2 and labels

    Returns:
      Tuple containing the updated Byol state after processing the inputs, and
      various logs.
    """
    online_params = byol_state.online_params
    target_params = byol_state.target_params
    online_state = byol_state.online_state
    target_state = byol_state.target_state
    opt_state = byol_state.opt_state

    # update online network
    grad_fn = jax.grad(self.loss_fn, argnums=0, has_aux=True)
    grads, (net_states, logs) = grad_fn(online_params, target_params,
                                        online_state, target_state, rng, inputs)

    # cross-device grad and logs reductions
    grads = jax.tree_map(lambda v: jax.lax.pmean(v, axis_name='i'), grads)
    logs = jax.tree_multimap(lambda x: jax.lax.pmean(x, axis_name='i'), logs)

    lr = self.lr_schedule(global_step)
    updates, opt_state = self._optimizer(lr).update(grads, opt_state,
                                                    online_params)
    online_params = optax.apply_updates(online_params, updates)

    # update target network
    tau = schedules.target_ema(
        global_step,
        base_ema=self.config.training.base_target_ema,
        max_steps=self.config.training.max_steps)
    target_params = jax.tree_multimap(lambda x, y: x + (1 - tau) * (y - x),
                                      target_params, online_params)
    logs['tau'] = tau
    logs['lr'] = lr

    n_params = 0

    for key in online_params:
      for l in online_params[key]:
        n_params += np.prod(online_params[key][l].shape)

    logs['n_params'] = n_params

    return pretrain_common.ExperimentState(
        online_params=online_params,
        target_params=target_params,
        online_state=net_states['online_state'],
        target_state=net_states['target_state'],
        opt_state=opt_state), logs

  def _make_initial_state(
      self,
      rng: jnp.ndarray,
      dummy_input: image_dataset.Batch,
  ) -> pretrain_common.ExperimentState:
    """BYOL's _ExperimentState initialization.

    Args:
      rng: random number generator used to initialize parameters. If working in
        a multi device setup, this need to be a ShardedArray.
      dummy_input: a dummy image, used to compute intermediate outputs shapes.

    Returns:
      Initial Byol state.
    """
    rng_online, rng_target = jax.random.split(rng)

    dummy_input = self.dataset_adapter.maybe_transpose_on_device(dummy_input)

    # Online and target parameters are initialized using different rngs,
    # in our experiments we did not notice a significant different with using
    # the same rng for both.
    online_params, online_state = self.forward.init(
        rng_online,
        dummy_input,
        is_training=True,
    )
    target_params, target_state = self.forward.init(
        rng_target,
        dummy_input,
        is_training=True,
    )
    opt_state = self._optimizer(0).init(online_params)
    return pretrain_common.ExperimentState(
        online_params=online_params,
        target_params=target_params,
        opt_state=opt_state,
        online_state=online_state,
        target_state=target_state,
    )


if __name__ == '__main__':
  flags.mark_flag_as_required('config')
  platform.main(PretrainExperiment, sys.argv[1:])
