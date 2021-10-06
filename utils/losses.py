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

"""DetCon/BYOL losses."""

import haiku as hk
import jax
import jax.numpy as jnp

from detcon.utils import helpers


def manual_cross_entropy(labels, logits, weight):
  ce = - weight * jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
  return jnp.mean(ce)


def byol_nce_detcon(pred1, pred2, target1, target2,
                    pind1, pind2, tind1, tind2,
                    temperature=0.1, use_replicator_loss=True,
                    local_negatives=True):
  """Compute the NCE scores from pairs of predictions and targets.

  This implements the batched form of the loss described in
  Section 3.1, Equation 3 in https://arxiv.org/pdf/2103.10957.pdf.

  Args:
    pred1 (jnp.array): the prediction from first view.
    pred2 (jnp.array): the prediction from second view.
    target1 (jnp.array): the projection from first view.
    target2 (jnp.array): the projection from second view.
    pind1 (jnp.array): mask indices for first view's prediction.
    pind2 (jnp.array): mask indices for second view's prediction.
    tind1 (jnp.array): mask indices for first view's projection.
    tind2 (jnp.array): mask indices for second view's projection.
    temperature (float): the temperature to use for the NCE loss.
    use_replicator_loss (bool): use cross-replica samples.
    local_negatives (bool): whether to include local negatives

  Returns:
    A single scalar loss for the XT-NCE objective.

  """
  batch_size = pred1.shape[0]
  num_rois = pred1.shape[1]
  feature_dim = pred1.shape[-1]
  infinity_proxy = 1e9  # Used for masks to proxy a very large number.

  def make_same_obj(ind_0, ind_1):
    same_obj = jnp.equal(ind_0.reshape([batch_size, num_rois, 1]),
                         ind_1.reshape([batch_size, 1, num_rois]))
    return jnp.expand_dims(same_obj.astype("float32"), axis=2)
  same_obj_aa = make_same_obj(pind1, tind1)
  same_obj_ab = make_same_obj(pind1, tind2)
  same_obj_ba = make_same_obj(pind2, tind1)
  same_obj_bb = make_same_obj(pind2, tind2)

  # L2 normalize the tensors to use for the cosine-similarity
  pred1 = helpers.l2_normalize(pred1, axis=-1)
  pred2 = helpers.l2_normalize(pred2, axis=-1)
  target1 = helpers.l2_normalize(target1, axis=-1)
  target2 = helpers.l2_normalize(target2, axis=-1)

  if jax.device_count() > 1 and use_replicator_loss:
    # Grab tensor across replicas and expand first dimension
    target1_large = jax.lax.all_gather(target1, axis_name="i")
    target2_large = jax.lax.all_gather(target2, axis_name="i")

    # Fold into batch dimension
    target1_large = target1_large.reshape(-1, num_rois, feature_dim)
    target2_large = target2_large.reshape(-1, num_rois, feature_dim)

    # Create the labels by using the current replica ID and offsetting.
    replica_id = jax.lax.axis_index("i")
    labels_idx = jnp.arange(batch_size) + replica_id * batch_size
    labels_idx = labels_idx.astype(jnp.int32)
    enlarged_batch_size = target1_large.shape[0]
    labels_local = hk.one_hot(labels_idx, enlarged_batch_size)
    labels_ext = hk.one_hot(labels_idx, enlarged_batch_size * 2)

  else:
    target1_large = target1
    target2_large = target2
    labels_local = hk.one_hot(jnp.arange(batch_size), batch_size)
    labels_ext = hk.one_hot(jnp.arange(batch_size), batch_size * 2)

  labels_local = jnp.expand_dims(jnp.expand_dims(labels_local, axis=2), axis=1)
  labels_ext = jnp.expand_dims(jnp.expand_dims(labels_ext, axis=2), axis=1)

  # Do our matmuls and mask out appropriately.
  logits_aa = jnp.einsum("abk,uvk->abuv", pred1, target1_large) / temperature
  logits_bb = jnp.einsum("abk,uvk->abuv", pred2, target2_large) / temperature
  logits_ab = jnp.einsum("abk,uvk->abuv", pred1, target2_large) / temperature
  logits_ba = jnp.einsum("abk,uvk->abuv", pred2, target1_large) / temperature

  labels_aa = labels_local * same_obj_aa
  labels_ab = labels_local * same_obj_ab
  labels_ba = labels_local * same_obj_ba
  labels_bb = labels_local * same_obj_bb

  logits_aa = logits_aa - infinity_proxy * labels_local * same_obj_aa
  logits_bb = logits_bb - infinity_proxy * labels_local * same_obj_bb
  labels_aa = 0. * labels_aa
  labels_bb = 0. * labels_bb
  if not local_negatives:
    logits_aa = logits_aa - infinity_proxy * labels_local * (1 - same_obj_aa)
    logits_ab = logits_ab - infinity_proxy * labels_local * (1 - same_obj_ab)
    logits_ba = logits_ba - infinity_proxy * labels_local * (1 - same_obj_ba)
    logits_bb = logits_bb - infinity_proxy * labels_local * (1 - same_obj_bb)

  labels_abaa = jnp.concatenate([labels_ab, labels_aa], axis=2)
  labels_babb = jnp.concatenate([labels_ba, labels_bb], axis=2)

  labels_0 = jnp.reshape(labels_abaa, [batch_size, num_rois, -1])
  labels_1 = jnp.reshape(labels_babb, [batch_size, num_rois, -1])

  num_positives_0 = jnp.sum(labels_0, axis=-1, keepdims=True)
  num_positives_1 = jnp.sum(labels_1, axis=-1, keepdims=True)

  labels_0 = labels_0 / jnp.maximum(num_positives_0, 1)
  labels_1 = labels_1 / jnp.maximum(num_positives_1, 1)

  obj_area_0 = jnp.sum(make_same_obj(pind1, pind1), axis=[2, 3])
  obj_area_1 = jnp.sum(make_same_obj(pind2, pind2), axis=[2, 3])

  weights_0 = jnp.greater(num_positives_0[..., 0], 1e-3).astype("float32")
  weights_0 = weights_0 / obj_area_0
  weights_1 = jnp.greater(num_positives_1[..., 0], 1e-3).astype("float32")
  weights_1 = weights_1 / obj_area_1

  logits_abaa = jnp.concatenate([logits_ab, logits_aa], axis=2)
  logits_babb = jnp.concatenate([logits_ba, logits_bb], axis=2)

  logits_abaa = jnp.reshape(logits_abaa, [batch_size, num_rois, -1])
  logits_babb = jnp.reshape(logits_babb, [batch_size, num_rois, -1])

  loss_a = manual_cross_entropy(labels_0, logits_abaa, weights_0)
  loss_b = manual_cross_entropy(labels_1, logits_babb, weights_1)
  loss = loss_a + loss_b

  return loss
