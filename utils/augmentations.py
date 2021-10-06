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

"""Data preprocessing and augmentations."""

import functools
from typing import Any, Mapping, Text

import dm_pix as pix
import jax
import jax.numpy as jnp


# typing
JaxBatch = Mapping[Text, jnp.ndarray]
ConfigDict = Mapping[Text, Any]

augment_config = dict(
    view1=dict(
        random_flip=False,  # Random left/right flip
        color_transform=dict(
            apply_prob=1.0,
            # Range of jittering
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1,
            # Probability of applying color jittering
            color_jitter_prob=0.8,
            # Probability of converting to grayscale
            to_grayscale_prob=0.2,
            # Shuffle the order of color transforms
            shuffle=True),
        gaussian_blur=dict(
            apply_prob=1.0,
            # Kernel size ~ image_size / blur_divider
            blur_divider=10.,
            # Kernel distribution
            sigma_min=0.1,
            sigma_max=2.0),
        solarize=dict(apply_prob=0.0, threshold=0.5),
    ),
    view2=dict(
        random_flip=False,
        color_transform=dict(
            apply_prob=1.0,
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1,
            color_jitter_prob=0.8,
            to_grayscale_prob=0.2,
            shuffle=True),
        gaussian_blur=dict(
            apply_prob=0.1, blur_divider=10., sigma_min=0.1, sigma_max=2.0),
        solarize=dict(apply_prob=0.2, threshold=0.5),
    ))


def postprocess(inputs: JaxBatch, rng: jnp.ndarray):
  """Apply the image augmentations to crops in inputs (view1 and view2)."""

  def _postprocess_image(
      images: jnp.ndarray,
      rng: jnp.ndarray,
      presets: ConfigDict,
  ) -> JaxBatch:
    """Applies augmentations in post-processing.

    Args:
      images: an NHWC tensor (with C=3), with float values in [0, 1].
      rng: a single PRNGKey.
      presets: a dict of presets for the augmentations.

    Returns:
      A batch of augmented images with shape NHWC, with keys view1, view2
      and labels.
    """
    flip_rng, color_rng, blur_rng, solarize_rng = jax.random.split(rng, 4)
    out = images
    if presets['random_flip']:
      out = random_flip(out, flip_rng)
    if presets['color_transform']['apply_prob'] > 0:
      out = color_transform(out, color_rng, **presets['color_transform'])
    if presets['gaussian_blur']['apply_prob'] > 0:
      out = gaussian_blur(out, blur_rng, **presets['gaussian_blur'])
    if presets['solarize']['apply_prob'] > 0:
      out = solarize(out, solarize_rng, **presets['solarize'])
    out = jnp.clip(out, 0., 1.)
    return jax.lax.stop_gradient(out)

  rng1, rng2 = jax.random.split(rng, num=2)
  view1 = _postprocess_image(inputs['view1'], rng1, augment_config['view1'])
  view2 = _postprocess_image(inputs['view2'], rng2, augment_config['view2'])
  outputs = dict(view1=view1, view2=view2, labels=inputs['labels'])
  for k in ['fh_segmentations1', 'fh_segmentations2',
            'gt_segmentations1', 'gt_segmentations2']:
    if k in inputs:
      outputs[k] = inputs[k]
  return outputs


def _maybe_apply(apply_fn, inputs, rng, apply_prob):
  should_apply = jax.random.uniform(rng, shape=()) <= apply_prob
  return jax.lax.cond(should_apply, inputs, apply_fn, inputs, lambda x: x)


def _random_gaussian_blur(image, rng, kernel_size, padding, sigma_min,
                          sigma_max, apply_prob):
  """Applies a random gaussian blur."""
  apply_rng, transform_rng = jax.random.split(rng)

  def _apply(image):
    sigma_rng, = jax.random.split(transform_rng, 1)
    sigma = jax.random.uniform(
        sigma_rng,
        shape=(),
        minval=sigma_min,
        maxval=sigma_max,
        dtype=jnp.float32)
    return pix.gaussian_blur(image, sigma, kernel_size, padding=padding)

  return _maybe_apply(_apply, image, apply_rng, apply_prob)


def _color_transform_single_image(image, rng, brightness, contrast, saturation,
                                  hue, to_grayscale_prob, color_jitter_prob,
                                  apply_prob, shuffle):
  """Applies color jittering to a single image."""
  apply_rng, transform_rng = jax.random.split(rng)
  perm_rng, b_rng, c_rng, s_rng, h_rng, cj_rng, gs_rng = jax.random.split(
      transform_rng, 7)

  # Whether the transform should be applied at all.
  should_apply = jax.random.uniform(apply_rng, shape=()) <= apply_prob
  # Whether to apply grayscale transform.
  should_apply_gs = jax.random.uniform(gs_rng, shape=()) <= to_grayscale_prob
  # Whether to apply color jittering.
  should_apply_color = jax.random.uniform(cj_rng, shape=()) <= color_jitter_prob

  # Decorator to conditionally apply fn based on an index.
  def _make_cond(fn, idx):

    def identity_fn(unused_rng, x):
      return x

    def cond_fn(args, i):
      def clip(args):
        return jax.tree_map(lambda arg: jnp.clip(arg, 0., 1.), args)
      out = jax.lax.cond(should_apply & should_apply_color & (i == idx), args,
                         lambda a: clip(fn(*a)), args,
                         lambda a: identity_fn(*a))
      return jax.lax.stop_gradient(out)

    return cond_fn

  random_brightness = functools.partial(
      pix.random_brightness, max_delta=brightness)
  random_contrast = functools.partial(
      pix.random_contrast, lower=1-contrast, upper=1+contrast)
  random_hue = functools.partial(pix.random_hue, max_delta=hue)
  random_saturation = functools.partial(
      pix.random_saturation, lower=1-saturation, upper=1+saturation)
  to_grayscale = functools.partial(pix.rgb_to_grayscale, keep_dims=True)

  random_brightness_cond = _make_cond(random_brightness, idx=0)
  random_contrast_cond = _make_cond(random_contrast, idx=1)
  random_saturation_cond = _make_cond(random_saturation, idx=2)
  random_hue_cond = _make_cond(random_hue, idx=3)

  def _color_jitter(x):
    if shuffle:
      order = jax.random.permutation(perm_rng, jnp.arange(4, dtype=jnp.int32))
    else:
      order = range(4)
    for idx in order:
      if brightness > 0:
        x = random_brightness_cond((b_rng, x), idx)
      if contrast > 0:
        x = random_contrast_cond((c_rng, x), idx)
      if saturation > 0:
        x = random_saturation_cond((s_rng, x), idx)
      if hue > 0:
        x = random_hue_cond((h_rng, x), idx)
    return x

  out_apply = _color_jitter(image)
  out_apply = jax.lax.cond(should_apply & should_apply_gs, out_apply,
                           to_grayscale, out_apply, lambda x: x)
  return jnp.clip(out_apply, 0., 1.)


def random_flip(images, rng):
  rngs = jax.random.split(rng, images.shape[0])
  return jax.vmap(pix.random_flip_left_right)(rngs, images)


def color_transform(images,
                    rng,
                    brightness=0.8,
                    contrast=0.8,
                    saturation=0.8,
                    hue=0.2,
                    color_jitter_prob=0.8,
                    to_grayscale_prob=0.2,
                    apply_prob=1.0,
                    shuffle=True):
  """Applies color jittering and/or grayscaling to a batch of images.

  Args:
    images: an NHWC tensor, with C=3.
    rng: a single PRNGKey.
    brightness: the range of jitter on brightness.
    contrast: the range of jitter on contrast.
    saturation: the range of jitter on saturation.
    hue: the range of jitter on hue.
    color_jitter_prob: the probability of applying color jittering.
    to_grayscale_prob: the probability of converting the image to grayscale.
    apply_prob: the probability of applying the transform to a batch element.
    shuffle: whether to apply the transforms in a random order.

  Returns:
    A NHWC tensor of the transformed images.
  """
  rngs = jax.random.split(rng, images.shape[0])
  jitter_fn = functools.partial(
      _color_transform_single_image,
      brightness=brightness,
      contrast=contrast,
      saturation=saturation,
      hue=hue,
      color_jitter_prob=color_jitter_prob,
      to_grayscale_prob=to_grayscale_prob,
      apply_prob=apply_prob,
      shuffle=shuffle)
  return jax.vmap(jitter_fn)(images, rngs)


def gaussian_blur(images,
                  rng,
                  blur_divider=10.,
                  sigma_min=0.1,
                  sigma_max=2.0,
                  apply_prob=1.0):
  """Applies gaussian blur to a batch of images.

  Args:
    images: an NHWC tensor, with C=3.
    rng: a single PRNGKey.
    blur_divider: the blurring kernel will have size H / blur_divider.
    sigma_min: the minimum value for sigma in the blurring kernel.
    sigma_max: the maximum value for sigma in the blurring kernel.
    apply_prob: the probability of applying the transform to a batch element.

  Returns:
    A NHWC tensor of the blurred images.
  """
  rngs = jax.random.split(rng, images.shape[0])
  kernel_size = images.shape[1] / blur_divider
  blur_fn = functools.partial(
      _random_gaussian_blur,
      kernel_size=kernel_size,
      padding='SAME',
      sigma_min=sigma_min,
      sigma_max=sigma_max,
      apply_prob=apply_prob)
  return jax.vmap(blur_fn)(images, rngs)


def _solarize_single_image(image, rng, threshold, apply_prob):

  solarize_fn = functools.partial(pix.solarize, threshold=threshold)
  return _maybe_apply(solarize_fn, image, rng, apply_prob)


def solarize(images, rng, threshold=0.5, apply_prob=1.0):
  """Applies solarization.

  Args:
    images: an NHWC tensor (with C=3).
    rng: a single PRNGKey.
    threshold: the solarization threshold.
    apply_prob: the probability of applying the transform to a batch element.

  Returns:
    A NHWC tensor of the transformed images.
  """
  rngs = jax.random.split(rng, images.shape[0])
  solarize_fn = functools.partial(
      _solarize_single_image, threshold=threshold, apply_prob=apply_prob)
  return jax.vmap(solarize_fn)(images, rngs)
