"""Configuration overrides tailored for KAN-based coupling networks.

Import this module by setting the DEEPMIH_CONFIG environment variable
(e.g. `DEEPMIH_CONFIG=kan_config python train.py`) to adopt training
hyper-parameters that keep the new KAN submodules stable.
"""

from config import *  # noqa: F401,F403

# Learning rate: start an order of magnitude lower than the historical
# Dense-based setup to stabilise the more expressive KAN parameters.
log10_lr = -5.5
lr = 10 ** log10_lr

# Match the identity-friendly defaults for the importance-map optimiser.
lr3 = 10 ** -5.0

# KAN layers are memory hungry; trimming the batch size helps prevent OOMs
# on common 24GB GPUs when both stages run simultaneously.
batch_size = 12

# Retain gradient clipping and checkpoint filtering behaviour from base config.
grad_clip_norm = 1.0
pretrained_skip_substrings = ("kan",)