# mim/test_loss.py

import torch

from mim.mim_loss import mim_loss

reconstruction = torch.randn(
    2,
    3,
    224,
    224
)

target = torch.randn(
    2,
    3,
    224,
    224
)

mask = torch.ones(
    2,
    1,
    224,
    224
) * 0.25

loss = mim_loss(
    reconstruction,
    target,
    mask
)

print(loss)