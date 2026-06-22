# mim/test_model.py

import torch

from mim.mim_model import SimMIM

model = SimMIM()

dummy = torch.randn(
    2,
    3,
    224,
    224
)

reconstruction, target, mask = model(
    dummy
)

print(reconstruction.shape)
print(target.shape)
print(mask.shape)
print(mask.mean())