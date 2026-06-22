import torch

from mae.mae_model import MAE

model = MAE()

dummy = torch.randn(
    1,
    3,
    224,
    224
)

masked, mask = model.create_patch_mask(
    dummy
)

print(mask.shape)

print(
    "Visible ratio:",
    mask.mean().item()
)