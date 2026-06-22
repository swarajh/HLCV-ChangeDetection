import torch

from mae.mae_model import MAE

model = MAE()

dummy = torch.randn(
    2,
    3,
    224,
    224
)

reconstruction, target, mask = model(
    dummy
)

print(
    reconstruction.shape
)

print(
    target.shape
)

print(
    mask.shape
)

print(
    mask.mean()
)