import torch

from simsiam.simsiam_loss import (
    simsiam_loss
)

p1 = torch.randn(
    4,
    2048
)

p2 = torch.randn(
    4,
    2048
)

z1 = torch.randn(
    4,
    2048
)

z2 = torch.randn(
    4,
    2048
)

loss = simsiam_loss(
    p1,
    p2,
    z1,
    z2
)

print(loss)