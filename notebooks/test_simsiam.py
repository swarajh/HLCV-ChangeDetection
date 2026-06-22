import torch

from simsiam.simsiam_model import SimSiam

model = SimSiam()

x1 = torch.randn(
    2,
    3,
    224,
    224
)

x2 = torch.randn(
    2,
    3,
    224,
    224
)

p1, p2, z1, z2 = model(
    x1,
    x2
)

print("p1:", p1.shape)
print("p2:", p2.shape)

print("z1:", z1.shape)
print("z2:", z2.shape)