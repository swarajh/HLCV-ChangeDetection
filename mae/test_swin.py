# mae/test_swin.py

import torch
import timm

model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True,
    num_classes=0
)

x = torch.randn(
    2,
    3,
    224,
    224
)

y = model(x)

print(y.shape)