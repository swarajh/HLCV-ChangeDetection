# mae/test_features.py

import torch
import timm

model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True,
    features_only=True
)

x = torch.randn(
    2,
    3,
    224,
    224
)

features = model(x)

for i, feat in enumerate(features):

    print(
        i,
        feat.shape
    )