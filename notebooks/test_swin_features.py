import torch
import timm

model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True,
    features_only=True
)

x = torch.randn(
    1,3,224,224
)

features = model(x)

print(type(features))

for i, feat in enumerate(features):
    print(
        f"Stage {i}:",
        feat.shape
    )