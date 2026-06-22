import torch

from change_detection.swin_model import SwinChangeDetector

model = SwinChangeDetector()

img_a = torch.randn(
    2, 3, 224, 224
)

img_b = torch.randn(
    2, 3, 224, 224
)

out = model(
    img_a,
    img_b
)

print(out.shape)