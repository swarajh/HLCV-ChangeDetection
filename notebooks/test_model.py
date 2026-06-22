import torch
from change_detection.baseline_model import SiameseBaseline

model = SiameseBaseline()

img_a = torch.randn(4,3,256,256)
img_b = torch.randn(4,3,256,256)

out = model(img_a,img_b)

print(out.shape)