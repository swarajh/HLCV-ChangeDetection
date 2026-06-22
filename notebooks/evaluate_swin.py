import torch

from torch.utils.data import DataLoader

from change_detection.dataset import LevirDataset
from change_detection.swin_model import SwinChangeDetector
from change_detection.metrics import (
    compute_iou,
    compute_f1
)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

dataset = LevirDataset(
    "datasets/LEVIR-CD/val"
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)

model = SwinChangeDetector(
    pretrained=False
)

model.load_state_dict(
    torch.load(
        "checkpoints/swin_baseline.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

total_iou = 0
total_f1 = 0
num_samples = 0

with torch.no_grad():

    for img_a, img_b, mask in loader:

        img_a = img_a.to(device)
        img_b = img_b.to(device)
        mask = mask.to(device)

        prediction = model(
            img_a,
            img_b
        )

        prediction = torch.sigmoid(
            prediction
        )

        prediction = (
            prediction > 0.5
        ).float()

        total_iou += compute_iou(
            prediction,
            mask
        )

        total_f1 += compute_f1(
            prediction,
            mask
        )

        num_samples += 1

avg_iou = total_iou / num_samples
avg_f1 = total_f1 / num_samples

print(
    f"Validation IoU: {avg_iou:.4f}"
)

print(
    f"Validation F1 : {avg_f1:.4f}"
)