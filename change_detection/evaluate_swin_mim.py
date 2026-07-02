import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from torch.utils.data import DataLoader

from change_detection.dataset import LevirDataset
from change_detection.swin_model_mim import (
    SwinChangeDetector
)
from change_detection.metrics import (
    compute_iou,
    compute_f1
)

print(
    "Evaluation script started",
    flush=True
)

# -----------------------
# Device
# -----------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print( "Using device:",device,flush=True)

# -----------------------
# Dataset
# -----------------------

print( "Creating validation dataset...", flush=True)

dataset = LevirDataset("datasets/LEVIR-CD/val")

print("Validation dataset created",flush=True
)

print("Validation size:",len(dataset),flush=True)

loader = DataLoader( dataset, batch_size=1, shuffle=False)

print("Validation loader created",flush=True)

print("Number of batches:",len(loader),flush=True)

# -----------------------
# Model
# -----------------------

print("Creating model...",flush=True)

model = SwinChangeDetector(pretrained=False)

print("Model created",flush=True)

print("Loading checkpoint...",flush=True)

model.load_state_dict(torch.load("checkpoints/swin_mim.pth",map_location=device))

print("Checkpoint loaded",flush=True)

model = model.to(device)

print("Model moved to device",flush=True)

model.eval()

print("Starting evaluation...", flush=True)

# -----------------------
# Evaluation
# -----------------------

total_iou = 0
total_f1 = 0
num_samples = 0

with torch.no_grad():

    for idx, (img_a,img_b,mask) in enumerate(loader):

        if idx == 0:

            print("First validation batch loaded",flush=True)

        img_a = img_a.to(device)
        img_b = img_b.to(device)
        mask = mask.to(device)

        prediction = model(img_a,img_b)

        prediction = torch.sigmoid(prediction)

        prediction = (prediction > 0.5).float()

        total_iou += compute_iou(prediction,mask)

        total_f1 += compute_f1(prediction,mask)

        num_samples += 1

        if idx % 20 == 0:

            print(f"Processed {idx}/{len(loader)} images",flush=True)

avg_iou = total_iou / num_samples
avg_f1 = total_f1 / num_samples

print(f"Validation IoU: {avg_iou:.4f}",flush=True)

print(f"Validation F1 : {avg_f1:.4f}",flush=True)

print("Evaluation finished!",flush=True)