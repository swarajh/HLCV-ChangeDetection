import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from change_detection.dataset import LevirDataset
from change_detection.swin_model_simsiam import (
    SwinChangeDetector
)
from change_detection.losses import DiceLoss


# -----------------------
# Datasets
# -----------------------

train_dataset = LevirDataset(
    "datasets/LEVIR-CD/train"
)

val_dataset = LevirDataset(
    "datasets/LEVIR-CD/val"
)

print("Train dataset created", flush=True)
print("Train size:", len(train_dataset), flush=True)

print("Val dataset created", flush=True)
print("Val size:", len(val_dataset), flush=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False
)

print("Train loader created", flush=True)
print("Number of train batches:", len(train_loader), flush=True)


# -----------------------
# Device
# -----------------------

print("Before device selection", flush=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device selected", flush=True)
print("Using device:", device, flush=True)


# -----------------------
# Model
# -----------------------

print("Creating model...", flush=True)

model = SwinChangeDetector(
    pretrained=False,
    simsiam_weights=
    "checkpoints/simsiam_encoder.pth"
)

print("Model object created", flush=True)

model = model.to(device)

print("Model moved to device", flush=True)


# -----------------------
# Losses
# -----------------------

print("Creating losses...", flush=True)

bce_loss = nn.BCEWithLogitsLoss()

dice_loss = DiceLoss()

print("Losses created", flush=True)


# -----------------------
# Optimizer
# -----------------------

print("Creating optimizer...", flush=True)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

print("Optimizer created", flush=True)


# -----------------------
# Checkpoint Folder
# -----------------------

os.makedirs(
    "checkpoints",
    exist_ok=True
)

print("Checkpoint directory ready", flush=True)


# -----------------------
# Training
# -----------------------

print("Starting training...", flush=True)

epochs = 30

best_loss = float("inf")

for epoch in range(epochs):

    print(
        f"Starting Epoch {epoch+1}",
        flush=True
    )

    model.train()

    running_loss = 0

    for batch_idx, (
        image_a,
        image_b,
        label
    ) in enumerate(train_loader):

        if batch_idx == 0:
            print(
                "First batch loaded",
                flush=True
            )

        image_a = image_a.to(device)
        image_b = image_b.to(device)
        label = label.to(device)

        output = model(
            image_a,
            image_b
        )

        loss = (
            bce_loss(output, label)
            +
            dice_loss(output, label)
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = (
        running_loss
        /
        len(train_loader)
    )

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss: {avg_loss:.4f}",
        flush=True
    )

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            "checkpoints/swin_simsiam.pth"
        )

        print(
            "Best model saved!",
            flush=True
        )

print(
    "Training finished!",
    flush=True
)
