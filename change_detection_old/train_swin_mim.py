import os

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from change_detection.dataset import LevirDataset
from change_detection.swin_model_mim import (
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


# -----------------------
# Device
# -----------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# -----------------------
# Model
# -----------------------

print(
    "Creating SimMIM model..."
)

model = SwinChangeDetector(
    pretrained=False,
    mim_weights=
    "checkpoints/mim_encoder.pth"
)

model = model.to(device)

print(
    "Model loaded successfully"
)


# -----------------------
# Losses
# -----------------------

bce_loss = nn.BCEWithLogitsLoss()

dice_loss = DiceLoss()


# -----------------------
# Optimizer
# -----------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)


# -----------------------
# Checkpoint Folder
# -----------------------

os.makedirs(
    "checkpoints",
    exist_ok=True
)


# -----------------------
# Training
# -----------------------

epochs = 30

best_loss = float("inf")

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for image_a, image_b, label in train_loader:

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
        f"Loss: {avg_loss:.4f}"
    )

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            "checkpoints/swin_mim.pth"
        )

        print(
            "Best model saved!"
        )

print(
    "Training finished!"
)