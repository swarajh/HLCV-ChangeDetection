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

from torch.utils.data import DataLoader

from mim_new.eurosat_dataset import (
    EuroSATMIMDataset
)

from mim_new.mim_model import (
    SimMIM
)

from mim_new.mim_loss import (
    mim_loss
)


# -----------------------
# Device
# -----------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(
    "Using device:",
    device
)


# -----------------------
# Dataset
# -----------------------

dataset = EuroSATMIMDataset(
    "datasets/EuroSAT"
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

print(
    "Dataset Size:",
    len(dataset)
)

print(
    "Number of batches:",
    len(loader)
)


# -----------------------
# Model
# -----------------------

model = SimMIM(
    mask_ratio=0.75
)

model = model.to(device)

print(
    "Model created"
)


# -----------------------
# Optimizer
# -----------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.05
)

# ------------------------------------
# Cosine Learning Rate Scheduler
# ------------------------------------

epochs = 100

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=1e-6
)


# -----------------------
# Checkpoints
# -----------------------

os.makedirs(
    "checkpoints",
    exist_ok=True
)


# -----------------------
# Training
# -----------------------

best_loss = float("inf")

for epoch in range(epochs):

    print(
        f"\nEpoch {epoch+1}/{epochs}"
    )

    model.train()

    running_loss = 0

    for batch_idx, images in enumerate(loader):

        images = images.to(device)

        reconstruction, target, mask = model(
            images
        )

        loss = mim_loss(
            reconstruction,
            target,
            mask
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:

            print(
                f"Batch {batch_idx}/{len(loader)} "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = running_loss / len(loader)

    print(
        f"Epoch Loss: {avg_loss:.4f}"
    )

    print(
        f"Learning Rate: {scheduler.get_last_lr()[0]:.7f}"
    )

    scheduler.step()

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            "checkpoints/mim_pretrain_v2.pth"
        )

        torch.save(
            model.encoder.state_dict(),
            "checkpoints/mim_encoder_v2.pth"
        )

        print(
            "Best model saved!"
        )

        print(
            "Encoder saved!"
        )

print(
    "Training finished!"
)