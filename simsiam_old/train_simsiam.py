import os

import torch

from torch.utils.data import DataLoader

from simsiam.eurosat_dataset import (
    EuroSATSimSiamDataset
)

from simsiam.simsiam_model import (
    SimSiam
)

from simsiam.simsiam_loss import (
    simsiam_loss
)


# --------------------
# Device
# --------------------

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(
    "Using device:",
    device
)


# --------------------
# Dataset
# --------------------

dataset = EuroSATSimSiamDataset(
    "datasets/EuroSAT"
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

print(
    "Dataset Size:",
    len(dataset)
)


# --------------------
# Model
# --------------------

model = SimSiam()

model = model.to(device)


# --------------------
# Optimizer
# --------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)


# --------------------
# Training
# --------------------

epochs = 50

best_loss = float("inf")

os.makedirs(
    "checkpoints",
    exist_ok=True
)

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for view1, view2 in loader:

        view1 = view1.to(device)
        view2 = view2.to(device)

        p1, p2, z1, z2 = model(
            view1,
            view2
        )

        loss = simsiam_loss(
            p1,
            p2,
            z1,
            z2
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = (
        running_loss
        /
        len(loader)
    )

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss: {avg_loss:.4f}"
    )

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            "checkpoints/simsiam_swin.pth"
        )

        print(
            "Best model saved!"
        )
        torch.save(
            model.encoder.state_dict(),
            "checkpoints/simsiam_encoder.pth"
        )

        print("Encoder saved!")