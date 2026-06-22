import torch

from torch.utils.data import DataLoader

from mae.eurosat_dataset import (
    EuroSATMAEDataset
)

from mae.mae_model import MAE

from mae.mae_loss import (
    mae_loss
)


# --------------------
# Device
# --------------------

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


# --------------------
# Dataset
# --------------------

dataset = EuroSATMAEDataset(
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


# --------------------
# Model
# --------------------

model = MAE()

model = model.to(device)

print(
    "Model created"
)


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

epochs = 1

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for batch_idx, images in enumerate(loader):

        images = images.to(device)

        reconstruction, target, mask = model(
            images
        )

        loss = mae_loss(
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

    avg_loss = (
        running_loss
        /
        len(loader)
    )

    print(
        f"Epoch {epoch+1} "
        f"Loss: {avg_loss:.4f}"
    )