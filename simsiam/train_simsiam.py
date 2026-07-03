import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from torch.utils.data import DataLoader

from simsiam.eurosat_dataset import (EuroSATSimSiamDataset)

from simsiam.simsiam_model import (SimSiam)

from simsiam.simsiam_loss import (simsiam_loss)

# --------------------
# Device
# --------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

# --------------------
# Dataset
# --------------------

dataset = EuroSATSimSiamDataset("datasets/EuroSAT")

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

print("Dataset Size:", len(dataset))
print("Number of batches:", len(loader))

# --------------------
# Model
# --------------------

model = SimSiam()
model = model.to(device)

# --------------------
# Optimizer
# --------------------

optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)

# --------------------
# Training
# --------------------

epochs = 20

best_loss = float("inf")

os.makedirs("checkpoints",exist_ok=True)

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for batch_idx, (view1, view2) in enumerate(loader):

        view1 = view1.to(device)
        view2 = view2.to(device)

        p1, p2, z1, z2 = model(view1,view2)
        loss = simsiam_loss(p1,p2,z1,z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_idx % 100 == 0:

            print(
                f"Batch {batch_idx}/{len(loader)} "
                f"Loss: {loss.item():.4f}",
                flush=True
            )

    avg_loss = running_loss / len(loader)

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss: {avg_loss:.4f}",
        flush=True
    )

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(),"checkpoints/simsiam_swin.pth")

        torch.save(model.encoder.state_dict(),"checkpoints/simsiam_encoder.pth")

        print("Best model saved!",flush=True)
