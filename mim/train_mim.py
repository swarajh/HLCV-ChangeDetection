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

from torch.utils.data import DataLoader #for batches

from mim.eurosat_dataset import (EuroSATMIMDataset)

from mim.mim_model import (SimMIM)

from mim.mim_loss import (mim_loss)


if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:",device)




dataset = EuroSATMIMDataset("datasets/EuroSAT")

loader = DataLoader(dataset,batch_size=16,shuffle=True)

print("Dataset Size:",len(dataset))

print("Number of batches:",len(loader))

model = SimMIM(mask_ratio=0.75)

model = model.to(device)

print("Model created")


optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4) #it combines adaptive learning rates with decoupled weight decay for better regularization.


os.makedirs("checkpoints",exist_ok=True)



epochs = 20

best_loss = float("inf")

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for batch_idx, images in enumerate(loader):

        images = images.to(device)

        reconstruction, target, mask = model(images)

        loss = mim_loss(reconstruction,target,mask)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:

            print(f"Batch {batch_idx}/{len(loader)} "f"Loss: {loss.item():.4f}",flush=True)

    avg_loss = (running_loss/len(loader))

    print(f"Epoch {epoch+1}/{epochs} "f"Loss: {avg_loss:.4f}",flush=True)

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(model.state_dict(),"checkpoints/mim_pretrain.pth")

        print("Best model saved!")
        torch.save(model.encoder.state_dict(),"checkpoints/mim_encoder.pth")       

        print("Encoder saved!")

print("Training finished!")
