import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from change_detection.dataset import LevirDataset
from change_detection.baseline_model import SiameseBaseline
from change_detection.losses import DiceLoss


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

print("Train:", len(train_dataset))
print("Val:", len(val_dataset))
model = SiameseBaseline()
bce_loss = nn.BCEWithLogitsLoss()
dice_loss = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs=10
for epoch in range(epochs):
    for image_a, image_b, label in train_loader:
        out = model(image_a, image_b)
        loss = bce_loss(out, label) + dice_loss(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
torch.save(
    model.state_dict(),
    "checkpoints/baseline.pth"
)

print("Model saved!")



val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False
)


