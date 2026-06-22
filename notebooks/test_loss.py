import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from change_detection.dataset import LevirDataset
from change_detection.baseline_model import SiameseBaseline

dataset = LevirDataset(
    "datasets/LEVIR-CD/train"
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

model = SiameseBaseline()

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for image_a, image_b, label in loader:
    out = model(image_a, image_b)
    print(image_a.shape, image_b.shape, label.shape)
    print(out.shape, label.shape)

    loss = criterion(out, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Loss:", loss.item())
    
    
    break