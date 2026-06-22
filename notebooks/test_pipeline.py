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

for image_a, image_b, label in loader:
    out = model(image_a, image_b)
    print(image_a.shape, image_b.shape, label.shape)
    print(out.shape, label.shape)
    
    break