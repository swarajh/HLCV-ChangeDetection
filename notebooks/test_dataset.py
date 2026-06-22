from torch.utils.data import DataLoader
from change_detection.dataset import LevirDataset

dataset=LevirDataset("datasets/LEVIR-CD/train")

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)



for image_a, image_b, label in loader:
    print(image_a.shape, image_b.shape, label.shape)
    break