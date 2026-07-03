from torch.utils.data import Dataset
from PIL import Image

import os

from torchvision import transforms


class EuroSATSimSiamDataset(Dataset):

    def __init__(self, root_dir):
        self.image_paths = []

        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir,class_name)
            if not os.path.isdir(class_dir):
                continue
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir,image_name)
                self.image_paths.append(image_path)

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224,scale=(0.6, 1.0)), # TODO: why 224 and not 256?
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.1),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image = Image.open(self.image_paths[idx]).convert("RGB")
        view1 = self.transform(image)
        view2 = self.transform(image)
        return view1, view2