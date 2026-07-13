from torch.utils.data import Dataset
from PIL import Image

import os

from torchvision import transforms


class EuroSATSimSiamDataset(Dataset):

    def __init__(self, root_dir, model_name=None):
        self.image_paths = []

        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir,class_name)
            if not os.path.isdir(class_dir):
                continue
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir,image_name)
                self.image_paths.append(image_path)
        
        image_size = 224

        if model_name is not None and "dinov2" in model_name.lower():
            image_size = 518

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size,scale=(0.6, 1.0)), 
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