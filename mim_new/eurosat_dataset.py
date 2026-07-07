import os

from PIL import Image

from torch.utils.data import Dataset

import torchvision.transforms as transforms


class EuroSATMIMDataset(Dataset):

    def __init__(
        self,
        root_dir
    ):

        self.image_paths = []

        for class_name in os.listdir(root_dir):

            class_dir = os.path.join(
                root_dir,
                class_name
            )

            if not os.path.isdir(class_dir):
                continue

            for image_name in os.listdir(class_dir):

                self.image_paths.append(
                    os.path.join(
                        class_dir,
                        image_name
                    )
                )

        # --------------------------------------
        # SimMIM / ImageNet-style Augmentation
        # --------------------------------------

        self.transform = transforms.Compose([

            transforms.RandomResizedCrop(
                size=224,
                scale=(0.67, 1.0)
            ),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):

        return len(
            self.image_paths
        )

    def __getitem__(
        self,
        idx
    ):

        image = Image.open(
            self.image_paths[idx]
        ).convert("RGB")

        image = self.transform(
            image
        )

        return image