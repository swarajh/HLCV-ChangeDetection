from torch.utils.data import Dataset
import torch
import numpy as np
import os
from PIL import Image

from torchvision import transforms


class LevirDataset(Dataset):
    def __init__(self, root_dir, image_size=224):
        self.root_dir=root_dir
        self.a_dir=os.path.join(root_dir,"A")
        self.b_dir=os.path.join(root_dir,"B")
        self.label_dir=os.path.join(root_dir,"label")
        self.image_names=sorted(os.listdir(self.a_dir))
        self.transform=transforms.Compose([transforms.Resize((image_size,image_size)), 
                                           transforms.ToTensor()])

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self,idx):
        image_name=self.image_names[idx]
        image_a=Image.open(os.path.join(self.a_dir,image_name)).convert("RGB")
        image_b=Image.open(os.path.join(self.b_dir,image_name)).convert("RGB")
        label=Image.open(os.path.join(self.label_dir,image_name)).convert("L")

        image_a=self.transform(image_a)
        image_b=self.transform(image_b)
        label=self.transform(label)
        label = (label > 0).float()

        return image_a, image_b, label
    

        

