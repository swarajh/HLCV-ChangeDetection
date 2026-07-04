import torch
import torch.nn as nn


class SiameseBaseline(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
        
        # Input: 3*256*256
        nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1),
        # Output: 32*256*256
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Input: 32*128*128
        nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
        # Output: 64*128*128
        nn.ReLU(),
        nn.MaxPool2d(2))

        self.decoder=nn.Sequential(
            #input: 64*64*64
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            #output: 32*64*64
            nn.ReLU(),
            #input: 32*64*64
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, padding=0))
            #output: 1*64*64

    def forward(self, image_a, image_b):
        encoded_a=self.encoder(image_a)
        encoded_b=self.encoder(image_b)

        diff=torch.abs(encoded_a-encoded_b)
        target_size = image_a.size()[2:]  # Get the height and width of the input images
        out=self.decoder(diff)

        out=torch.nn.functional.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
        return out
