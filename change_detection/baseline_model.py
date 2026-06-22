import torch
import torch.nn as nn


class SiameseBaseline(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(

        nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        ),

        nn.ReLU(),

        nn.MaxPool2d(2),

        nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        ),

        nn.ReLU(),

        nn.MaxPool2d(2))

        self.decoder=nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, padding=0))
    def forward(self, image_a, image_b):
        encoded_a=self.encoder(image_a)
        encoded_b=self.encoder(image_b)

        diff=torch.abs(encoded_a-encoded_b)

        out=self.decoder(diff)

        
        out=torch.nn.functional.interpolate(out, size=(256,256), mode="bilinear", align_corners=False)
        return out
