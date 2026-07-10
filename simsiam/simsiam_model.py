import torch
import torch.nn as nn
import timm


class SimSiam(nn.Module):

    def __init__(self,model_name="swin_tiny_patch4_window7_224"):
        super().__init__()
        self.encoder = timm.create_model(model_name,pretrained=True,num_classes=0)

        self.projector = nn.Sequential(
            nn.Linear(768,2048), # 768 -> 2048
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048,2048), # 2048 -> 2048
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048,2048), # 2048 -> 2048
            nn.BatchNorm1d(2048) # Add BatchNorm1d after the last Linear layer from the paper
        )

        self.predictor = nn.Sequential(
            # 2048 -> 512 -> 2048
            nn.Linear(2048,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,2048)
        )

    def forward(self, x1, x2):

        f1 = self.encoder(x1)
        f2 = self.encoder(x2)

        z1 = self.projector(f1)
        z2 = self.projector(f2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # detach z1 and z2 to avoid computing gradients for them(stop gradient)
        return p1, p2, z1.detach(), z2.detach() 