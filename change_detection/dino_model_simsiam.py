import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from ssl_training import Projector


class DINOProjectorChangeDetector(nn.Module):

    def __init__(self,model_name="vit_large_patch14_dinov2.lvd142m",projector_weights=None,freeze_backbone=True):
        super().__init__()
        self.backbone = timm.create_model(model_name,pretrained=True,num_classes=0)

        embed_dim = self.backbone.embed_dim

        print("DINOv2 embedding dimension:", embed_dim)

        self.projector = Projector(embed_dim)

        if projector_weights is not None:
            checkpoint = torch.load(projector_weights,map_location="cpu")

            projector_state = {
                k.replace("projector.", ""): v
                for k, v in checkpoint.items()
                if k.startswith("projector.")
            }

            missing, unexpected = self.projector.load_state_dict(projector_state,strict=False)

            print("Loaded SimSiam projector weights")
            print("Missing projector keys:", missing)
            print("Unexpected projector keys:", unexpected)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.decoder = nn.Sequential(

            nn.Conv2d(2048,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,1,kernel_size=1)
        )



    def forward(self, image_a, image_b):

        feat_a = self.backbone.forward_features(image_a)
        feat_b = self.backbone.forward_features(image_b)

        if feat_a.shape[1] > 256:
            feat_a = feat_a[:,1:,:]
            feat_b = feat_b[:,1:,:]

        B, N, C = feat_a.shape

        feat_a = self.projector(feat_a.reshape(-1,C))

        feat_b = self.projector(feat_b.reshape(-1,C))

        feat_a = feat_a.reshape(B,N,2048)

        feat_b = feat_b.reshape(B,N,2048)

        H = W = int(N ** 0.5)

        feat_a = feat_a.transpose(1,2).reshape(B,2048,H,W)

        feat_b = feat_b.transpose(1,2).reshape(B,2048,H,W)

        diff = torch.abs(feat_a - feat_b)

        out = self.decoder(diff)

        out = F.interpolate(out,size=(518,518),mode="bilinear",align_corners=False)
        return out