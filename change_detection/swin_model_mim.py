import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SwinChangeDetector(nn.Module):

    def __init__(self,pretrained=True,mim_weights=None):
        super().__init__()

        # TODO: Add support for other Swin variants (e.g., Swin-S, Swin-B, etc.) and make the model selection configurable.
        self.encoder = timm.create_model("swin_tiny_patch4_window7_224",pretrained=pretrained,features_only=True)

        # -----------------------
        # Load SimMIM Encoder
        # -----------------------

        if mim_weights is not None:

            checkpoint = torch.load(mim_weights,map_location="cpu")

            missing, unexpected = (self.encoder.load_state_dict(checkpoint,strict=False))

            print("Loaded SimMIM weights")

            print("Missing keys:",len(missing))

            print("Unexpected keys:",len(unexpected))

            print("\nFirst 10 Missing:")
            for k in missing[:10]:
                print(k)

            print("\nFirst 10 Unexpected:")
            for k in unexpected[:10]:
                print(k)

        # -----------------------
        # Fusion
        # -----------------------

        self.fusion = nn.Sequential(

            nn.Conv2d(96 + 192 + 384 + 768,512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,256,kernel_size=3,padding=1),
            nn.ReLU()
        )

        # -----------------------
        # Decoder
        # -----------------------

        self.decoder = nn.Sequential(

            nn.Conv2d(256,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,1,kernel_size=1)
        )

    def forward(self,image_a,image_b):

        features_a = self.encoder(image_a)

        features_b = self.encoder(image_b)

        diffs = []

        for feat_a, feat_b in zip(features_a,features_b):

            feat_a = feat_a.permute(0, 3, 1, 2)

            feat_b = feat_b.permute(0, 3, 1, 2)

            diff = torch.abs(feat_a - feat_b)

            diffs.append(diff)


        # TODO: Can we make this more efficient? Maybe use a loop or a list comprehension to handle the interpolation of diffs[1], diffs[2], and diffs[3] to size (56, 56) instead of hardcoding each one.
        diff0 = diffs[0]
        diff1 = diffs[1]
        diff2 = diffs[2]
        diff3 = diffs[3]

        diff1 = F.interpolate(diff1,size=(56, 56),mode="bilinear",align_corners=False)

        diff2 = F.interpolate(diff2,size=(56, 56),mode="bilinear",align_corners=False)

        diff3 = F.interpolate(diff3,size=(56, 56),mode="bilinear",align_corners=False)

        fused = torch.cat([diff0,diff1,diff2,diff3],dim=1)

        fused = self.fusion(fused)

        out = self.decoder(fused)

        out = F.interpolate(out,size=(224, 224),mode="bilinear",align_corners=False)

        return out