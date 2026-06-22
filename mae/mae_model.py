import torch
import torch.nn as nn
import torch.nn.functional as F

import timm


class MAE(nn.Module):

    def __init__(
        self,
        mask_ratio=0.75
    ):
        super().__init__()

        self.mask_ratio = mask_ratio

        self.encoder = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            features_only=True
        )

        self.decoder = nn.Sequential(

            nn.Conv2d(
                768,
                512,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                512,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                256,
                3,
                kernel_size=1
            )
        )

    
    def create_patch_mask(
        self,
        images
    ):

        B, C, H, W = images.shape

        patch_size = 4

        num_patches_h = H // patch_size
        num_patches_w = W // patch_size

        patch_mask = (
        torch.rand(
            B,
            1,
            num_patches_h,
            num_patches_w,
            device=images.device
        )
            > self.mask_ratio
        ).float()

        mask = torch.nn.functional.interpolate(
        patch_mask,
        size=(H, W),
        mode="nearest"
    )

        masked_images = images * mask

        return masked_images, mask

    def forward(
        self,
        images
    ):

        masked_images, mask = (
            self.create_patch_mask(images)
        )

        features = self.encoder(
            masked_images
        )

        latent = features[-1]

        latent = latent.permute(
            0, 3, 1, 2
        )

        reconstruction = self.decoder(
            latent
        )

        reconstruction = F.interpolate(
            reconstruction,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        return (
            reconstruction,
            images,
            mask
        )
    