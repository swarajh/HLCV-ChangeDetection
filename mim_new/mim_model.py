import torch
import torch.nn as nn
import torch.nn.functional as F

import timm


class SimMIM(nn.Module):

    def __init__(
        self,
        mask_ratio=0.75
    ):

        super().__init__()

        self.mask_ratio = mask_ratio

        # -----------------------
        # Swin Encoder
        # -----------------------

        self.encoder = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            features_only=True
        )

        # --------------------------------------------------
        # SimMIM Paper:
        # One linear layer predicts one RGB patch.
        #
        # Last Swin feature:
        # 7 x 7 x 768
        #
        # Each token predicts
        # 32 x 32 x 3 = 3072 values
        # --------------------------------------------------

        self.decoder = nn.Linear(
            768,
            32 * 32 * 3
        )

    # --------------------------------------------------
    # Create 32x32 Patch Mask
    # --------------------------------------------------

    def create_patch_mask(
        self,
        images
    ):

        B, C, H, W = images.shape

        patch_size = 32

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
            >
            self.mask_ratio
        ).float()

        mask = F.interpolate(
            patch_mask,
            size=(H, W),
            mode="nearest"
        )

        masked_images = images * mask

        return (
            masked_images,
            mask
        )

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward(
        self,
        images
    ):

        masked_images, mask = self.create_patch_mask(
            images
        )

        features = self.encoder(
            masked_images
        )

        latent = features[-1]

        # timm:
        # B,H,W,C

        B, H, W, C = latent.shape

        # ---------------------------------------
        # Linear prediction
        # ---------------------------------------

        latent = latent.reshape(
            B,
            H * W,
            C
        )

        reconstruction = self.decoder(
            latent
        )

        # ---------------------------------------
        # Reshape into image
        # ---------------------------------------

        reconstruction = reconstruction.view(
            B,
            H,
            W,
            32,
            32,
            3
        )

        reconstruction = reconstruction.permute(
            0,
            5,
            1,
            3,
            2,
            4
        )

        reconstruction = reconstruction.reshape(
            B,
            3,
            H * 32,
            W * 32
        )

        # Crop because:
        # 7*32 = 224 exactly,
        # but crop keeps code generic.

        reconstruction = reconstruction[
            :,
            :,
            :224,
            :224
        ]

        return (
            reconstruction,
            images,
            mask
        )