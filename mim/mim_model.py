import torch
import torch.nn as nn
import torch.nn.functional as F

import timm


class SimMIM(nn.Module):

    def __init__(self,mask_ratio=0.75): #According to SIMMIM paper we took 75% mask ratio
        super().__init__()

        self.mask_ratio = mask_ratio

        self.encoder = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, features_only=True) #Patch_size=4, window_size=7, input_resolution= 224 features_only=True as we want feature maps

        self.decoder = nn.Sequential(

            nn.Conv2d(768,512,kernel_size=3,padding=1), #the last swin feature map has 768 channels, we reduce to 512
            nn.ReLU(),
            nn.Conv2d(512,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,3,kernel_size=1)
            )

    def create_patch_mask(self,images):

        B, C, H, W = images.shape

        patch_size = 4

        num_patches_h = H // patch_size
        num_patches_w = W // patch_size

        patch_mask = (
            torch.rand(B,1,num_patches_h,num_patches_w,device=images.device) > self.mask_ratio).float()#we generate numbers btw 0 and 1 and numbers which are greater than 0.75 become True and are visible

        mask = F.interpolate(patch_mask,size=(H, W),mode="nearest")#patch size =56*56, we need 224*224

        masked_images = images * mask

        return (masked_images,mask)

    def forward(self,images):

        masked_images, mask = (self.create_patch_mask(images))

        features = self.encoder(masked_images)

        latent = features[-1]#only take the deepest feature map

        latent = latent.permute(0,3,1,2)#CNN expects B,C,H,W timm gives B,H,W,C so we permute

        reconstruction = self.decoder(latent)

        reconstruction = F.interpolate(reconstruction,size=(224, 224),mode="bilinear",align_corners=False)#the decoder gives 7*7 feature map but img size is 224*224, hence we use bilinear interpolation, it estimates new pixels by using 4 nearest neighbouring pixels

        return (reconstruction,images,mask)