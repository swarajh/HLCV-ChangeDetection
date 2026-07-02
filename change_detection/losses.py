import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,logits,targets):

        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = ( probs * targets).sum()

        dice = ( 2 * intersection + 1e-6) / (probs.sum()+ targets.sum()+ 1e-6)

        return 1 - dice