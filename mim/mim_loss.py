import torch


def mim_loss(reconstruction,target,mask):

    loss = (reconstruction - target) ** 2 #pixel-wise squared difference between the reconstructed image and the original image.

    # Only compute loss on masked pixels

    loss = loss * (1 - mask) #model should not be rewarded or penalized for reconstructing pixels it already saw, instead it learns by predicting pixels which were hidden 
    #mask contains 1(visible)if visible loss won't be counted for that or 0(hidden)