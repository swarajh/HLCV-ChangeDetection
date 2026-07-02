import torch
import torch.nn.functional as F


def negative_cosine_similarity(p, z):

    z = F.normalize(z,dim=1)
    p = F.normalize(p,dim=1)
    return -(p * z).sum(dim=1).mean()


def simsiam_loss(p1,p2,z1,z2):
    #  Here z1 and z2 are detached from the computation graph, so we don't backpropagate through them
    loss1 = negative_cosine_similarity(p1,z2)
    loss2 = negative_cosine_similarity(p2,z1)
    loss = (loss1 + loss2) / 2
    return loss