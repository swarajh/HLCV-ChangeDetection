import torch


def mim_loss(
    reconstruction,
    target,
    mask
):

    loss = (
        reconstruction - target
    ) ** 2

    # Only compute loss on masked pixels

    loss = loss * (
        1 - mask
    )

    return loss.mean()