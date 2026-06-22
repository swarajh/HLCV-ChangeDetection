import torch


def mae_loss(
    reconstruction,
    target,
    mask
):

    loss = (
        reconstruction - target
    ) ** 2

    # Only compute loss on masked regions

    loss = loss * (
        1 - mask
    )

    return loss.mean()