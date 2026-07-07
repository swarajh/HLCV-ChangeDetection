import torch


def mim_loss(
    reconstruction,
    target,
    mask
):
    """
    SimMIM reconstruction loss.

    Computes L1 loss only on the masked pixels,
    following the original SimMIM paper.
    """

    # ---------------------------------------
    # Pixel-wise L1 reconstruction loss
    # ---------------------------------------

    loss = torch.abs(
        reconstruction - target
    )

    # ---------------------------------------
    # Only masked pixels contribute
    #
    # mask:
    # 1 -> visible
    # 0 -> masked
    # ---------------------------------------

    loss = loss * (1 - mask)

    # ---------------------------------------
    # Average over masked pixels only
    # ---------------------------------------

    loss = loss.sum()

    num_masked_pixels = (
        (1 - mask).sum() * 3
    )

    loss = loss / (
        num_masked_pixels + 1e-8
    )

    return loss