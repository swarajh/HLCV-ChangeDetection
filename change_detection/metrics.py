import torch


def compute_iou(pred, target, eps=1e-6):
    """
    pred   : binary prediction [B,1,H,W]
    target : binary ground truth [B,1,H,W]
    """

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()

    union = (
        pred.sum()
        + target.sum()
        - intersection
    )

    iou = (
        intersection + eps
    ) / (
        union + eps
    )

    return iou.item()


def compute_f1(pred, target, eps=1e-6):
    """
    pred   : binary prediction [B,1,H,W]
    target : binary ground truth [B,1,H,W]
    """

    pred = pred.view(-1)
    target = target.view(-1)

    tp = (pred * target).sum()

    fp = (
        pred * (1 - target)
    ).sum()

    fn = (
        (1 - pred) * target
    ).sum()

    precision = (
        tp + eps
    ) / (
        tp + fp + eps
    )

    recall = (
        tp + eps
    ) / (
        tp + fn + eps
    )

    f1 = (
        2 * precision * recall
    ) / (
        precision + recall + eps
    )

    return f1.item()