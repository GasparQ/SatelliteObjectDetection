import torch

def intersection_over_union(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute the metric intersection over union (IoU)

    Args:
        outputs (torch.Tensor): Set of predicted values by a model
        labels (torch.Tensor): Set of expected values

    Returns:
        torch.Tensor: Computed IoU for every sample provided
    """
    # compute intersection over Channels (-3), Height (-2) and Width (-1)
    intersection = (outputs & labels).sum((-3, -2, -1))

    # compute union over Channels (-3), Height (-2) and Width (-1)
    union = (outputs | labels).sum((-3, -2, -1))

    # We smooth our devision to avoid 0/0
    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou
