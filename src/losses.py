import torch
import torch.nn.functional as F


def masked_anchor_losses(predictions, targets, masks, pos_weights):
    """Independent missing-value masks; empty targets contribute exactly zero."""
    valid = masks[:, 0].bool()
    t_size = (
        F.smooth_l1_loss(predictions[valid, 0], targets[valid, 0])
        if valid.any() else predictions[:, 0].sum() * 0.0
    )
    binary = []
    for column, weight in ((1, pos_weights[0]), (2, pos_weights[1])):
        valid = masks[:, column].bool()
        value = (
            F.binary_cross_entropy_with_logits(
                predictions[valid, column], targets[valid, column], pos_weight=weight
            )
            if valid.any() else predictions[:, column].sum() * 0.0
        )
        binary.append(value)
    return t_size, binary[0], binary[1]


def soft_cross_entropy(logits, probability_lusc, weights=None):
    targets = torch.stack([1.0 - probability_lusc, probability_lusc], dim=1)
    per_sample = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1)
    if weights is None:
        return per_sample.mean()
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)

