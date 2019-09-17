import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def nll_mask_loss(output, target, mask):
    pass
