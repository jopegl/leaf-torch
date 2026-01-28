import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_mse_loss(pred, targets, mask):
    difference = (pred - targets) ** 2
    masked_difference = difference * mask
    loss = masked_difference.sum() / mask.sum()
    return loss
