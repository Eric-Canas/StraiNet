"""
This class implements a loss function for rotation degrees.
Rotation degrees are originally converted from the range [0, 360) to the range [0, 1).
The model predicts a number in the range (-1, 1)
The loss function is the mean squared error between the predicted rotation degrees and the target rotation degrees,
but taking into account the fact that the rotation degrees are cyclic. So, for example 90 degrees can be considered
equivalent to 0.25 (90/360) or -0.75 (270/360).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RotationLoss(nn.Module):
    def __init__(self):
        super(RotationLoss, self).__init__()

    def forward(self, pred, target):
        """
        Computes the loss between the predicted rotation degrees and the target rotation degrees.

        :param pred: Tensor. The predicted rotation degrees in the range (-1, 1). Shape: (N, 1)
        :param target: Tensor. The target rotation degrees in the range (0, 360). Shape: (N, 1)
        :return: Tensor. The loss. Shape: (1,)
        """
        # Convert the target rotation degrees from the range (0, 1) to the range (0, 360)
        target = target
        # Convert the predicted rotation degrees from the range (-1, 1) to the range (0, 360)
        pred = (pred + 1) * 180
        # Compute the difference between the target rotation degrees and the predicted rotation degrees
        diff = torch.abs(target - pred)
        # Compute the loss
        loss = torch.mean(torch.min(diff, 360 - diff))
        return loss

