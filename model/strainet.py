"""
This file implements the Convolutional - Regression model that will receive an image and output the rotation angle.

Email: eric@ericcanas.com
Date: 26-01-2023
Github: https://github.com/Eric-Canas
"""

import torch.nn as nn
# To use the rotate function
from torchvision.models.inception import inception_v3, Inception_V3_Weights


class StraiNet(nn.Module):

    def __init__(self, output_channels=1):
        super(StraiNet, self).__init__()

        #InceptionV3 backbone to extract features (expecting 299x299x3) (it will take care of the preprocessing)
        self.backbone = inception_v3(pretrained=True, aux_logits=True)

        # Build the regression head
        self.regression_head = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_channels),
            nn.Tanh()
        )

    def forward(self, x):
        assert x.shape[1:] == (3, 299, 299), 'Input shape must be (N, 3, 299, 299)'
        assert len(x.shape) == 4, 'Input must be a 4D tensor'
        # Get the features from the backbone
        x = self.backbone(x)
        # Get the features from the auxiliary classifier (remove the aux_logits on training)
        if len(x) == 2:
            x = x[0]
        # Flatten the features
        x = x.view(x.size(0), -1)
        # Pass the features through the regression head
        x = self.regression_head(x)
        return x
