import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import TwoLayerMLP


class Vision(nn.Module):
    def __init__(self, input_channels):

        """
        4-layer cnn with batch-norm and residual connections
        """

        super(Vision, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)

    def forward(self, x):
        x1 = self.conv1(x)
        x = F.relu(self.bn1(x1))

        x2 = self.conv2(x)
        x = F.relu(self.bn2(x2))

        x3 = self.conv3(x)
        x = F.relu(self.bn3(x3))

        x4 = self.conv4(x)
        x = F.relu(self.bn4(x4))

        return x
