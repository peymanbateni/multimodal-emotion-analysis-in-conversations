import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import pdb

class AttentionConvolutionNetwork(nn.Module):

    def __init__(self, input_size):
        super.(AttentionConvolutionNetwork, self).__init__()
        self.input_size = input_size

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3)
            nn.MaxPool2d(2, stride=2)
            nn.ReLU(True)
            nn.Conv2d(8, 10, kernel_size=3)
            nn.MaxPool2d(2, stride=2)
            nn.ReLU(True)
            nn.Linear(10 * 3 * 3, 32)
            nn.ReLU(True)
            nn.Linear(32, 6)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3)
            nn.ReLU(True)
            nn.Conv2d(10, 10 kernel_size=3)
            nn.MaxPool2d(2, stride=2)
            nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 10 kernel_size=3)
            nn.ReLU(True)
            nn.Conv2d(10, 10, kernel_size=3)
            nn.MaxPool2d(2, stride=2)
            nn.ReLU(True)
        )

        self.dropout = nn.Dropout2d()

        self.fc_1 = nn.Linear(180, 50)
        self.fc_2 = nn.Linear(50, 7)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        theta = self.localization(x)
        theta = theta.view(-1, 2, 3)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.softmax(x)

        return x
