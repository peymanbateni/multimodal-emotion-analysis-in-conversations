import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import pdb

class AttentionConvolutionNetwork(nn.Module):

    def __init__(self):
        super(AttentionConvolutionNetwork, self).__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=(3,3)),
            nn.MaxPool2d(4, stride=4),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=(3,3)),
            nn.MaxPool2d(3, stride=3),
            nn.ReLU(True)
        )

        self.localization_fc = nn.Sequential(
            nn.Linear(90, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )

        self.localization_fc[2].weight.data.zero_()
        self.localization_fc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.dropout = nn.Dropout2d()

        self.emo_fc_1 = nn.Linear(810, 50)
        self.emo_fc_2 = nn.Linear(50, 7)
        self.emo_softmax = nn.Softmax(dim=1)

        self.sent_fc_1 = nn.Linear(810, 50)
        self.sent_fc_2 = nn.Linear(50, 3)
        self.sent_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #print(x.shape)
        loc = self.localization(x)
        loc = loc.view(-1, 90)
        #print(loc.shape)
        theta = self.localization_fc(loc)
        theta = theta.view(-1, 2, 3)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)

        #print(theta.shape)
        #print(x.shape)
        #print(x.size())

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        #print(x.shape)
        x = x.view(-1, 810)

        emotion_x = self.emo_fc_1(x)
        emotion_x = self.emo_fc_2(emotion_x)
        emotions = self.emo_softmax(emotion_x)
        #print(emotions.shape)

        sentiment_x = self.sent_fc_1(x)
        sentiment_x = self.sent_fc_2(sentiment_x)
        sentiments = self.sent_softmax(sentiment_x)

        # Placeholder of sentiment

        return emotions, sentiments
