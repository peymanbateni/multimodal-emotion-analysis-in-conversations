from __future__ import print_function
import argparse
import time
import torch
import torch.backends.cudnn as cudnn
import os
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from models import frame_attention_network
from models import visual_features

def load_parameter(_structure, _parameterDir):

    checkpoint = torch.load(_parameterDir)
    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = _structure.state_dict()

    for key in pretrained_state_dict:
        if ((key == 'module.fc.weight') | (key == 'module.fc.bias')):

            pass
        else:
            model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]

    _structure.load_state_dict(model_state_dict)
    model = torch.nn.DataParallel(_structure).cuda()

    return model

# at_type: 0 = self-ref, 1 = interref

class ExpressionDetector(torch.nn.Module):

    def __init__(self):
        structure = frame_attention_network.resnet18_AT(at_type=0)
        parameter_dir = './parameters/Resnet18_FER+_pytorch.pth.tar'
        self.frame_attention_network = load_parameter(structure, parameter_dir)
        self.face_detecor = visual_features.FaceModule()

    def forward(self, x):
        transcript, video, audio, speakers = x
        faces_vector = self.face_detecor(video)
        return self.frame_attention_network(faces_vector)

print(model)
