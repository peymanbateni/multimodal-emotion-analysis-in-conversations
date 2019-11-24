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
import numpy as np
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

    def __init__(self, parameter_path):
        super(ExpressionDetector, self).__init__()
        structure = frame_attention_network.resnet18_AT(at_type='self-attention') #or relation-attention
        #print(structure)
        parameter_dir = parameter_path
        self.frame_attention_network = load_parameter(structure, parameter_dir)
        print(self.frame_attention_network)
        self.face_detecor = visual_features.FaceModule()

    def forward(self, x):
        transcript, videos, audio, speakers = x

        # USING FACE DETECTOR 
        
        faces_vector = self.face_detecor(videos)
        emotion_output = []
        for faces in faces_vector:
            # note each of these is all the faces in one utterances (N, C, W, H)
            emotions = self.frame_attention_network(faces.squeeze(0))
            summed_emotions = torch.sum(emotions, axis=0)
            print(len(faces))
            print(emotions)
            print(summed_emotions)
            emotion_output.append(summed_emotions.unsqueeze(0))
        
        # placeholder:
        sentiment_output = torch.tensor(np.zeros(len(videos)), dtype=torch.float)
        return (torch.cat(emotion_output), sentiment_output)
