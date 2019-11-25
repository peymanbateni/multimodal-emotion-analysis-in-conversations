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
from models import attention_convolution_network

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
        self.frame_attention_network =  structure #load_parameter(structure, parameter_dir)
        print(self.frame_attention_network)
        #self.face_detecor = visual_features.FaceModule()

    def forward(self, x):
        transcript, faces_vector, audio, speakers = x
        # USING FACE DETECTOR 
        
        #faces_vector = self.face_detecor(videos)
        emotion_output = []
        for i, faces in enumerate(faces_vector):
            # note each of these is all the faces in one utterances (N, C, W, H)
            if (faces.size(1) != 0):
                faces = faces.cuda()
                print("faces size", faces.size())
                print(faces.is_cuda)
                emotions = self.frame_attention_network(faces.squeeze(0))
                print(emotions.size())
                summed_emotions = torch.sum(emotions, axis=0)
                print(summed_emotions.size())
            #print(len(faces))
            #print(emotions)
            #print(summed_emotions)
                emotion_output.append(summed_emotions.unsqueeze(0))
                faces = faces.cpu()
            else:
                emotion_output.append(torch.zeros(1, 7).cuda())

        print(len(emotion_output))

        # placeholder:
        sentiment_output = torch.zeros(len(emotion_output), 3).cuda()
        emotion_output = torch.cat(emotion_output, dim=0)
        print("EMOTION", emotion_output.size())
        print("SENTIMENT", sentiment_output.size())
        return emotion_output, sentiment_output

class AttentionConvWrapper(torch.nn.Module):

    def __init__(self):
        super(AttentionConvWrapper, self).__init__()
        self.model = attention_convolution_network.AttentionConvolutionNetwork()

    def forward(self, x):
        transcript, face_vector, audio, speakers = x

        #print(face_vector[0].shape)

        emotion_output = []
        sentiment_output = []
        for faces in face_vector:
            _, N, F, C, W, H = faces.shape

            face_stack = faces.squeeze(0).view(N * F, C, W, H).to("cuda")
            emotions, sentiments = self.model(face_stack)

            # TODO: placeholder aggregation method 
            emotion_output.append(torch.max(emotions, dim=0).values.unsqueeze(0))
            sentiment_output.append(torch.max(sentiments, dim=0).values.unsqueeze(0))

            #print(emotions)
        #print(emotion_output[0].shape)
        return torch.cat(emotion_output), torch.cat(sentiment_output)