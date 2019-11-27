from __future__ import print_function
import argparse
import time
import torch
import torch.backends.cudnn as cudnn
import os
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as functional
import torch.optim
import torch.utils.data
from torchvision.transforms import ToPILImage
import numpy as np
from PIL import Image
from models import frame_attention_network
from models import visual_features
from models import attention_convolution_network
from facenet_pytorch_local.models.inception_resnet_v1 import InceptionResnetV1

class FCProj(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(FCProj, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, input_size)
        self.linear2 = torch.nn.Linear(input_size, output_size)
        self.elu = torch.nn.ELU()

    def forward(self, x):
        return self.linear2(self.elu(self.linear1(x)))

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

    def __init__(self, parameter_path, face_matching=False):
        super(ExpressionDetector, self).__init__()
        structure = frame_attention_network.resnet18_AT(at_type='self-attention') #or relation-attention
        #print(structure)
        parameter_dir = parameter_path
        self.frame_attention_network =  structure #load_parameter(structure, parameter_dir)

        self.face_matching = face_matching
        if self.face_matching:
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

        self.attention_network = FCProj(7, 7)
        self.classifier = torch.nn.Linear(7,7) #FCProj(512, 7)
        #self.projector = torch.nn.Linear(512, 100)
        self.softmax = torch.nn.Softmax()
        #print(self.frame_attention_network)
        #self.face_detecor = visual_features.FaceModule()

    def get_face_matchings(self, face_tensor):
        """
        Method for generating face embeddings based on the Facenet model.
        Input:
            face_tensor(torch.tensor(N, F, C, W, H)): dimensions returned by the MTCNN detctor

        Ouput:
            torch.tensor(N,F): N - number of frames, F face per frame
        """

        # first generate facenet embeddings:

        _, N, F, C, W, H, = face_tensor.shape
        face_tensor = face_tensor.view(-1, C, W, H)

        # Instantiate facenet model
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
        # Generate the embeddings for all faces
        embeddings = resnet(face_tensor).view(N, F, -1)

        # Create a similarity matrix comparing each face with each other face in 
        # all possible frames 

        sim_matrix = []
        for idx, embedding_faces in enumerate(embeddings):
            distances = embedding_faces.view(F, 1, 1, 512) - embeddings #torch.cat((embeddings[:idx], embeddings[idx + 1:]))
            #distances is now shape (N - 1, F, F, 512) 

            norms = torch.norm(distances, p=2, dim=3).permute(1, 0, 2)
            sim_matrix.append(norms)

        sim_matrix = torch.stack(sim_matrix)

        #print(sim_matrix)

        # sim_matrix is now tensor(N, N, F, F)
        face_tensor = face_tensor.view(N, F, C, W, H)

        aligned_tensor = []

        for idx, frame in enumerate(sim_matrix):

            # Find the closest match (value, index) for each face in each other frame

            match_values, match_indices = torch.min(frame, dim=2)

            # We need to ensure that it does not pick a face from the frame we are on

            mask = torch.zeros(N)
            mask[idx] = 100.0
            match_values = match_values + mask.view(N, 1)
            #match_values = tensor(N, F)

            # Aggregate the by finding the closest match for each face across all frames

            best_match_index = torch.argmin(match_values, dim=0)

            # min_orders is tensor(F)

            # this next line of code is kind of funky but we need torch.take can only 
            # be applied to 1D tensor, those we have to compensate by multiplying the 
            # indices of the 2D tensor by a given offset determined by the tensor size
            best_match_index = best_match_index + (torch.arange(0, F) * N)
            # need to apply a transpose to match_indices to get it into the proper ordering
            # for when torch.take squishes it
            face_ordering = torch.take(torch.t(match_indices), best_match_index)

            # retrieve original indices of faces for the 

            #print(face_ordering)
            # reorder the faces
            ordered_tensor = torch.index_select(face_tensor[idx], 0, face_ordering)
            # reconstruct the video tensor 
            aligned_tensor.append(ordered_tensor)   

        aligned_tensor = torch.stack(aligned_tensor)
        return aligned_tensor.unsqueeze(0)

        # SAVE IMAGES FOR DEBUGGING PURPOSES:

        #image_converter = ToPILImage()
        #for fdx, face in enumerate(aligned_tensor.squeeze(0).view(F, N, C, W, H)):
        #    for jdx, frame in enumerate(face):
        #        img = image_converter(frame)
        #        file_path = "test/face_{}_image{}.jpeg".format(fdx, jdx)
        #        img.save(file_path)

        #print("ALL SAVED")
        #print(aligned_tensor.shape)


    def forward(self, x):
        #transcript, faces_vector, audio, speakers = x

        faces_vector = x
        # USING FACE DETECTOR 
        
        #faces_vector = self.face_detecor(videos)
        emotion_output = []
        for i, faces in enumerate(faces_vector):
            # note each of these is all the faces in one utterances (N, C, W, H)
            if (faces.size(1) != 0):
                if self.face_matching and faces.size(1) > 1:
                    faces = self.get_face_matchings(faces)

                _, N, F, C, W, H = faces.shape
                faces = faces.view(F, N, C, W, H).cuda()
                #print("faces size", faces.size())
                #print(faces.is_cuda)
                #emotions = self.frame_attention_network(faces.squeeze(0))
                #print(emotions.size())
                #summed_emotions = torch.sum(emotions, axis=0)
                #print(summed_emotions.size())
                #emotion_output.append(summed_emotions.unsqueeze(0))
                emotions = self.frame_attention_network(faces)
                #print(emotions.shape)
                attention_weights = self.softmax(self.attention_network(emotions))
                #print(attention_weights.shape)
                #print(emotions.shape)
                predicted_emotions = torch.bmm(torch.t(attention_weights).unsqueeze(1), torch.t(emotions).unsqueeze(2))
                #predicted_emotions = self.classifier(summed_emotions)
                emotion_output.append(predicted_emotions.view(1, F)) #summed_emotions.unsqueeze(0))
            else:
                emotion_output.append(torch.zeros(1, 7).cuda())

        #print(len(emotion_output))

        # placeholder:
        sentiment_output = torch.zeros(len(emotion_output), 3).cuda()
        emotion_output = torch.cat(emotion_output, dim=0)
        #print("EMOTION", emotion_output.size())
        #print("SENTIMENT", sentiment_output.size())
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

            if N * F == 0:
                pass
                #return torch.zeros(1, 7, dtype=float).to("cuda"), torch.zeros(0, 3, dtype=float).to("cuda")
            if W != 48 or H != 48:
                face_stack = functional.interpolate(face_stack, (48, 48))
            #print(face_stack.shape)

            emotions, sentiments = self.model(face_stack)

            # TODO: placeholder aggregation method 
            emotion_output.append(torch.max(emotions, dim=0).values.unsqueeze(0))
            sentiment_output.append(torch.max(sentiments, dim=0).values.unsqueeze(0))

            #print(emotions)
        #print(emotion_output[0].shape) 
        return torch.cat(emotion_output), torch.cat(sentiment_output)