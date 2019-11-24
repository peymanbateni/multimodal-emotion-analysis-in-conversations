"""
Module for generating visual feature extraction.
"""
import cv2
from PIL import Image
from torch_mtcnn import detect_faces
import torch
import os
from facenet_pytorch_local.models.mtcnn import MTCNN
from facenet_pytorch_local.models.inception_resnet_v1 import InceptionResnetV1


# NOT being used right now 
class FaceModule(torch.nn.Module):
    def __init__(self, output_size=224, max_persons=2):
        super(FaceModule, self).__init__()
        self.output_size = output_size
        self.max_persons = max_persons

    def forward(self, video_input):
        #TODO: this still probably is not the best way to do this in a loop
        #faces_vector = [detect_faces_mtcnn(video.squeeze(0), self.max_persons, self.output_size) for video in video_input]
        #print(faces_tensor.size())
        #faces_embeddings = [get_face_embeddings(faces) for faces in faces_vector]
        return video_input
        #print(len(faces_embeddings))
        #print("Got here!")

"""
Currently, two methods for extracting bounding-boxes on faces have been tested,
a prebuilt mtcnn library and an OpenCV method based on Haar Cascades. So far,
mtcnn seems significally slower than the haar cascades method, but may be due
to the fact that it was run on cpu. Haar cascades is fast, but fails at detecting
side profiles and also produces false positives
"""

def detect_faces_mtcnn(video_tensor, max_persons=7, output_size=160, sampling_rate=30, display_images=False):
    """
    Method for detecing faces on an input video Tensor using an implementation of
    MTCNNs.

    Inputs:
        video_tensor(torch.tensor(N, W, H, 3)): N number of images, W width, H height
        max_persons(int): the total number of people to return in output tensor
        output_size(int): in pixels the output size of the images (default 160 for facenet)
        sampling_rate(int): how often to sample images eg 60 -> once every 60 frames
        display_images(bool): debugging features to see output of tensors

    Output:
        torch.tensor(N/sampling_rate, max_persons, 3, output_size, output_size)

        axis-1 = number of images / sampling rate
        axis-2 = each face
        axis-3 = color channels
        axis-4 = width
        axis-5 = height

    NB: output tensor is normalized
    """
    # Istantiate mtcnn detector
    mtcnn = MTCNN(image_size=output_size, margin=0, keep_all=True)

    # Compiling sampling and pass into MTCNN, currently this is quite wasteful
    # as we are converting to numpy array then to PIL image then it gets converted
    # back to torch tensor within the method, TODO: optimize data flow to
    # avoid type casting

    video = []
    for image in video_tensor[::sampling_rate]:
        image_np = image.numpy()
        image_pil = Image.fromarray(image_np)
        video.append(image_pil)
    #print(len(video))
    #print(video[0].size)

    # TODO: for some reason the following call errors out sometimes, might be a bug in the
    # library implementation in which case we might need to clone the repo and modify it ourselves
    images = mtcnn(video)
    #print(len(images))
    #print(images[0].shape)

    # pad with empty tensors if neccesary:
    # in the case where there are fewer than max_persons detected, return
    # tensor appended by 0's
    #if len(images) > max_persons:
    #    images = images[:max_persons]

    #print(images.shape)

    target = torch.zeros(len(images), max_persons, 3, output_size, output_size)
    for idx, image in enumerate(images):
        #TODO: not really sure why but sometimes None is returned so this check is neccesary
        if image is not None:
            target[idx, :image.shape[0]] = image[:max_persons]

    #print(target.shape)
    if display_images:
        for idx, image in enumerate(target):
            for face in image:
                cv2.imshow("Face Found in Frame {}".format(idx), face.permute(1,2,0).numpy())
                cv2.waitKey(0)
    return target

def get_face_embeddings(face_tensor):
    """
    Method for generating face embeddings based on the Facenet model.
    Input:
        face_tensor(torch.tensor(N, F, C, W, H)): dimensions returned by the MTCNN detctor

    Ouput:
        torch.tensor(N,F): N - number of frames, F face per frame
    """
    N, F, C, W, H, = face_tensor.shape
    face_tensor = face_tensor.view(-1, C, W, H)

    # Instantiate facenet model
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    embeddings = resnet(face_tensor)

    return embeddings.view(N, F, -1)
# Alternate implementation of mtcnn:

# def detect_faces_mtcnn(video_tensor, display_images=False):
#
#     #Method for facial extraction using a prebuilt mtcnn model
#
#
#     total_predictions = []
#
#     for image in video_tensor:
#
#         image_np = image.numpy()
#
#         image_pil = Image.fromarray(image_np)
#         predictions, landmarks = detect_faces(image_pil)
#
#         print("Detected {} faces!".format(len(predictions)))
#         total_predictions.append(predictions)
#
#         if display_images:
#             for (x, y, x2, y2, s) in predictions:
#                 #print(x, y, w, h)
#                 cv2.rectangle(image_np, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)
#
#             cv2.imshow("Faces Found", image_np)
#             cv2.waitKey(0)
#
#     return total_predictions

def detect_faces_cascade(video_tensor, cascade_path, display_images=False):
    """
    Method for extracting facial extraction using an OpenCV implementation of
    the haar cascade algorithm

    Must provide a path to the cascade xml file.
    """
    faceCascade = cv2.CascadeClassifier(cascade_path)

    total_predictions = []

    for image in video_tensor:

        image_np = image.numpy()
        # Convert to grayscale
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        predictions = faceCascade.detectMultiScale(
            image_np,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        print("Detected {} faces!".format(len(predictions)))
        total_predictions.append(predictions)

        # For debugging purposes:
        if display_images:
            for (x, y, w, h) in predictions:
                cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow("Faces Found", image_np)
            cv2.waitKey(0)

    return total_predictions
