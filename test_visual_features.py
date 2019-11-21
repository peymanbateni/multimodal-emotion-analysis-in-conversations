import torch
import pickle
from dataset import MELDDataset
from models.visual_features import detect_faces_mtcnn, get_face_embeddings

"""
File for testing the visual features module
"""

cascade_path = 'haarcascade_frontalface_default.xml'


audio_embed_path = "../MELD.Raw/audio_embeddings_feature_selection_sentiment.pkl"

train_audio_emb, val_audio_emb, test_audio_emb = pickle.load(open(audio_embed_path, 'rb'))

dataset = MELDDataset("../MELD.Raw/dev_sent_emo.csv", "../MELD.Raw/dev_splits_complete/", val_audio_emb)

video = dataset[3][0][1][0]

print(dataset[3][0])

face_tensors = detect_faces_mtcnn(video, display_images=True)
embeddings = get_face_embeddings(face_tensors)

def mse(vector):
    zero_vec = vector[-1]
    mse = []
    for element in vector:
        mse.append(torch.sum(torch.abs(element - vector)).item())
    return mse

print(torch.sum(embeddings, axis=2))
print([mse(vector) for vector in embeddings])
# TODO: weirdly the facenet features don't return 0 for empty image. 
