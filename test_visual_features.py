import torch
import pickle
from dataset import MELDDataset
from models.visual_features import detect_faces_cascade, detect_faces_mtcnn

"""
File for testing the visual features module
"""

cascade_path = 'haarcascade_frontalface_default.xml'


audio_embed_path = "../MELD.Raw/audio_embeddings_feature_selection_sentiment.pkl"

train_audio_emb, val_audio_emb, test_audio_emb = pickle.load(open(audio_embed_path, 'rb'))

dataset = MELDDataset("../MELD.Raw/dev_sent_emo.csv", "../MELD.Raw/dev_splits_complete/", val_audio_emb)

video = dataset[2][0][1][0]

detect_faces_cascade(video, cascade_path, display_images=True)
