import torch
import torch.nn as nn
from dataset import MELDDataset, Utterance
import pickle

<<<<<<< HEAD
#audio_embed_path = "../MELD.Features.Models/features/audio_embeddings_feature_selection_emotion.pkl"
audio_embed_path = "../MELD.Features.Models/features/audio_embeddings_feature_selection_sentiment.pkl"

train_audio_emb, val_audio_emb, test_audio_emb = pickle.load(open(audio_embed_path, 'rb'))

dataset = MELDDataset("../MELD.Raw/dev_sent_emo.csv", "../MELD.Raw/dev_splits_complete/")
print(len(dataset))
utterance = dataset[0]

(transcripts, video, audio), (emotion_labels, sentiment_labels) = dataset[0:3]

# Transcripts
print(transcripts)
# Video
print(video[0].shape)
# Audio
print(audio[0][1].shape)

# labels
print(emotion_labels)
print(sentiment_labels)
#utterance = Utterance("", 1, 1, 1, "../MELD.Raw/dev_splits_complete/dia0_utt0.mp4")
#print(utterance.load_audio()[1].shape)
#print(utterance.load_video().shape)

dataset = MELDDataset("../MELD.Raw/dev_sent_emo.csv", "/MELD.Raw/dev_splits_complete/", val_audio_emb)
utterance = Utterance("", 1, 1, 1, "../MELD.Raw/dev_splits_complete/dia0_utt0.mp4", None)
print(utterance.load_video().shape)
#dataset_loader.load_image("../MELD.Raw/image.png")
