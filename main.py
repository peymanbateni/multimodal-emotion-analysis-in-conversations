import torch
import torch.nn as nn
from dataset import MELDDataset, Utterance
import pickle

#audio_embed_path = "../MELD.Features.Models/features/audio_embeddings_feature_selection_emotion.pkl"
audio_embed_path = "../MELD.Features.Models/features/audio_embeddings_feature_selection_sentiment.pkl"

train_audio_emb, val_audio_emb, test_audio_emb = pickle.load(open(audio_embed_path, 'rb'))

dataset = MELDDataset("../MELD.Raw/dev_sent_emo.csv", "/MELD.Raw/dev_splits_complete/", val_audio_emb)
utterance = Utterance("", 1, 1, 1, "../MELD.Raw/dev_splits_complete/dia0_utt0.mp4", None)
print(utterance.load_video().shape)
#dataset_loader.load_image("../MELD.Raw/image.png")
