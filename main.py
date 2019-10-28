import torch
import torch.nn as nn
from dataset import MELDDataset, Utterance

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

#dataset_loader.load_image("../MELD.Raw/image.png")
