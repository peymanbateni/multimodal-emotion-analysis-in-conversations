import torch
import torch.nn as nn
from dataset import MELDDataset, Utterance

dataset_loader = MELDDataset("../MELD.Raw/dev_sent_emo.csv", "/MELD.Raw/dev_splits_complete/")
utterance = Utterance("", 1, 1, 1, "../MELD.Raw/dev_splits_complete/dia0_utt0.mp4")
print(utterance.load_video().shape)
#dataset_loader.load_image("../MELD.Raw/image.png")
