import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import os
import torchaudio

class MELDDataset(Dataset):

    def __init__(self, csv_file, root_dir):
        self.csv_records = pd.read_csv(csv_file)
        self.root_dir = root_dir
        
        self.list_of_speakers = list(set(self.csv_records.loc[:, "Speaker"].values.tolist()))
        self.list_of_emotions = list(set(self.csv_records.loc[:, "Emotion"].values.tolist()))
        self.list_of_sentiments = list(set(self.csv_records.loc[:, "Sentiment"].values.tolist()))

        self.speakers_to_label = {}
        for i, speaker in enumerate(self.list_of_speakers):
            self.speakers_to_label[speaker] = i

        self.emotions_to_label = {}
        for i, emotion in enumerate(self.list_of_emotions):
            self.emotions_to_label[speaker] = i

        self.sentiments_to_label = {}
        for i, sentiment in enumerate(self.list_of_sentiments):
            self.sentiments_to_label[sentiment] = i

        print(self.speakers_to_label[self.csv_records.loc[:, "Speaker"].values.tolist()])

        data_points = []
        #for file_name in os.listdir(self.root_dir):
        #    if file_name.endswith(".mp4"):
        self 
        print()
        print(self.csv_records.iloc[0:10,:])

    def __len__(self):
        return len(self.csv_records)

    def __getitem__(self, idx):

        return None

    def load_sample_audio(audio_file_address):
        return torchaudio.load(audio_file_address)

    def load_transcript(csv_file_address):
        return None

    def load_image(image_file_address):
        image = Image.open(image_file_address)
        image = ToTensor()(image).unsqueeze(0) # unsqueeze to add artificial first dimension
        image = Variable(image)
        return image

    def load_image_set(image_dir_address):
        
        if not image_dir_address.endswith("/"):
            image_dir_address += "/"
        
        image_list = []
        for file_name in os.listdir(image_dir_address):
            image_tensor = load_image(image_dir_address + file_name)
            image_list.append(image_tensor)

        image_tensors = torch.cat(image_list, dim=0)
        return image_tensors