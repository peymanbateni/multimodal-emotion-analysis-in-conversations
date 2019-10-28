import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import os
import cv2
from scipy.io import wavfile
import pickle

def video_to_tensor(video_file):
    """ Converts a mp4 file into a numpy array"""

    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        ret, buf[fc] = cap.read()
        fc += 1

    cap.release()
    return torch.tensor(buf)


class Utterance(object):
    """
    Class for representing a single utterance in all 3 modalities
    """
    def __init__(self, transcript, speaker, emotion, sentiment, file_path, utt_audio):
        self.transcript = transcript
        self.speaker = speaker
        self.emotion = emotion
        self.sentiment = sentiment
        self.file_path = file_path
        self.utt_audio = utt_audio

    def get_transcript(self):
        return self.transcript

    def load_video(self):
        return video_to_tensor(self.file_path)


class MELDDataset(Dataset):

    def __init__(self, csv_file, root_dir, audio_embs):
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
            self.emotions_to_label[emotion] = i

        self.sentiments_to_label = {}
        for i, sentiment in enumerate(self.list_of_sentiments):
            self.sentiments_to_label[sentiment] = i

        print("Speaker mapping: {}".format(self.speakers_to_label))
        print("Emotion mapping: {}".format(self.emotions_to_label))
        print("Sentiment mapping: {}".format(self.sentiments_to_label))
        #print([self.speakers_to_label[key] for key in self.csv_records.loc[:, "Speaker"].values.tolist()])

        #data_poin
        #for file_name in os.listdir(self.root_dir):
        #    if file_name.endswith(".mp4"):
        #self
        print(self.csv_records.iloc[0:10,:])

        self.data = []
        for record in self.csv_records.loc[:].values:
            print(record)
            id, transcript, speaker, emotion, sentiment, d_id, u_id, _, _, _, _ = record

            # TODO: Still some issues with parsing the transcript, specifically wrt special symbols
            file_path = "dia{}_utt{}.mp4".format(d_id, u_id)
            file_path = os.path.join(self.root_dir, file_path)
            utt_audio_embed_id = str(d_id) + "_" + str(u_id)
            utt_audio_embed = audio_embs[utt_audio_embed_id]
            utterance = Utterance(
                transcript,
                self.speakers_to_label[speaker],
                self.emotions_to_label[emotion],
                self.sentiments_to_label[sentiment],
                file_path,
                utt_audio_embed
            )
            self.data.append(utterance)

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
