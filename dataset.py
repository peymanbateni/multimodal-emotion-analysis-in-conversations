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
    """ Converts a mp4 file into a pytorch tensor"""

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
        """
        Returns a string representation of the transcript
        """
        # TODO: Still some issues with parsing the transcript, specifically wrt special symbols
        return self.transcript

    def get_speaker(self):
        """
        Returns the integer-mapped speaker id
        """
        return self.speaker

    def get_label(self):
        """
        Returns a tuple of the integer-mapped label as (emotion, sentiment)
        """

        return (self.emotion, self.sentiment)

    def load_video(self):
        """
        Loads the video into memory and converts the frames into a pyTorch tensor
        """
        return video_to_tensor(self.file_path)

    def load_audio(self):
        """
        Returns the Audio embeddings.
        """
        return self.utt_audio


class MELDDataset(Dataset):
    """
    Class representing MELD dataset. Initialization is against a csv file and
    root directory.

    Accessing the dataset via an accessor will load the appropriate video and
    audio representations in memory and will be returned in the form:

        ([transcript], [video], [audio]) ([emotion_label], [sentiment_label])

    Attributes:

    csv_recods: pandas representation of csv file
    root_dir: pah to root directory of data

    speaker_mapping: dictionary mapping speaker name to speaker id
    emotion_mapping: dictionary mapping emotion to emotion id
    sentiment_mapping: dictionary mapping sentiment to sentiment id

    Methods:

    load_sample_audio(index): loads the audio tuple retrieved from index
    load_sample_video(index): loads the video tensor retrieved from index
    """

    def __init__(self, csv_file, root_dir, audio_embs):
        self.csv_records = pd.read_csv(csv_file)
        self.root_dir = os.path.abspath(root_dir)

        speaker_set = set(self.csv_records.loc[:, "Speaker"].values.tolist())
        emotion_set = set(self.csv_records.loc[:, "Emotion"].values.tolist())
        sentiment_set = set(self.csv_records.loc[:, "Sentiment"].values.tolist())

        # TODO: we have combination mappings right now (eg. "Monica and Rachael")
        self.speaker_mapping = {speaker: id for id, speaker in enumerate(speaker_set)}
        self.emotion_mapping = {emotion: id for id, emotion in enumerate(emotion_set)}
        self.sentiment_mapping = {sentiment: id for id, sentiment in enumerate(sentiment_set)}

        print("Speaker mapping: {}".format(self.speaker_mapping))
        print("Emotion mapping: {}".format(self.emotion_mapping))
        print("Sentiment mapping: {}".format(self.sentiment_mapping))
        #print([self.speakers_to_label[key] for key in self.csv_records.loc[:, "Speaker"].values.tolist()])

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
                self.speaker_mapping[speaker],
                self.emotion_mapping[emotion],
                self.sentiment_mapping[sentiment],
                file_path,
                utt_audio_embed
            )
            self.data.append(utterance)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        utterances = self.data[idx]
        if not isinstance(utterances, list):
            # only single utterance
            input = ([utterances.get_transcript()], [utterances.load_video()], [utterances.load_audio()])
            label = utterances.get_label()
            return input, label
        else:
            transcripts = [utterance.get_transcript() for utterance in utterances]
            video = [utterance.load_video() for utterance in utterances]
            audio = [utterance.load_audio() for utterance in utterances]
            emotion_labels = [utterance.get_label()[0] for utterance in utterances]
            sentiment_labels = [utterance.get_label()[1] for utterance in utterances]

        input = (transcripts, video, audio)
        labels = (emotion_labels, sentiment_labels)
        return input, labels

    def load_sample_transcript(self, idx):
        return self.data[idx].get_transcript()

    def load_sample_audio(self, idx):
        return self.data[idx].load_audio()

    def load_sample_video(self, idx):
        return self.data[idx].load_video()

    """
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
    """
