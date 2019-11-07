import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self, dummy_value = 99):
        super(DummyModel, self).__init__()
        self.dummy_param_sentiment = nn.Parameter(torch.FloatTensor(1,3))
        self.dummy_param_emotion = nn.Parameter(torch.FloatTensor(1,7))
        self.dummy_value = dummy_value

    def forward(self, input):
        (transcripts, video_data, audio_data) = input
        batch_size = len(audio_data[0])

        return self.dummy_param_emotion.repeat(batch_size, 1), self.dummy_param_sentiment.repeat(batch_size, 1)