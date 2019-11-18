import torch
import torch.nn as nn
import torch.optim as optim
from dataset import MELDDataset, Utterance
import pickle
from dummy_model import DummyModel
from torch.utils.data import DataLoader
from models.config import Config
from models.dialogue_gcn import DialogueGCN

#audio_embed_path = "../MELD.Features.Models/features/audio_embeddings_feature_selection_emotion.pkl"
audio_embed_path = "../MELD.Raw/audio_embeddings_feature_selection_sentiment.pkl"

train_audio_emb, val_audio_emb, test_audio_emb = pickle.load(open(audio_embed_path, 'rb'))
#print(train_audio_emb.keys())
#x = int("hey")

#dataset = MELDDataset("../MELD.Raw/dev_sent_emo.csv", "../MELD.Raw/dev_splits_complete/", val_audio_emb)
#print(len(dataset))
#utterance = dataset[0]

#(transcripts, video, audio), (emotion_labels, sentiment_labels) = dataset[0:3]

# Transcripts
#print(transcripts)
# Video
#print(video[0].shape)
# Audio
#print(audio[0][1].shape)

# labels
#print(emotion_labels)
#print(sentiment_labels)
#utterance = Utterance("", 1, 1, 1, "../MELD.Raw/dev_splits_complete/dia0_utt0.mp4")
#print(utterance.load_audio()[1].shape)
#print(utterance.load_video().shape)

val_dataset = MELDDataset("../MELD.Raw/dev_sent_emo.csv", "/MELD.Raw/dev_splits_complete/", val_audio_emb)
train_dataset = MELDDataset("../MELD.Raw/train_sent_emo.csv", "/MELD.Raw/train_splits/", train_audio_emb)
test_dataset = MELDDataset("../MELD.Raw/test_sent_emo.csv", "/MELD.Raw/output_repeated_splits_test", test_audio_emb)
#utterance = Utterance("", 1, 1, 1, "../MELD.Raw/dev_splits_complete/dia0_utt0.mp4", None)
#print(utterance.load_video().shape)
#dataset_loader.load_image("../MELD.Raw/image.png")


def train_and_validate(model_name, model, optimiser, loss_emotion, loss_sentiment, train_data_loader, val_data_loader, epochs=5):

    for epoch in range(epochs):

        model = model.train()
        loss_acc = 0
        total_epoch_loss = 0
        for i, (batch_input, batch_labels) in enumerate(train_data_loader):
            batch_loss = train_step(model, batch_input, batch_labels, loss_emotion, loss_sentiment, optimiser)
            loss_acc += batch_loss
            total_epoch_loss += batch_loss
            if (i % 50):
                print("Epoch[" + str(epoch) + "/" + str(epochs) +"] - batch " + str(i) + " Error: " + str(loss_acc))
                loss_acc = 0

        model = model.eval()
        val_count = 0
        emotion_correct_count = 0
        sentiment_correct_count = 0
        for i, (val_batch_input, val_batch_labels) in enumerate(val_data_loader):
            batch_emotion_correct, batch_sentiment_correct, batch_val_count = validate_step(model, val_batch_input, val_batch_labels)
            val_count += batch_val_count
            emotion_correct_count += batch_emotion_correct.item()
            sentiment_correct_count += batch_sentiment_correct.item()

        print("Validation Accuracy (Emotion): ", str(emotion_correct_count / val_count))
        print("Validation Accuracy (Sentiment): ", str(sentiment_correct_count / val_count))

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
            'loss': total_epoch_loss
            },  "model_saves/" + model_name + "_epoch" + str(epoch) +".pt")

def test_model(model_name, model, test_loader):
    print("Testing " + model_name)
    model = model.eval()
    test_count = 0
    emotion_correct_count = 0
    sentiment_correct_count = 0
    for i, (test_batch_input, test_batch_labels) in enumerate(test_loader):
        batch_emotion_correct, batch_sentiment_correct, batch_test_count = test_step(model, test_batch_input, test_batch_labels)
        test_count += batch_test_count
        emotion_correct_count += batch_emotion_correct.item()
        sentiment_correct_count += batch_sentiment_correct.item()

    print("Test Accuracy (Emotion): ", str(emotion_correct_count / test_count))
    print("Test Accuracy (Sentiment): ", str(sentiment_correct_count / test_count))

    return (emotion_correct_count / test_count), (sentiment_correct_count / test_count)

def train_step(model, input, target, loss_emotion, loss_sentiment, optimiser):
    """Trains model for one batch of data."""
    optimiser.zero_grad()
    (batch_output_emotion, batch_output_sentiment) = model(input)
    target = torch.LongTensor(target).to("cuda")
    batch_loss_emotion = loss_emotion(batch_output_emotion, target[0])
    batch_loss_sentiment = loss_sentiment(batch_output_sentiment, target[1])
    total_loss = batch_loss_emotion + batch_loss_sentiment
    total_loss.backward()
    optimiser.step()
    return total_loss.item()

def validate_step(model, input, target):
    target = torch.LongTensor(target).to("cuda")
    (output_logits_emotion, output_logits_sentiment) = model(input)
    output_labels_emotion = torch.argmax(output_logits_emotion, dim=1)
    output_labels_sentiment = torch.argmax(output_logits_sentiment, dim=1)
    emotion_accuracy_acc = torch.eq(output_labels_emotion, target[0]).sum()
    sentiment_accuracy_acc = torch.eq(output_labels_emotion, target[1]).sum()
    return emotion_accuracy_acc, sentiment_accuracy_acc, target[0].size(0)

def test_step(model, input, target):
    (output_logits_emotion, output_logits_sentiment) = model(input)
    output_labels_emotion = torch.argmax(output_logits_emotion, dim=1)
    output_labels_sentiment = torch.argmax(output_logits_sentiment, dim=1)
    emotion_accuracy_acc = torch.eq(output_labels_emotion, target[0]).sum()
    sentiment_accuracy_acc = torch.eq(output_labels_emotion, target[1]).sum()
    return emotion_accuracy_acc, sentiment_accuracy_acc, target[0].size(0)

dumb_model = DummyModel()
config = Config()
emotion_criterion = nn.CrossEntropyLoss()
sentiment_criterion = nn.CrossEntropyLoss()
model_name = "Dumb_Model"
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)

train_and_validate(model_name, dumb_model, optimisation_unit, emotion_criterion, sentiment_criterion, train_loader, val_loader)
test_model(model_name, dumb_model, test_loader)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

if config.use_dummy:
    model = DummyModel()
else:
    model = DialogueGCN(config)
    model.to("cuda")

optimisation_unit = optim.Adam(model.parameters(), lr=0.001)
train_and_validate(model_name, model, optimisation_unit, emotion_criterion, sentiment_criterion, train_loader, val_loader)
