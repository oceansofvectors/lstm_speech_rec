import pickle
import time

import torch
from torch import nn as nn


class LSTMSpeechClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, labels):
        super(LSTMSpeechClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.labels = labels  # Add this line to store class labels

    def save_model(self):
        with open(f"model{time.time()}.ckpt", "wb") as file:
            pickle.dump(self, file)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
