import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import MFCCDataset
from lstm_model import LSTMSpeechClassifier

# Parameters
input_size = 13  # Number of MFCC features
hidden_size = 128  # Number of features in the hidden state
num_layers = 2  # Number of stacked LSTM layers
# Parameters update
num_classes = 30  # Update based on the number of labels you have

model = LSTMSpeechClassifier(input_size, hidden_size, num_layers, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Data loading
dataset = MFCCDataset('data/testing_list.txt', 'data')  # Update paths accordingly
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    for i, sample in enumerate(dataloader):
        mfccs = sample['mfccs']
        labels = sample['label']

        # Reshape mfccs to fit the model input
        mfccs = mfccs.float()

        # Forward pass
        outputs = model(mfccs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item()}')
