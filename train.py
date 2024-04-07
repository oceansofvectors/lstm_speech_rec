import mlflow
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import MFCCDataset
from lstm_model import LSTMSpeechClassifier

# MLflow setup
experiment_name = "LSTM Speech Classification"
mlflow.set_experiment(experiment_name)

# Parameters
input_size = 128  # Number of MFCC features
hidden_size = 128  # Number of features in the hidden state
num_layers = 2  # Number of stacked LSTM layers
num_classes = 30  # Update based on the number of labels you have
learning_rate = 0.01
labels = ['sheila', 'seven', 'right', 'one', 'house', 'down', 'zero', 'go',
          'yes', 'wow', 'six', 'no', 'three', 'happy', 'bird', 'stop', 'marvin',
          'two', 'five', 'on', 'off', 'four', 'dog', 'up', 'tree', 'cat', 'bed',
          'nine', 'eight', 'left']
model = LSTMSpeechClassifier(input_size, hidden_size, num_layers, num_classes, labels)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Data loading
dataset = MFCCDataset('data/testing_list.txt', 'data')  # Update paths accordingly
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 10

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_classes": num_classes,
        "learning_rate": learning_rate,
        "batch_size": 32,
        "num_epochs": num_epochs,
    })

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
                # Log metrics
                mlflow.log_metrics({
                    "loss": loss.item()
                }, step=epoch * len(dataloader) + i)

    # Log model
    mlflow.pytorch.log_model(model, "model")
    model.save_model()
