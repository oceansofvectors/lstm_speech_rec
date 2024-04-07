import pickle

import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for the progress bar

from data_loader import MFCCDataset

labels = ['sheila', 'seven', 'right', 'one', 'house', 'down', 'zero', 'go',
          'yes', 'wow', 'six', 'no', 'three', 'happy', 'bird', 'stop', 'marvin',
          'two', 'five', 'on', 'off', 'four', 'dog', 'up', 'tree', 'cat', 'bed',
          'nine', 'eight', 'left']


# Assuming your model file is saved with pickle and can be loaded like this
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


# Evaluation function
def evaluate_model(model, data_loader, device):
    model.eval()  # Set model to evaluation mode
    true_labels = []
    predictions = []

    with torch.no_grad():
        # Wrap data_loader in tqdm for the progress bar
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            inputs, batch_labels = batch['mfccs'].to(device), batch['label'].to(device)  # Changed variable name here
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            true_labels.extend(batch_labels.cpu().numpy())  # Use the updated variable name here
            predictions.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=labels)
    return accuracy, report

# Load your model (replace 'path_to_your_model.ckpt' with the actual model path)
model_path = 'model1712466047.2173529.ckpt'
model = load_model(model_path)

# Assuming the use of a GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Prepare your dataset
txt_file = 'data/validation_list.txt'
directory = 'data'  # Update this path
test_dataset = MFCCDataset(txt_file=txt_file, directory=directory)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Evaluate the model
accuracy, report = evaluate_model(model, test_loader, device)
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
