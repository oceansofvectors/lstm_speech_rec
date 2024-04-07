# Speech Recognition Project

## Overview
This project is focused on building a speech recognition system using a deep learning approach. It utilizes a Long Short-Term Memory (LSTM) network to classify audio signals into predefined categories. The project is currently in progress, with the model converging, although the loss is still higher than desired. Several adjustments, techniques, and normalization methods are under consideration for improvement.

### Dataset
The dataset used for training the model is a collection of speech recordings. To download the dataset, run the `download_data.py` script included in the project. This script automates the process of downloading and extracting the speech recognition dataset.

### Model Architecture
The core of the project is the `LSTMSpeechClassifier`, a PyTorch model that uses an LSTM network. This model takes input audio signals and classifies them into different categories based on the features learned during training.

### Feature Extraction
MFCC (Mel-Frequency Cepstral Coefficients) features are extracted from the audio files to serve as input to the LSTM model. The `Mfcc` class provided in the project handles the loading of audio files, extraction of MFCC features, and padding of these features to a fixed size.

### Current Progress
- The model is implemented and capable of training on the speech dataset.
- The training process is ongoing, with the model showing signs of convergence. However, the current loss value indicates that further optimization is needed.

### Next Steps
- Experiment with various adjustments to the model architecture and training procedure to reduce loss.
- Implement and evaluate different normalization techniques to improve model performance.
- Explore additional feature extraction methods that may enhance the model's ability to recognize speech patterns.

### How to Run
1. Download the dataset by running the `download_data.py` script.
2. Train the model using the provided training script.
3. Evaluate the model's performance and iterate on the model design based on the results.

### Dependencies
- PyTorch
- LibROSA
- NumPy
- Requests
