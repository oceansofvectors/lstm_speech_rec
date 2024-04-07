# Speech Recognition Project

## Overview

This project is focused on building a speech recognition system using a deep learning approach. It utilizes a Long Short-Term Memory (LSTM) network to classify audio signals into predefined categories. The project is currently in progress, with the model converging, although the loss is still slightly higher than desired, at `0.85`. Several adjustments, techniques, and normalization methods are under consideration for improvement.

### Dataset

The dataset used for training the model is the Speech Commands Dataset, created by the TensorFlow and AIY teams. This dataset comprises 65,000 one-second-long utterances of 30 short words, recorded by thousands of different people. The contributions come from members of the public through the AIY website. This diverse collection of speech recordings is instrumental in training our model to recognize different speech patterns accurately.

The voice command words: 

`'sheila', 'seven', 'right', 'one', 'house', 'down', 'zero', 'go',
          'yes', 'wow', 'six', 'no', 'three', 'happy', 'bird', 'stop', 'marvin',
          'two', 'five', 'on', 'off', 'four', 'dog', 'up', 'tree', 'cat', 'bed',
          'nine', 'eight', 'left'`

To download the dataset, run the `download_data.py` script included in the project. This script automates the process of downloading and extracting the speech recognition dataset. For more information about the dataset and its creation, visit the official announcement on Google's Research Blog: [Launching the Speech Commands Dataset](https://blog.research.google/2017/08/launching-speech-commands-dataset.html).

### Model Architecture

The core of the project is the `LSTMSpeechClassifier`, a PyTorch model that uses an LSTM network. This model takes input audio signals and classifies them into different categories based on the features learned during training.

### Feature Extraction

MFCC (Mel-Frequency Cepstral Coefficients) features are extracted from the audio files to serve as input to the LSTM model. The `Mfcc` class provided in the project handles the loading of audio files, extraction of MFCC features, and padding of these features to a fixed size.

#### Audio Normalization

The first step involves preprocessing the raw audio data to ensure consistency and improve model performance. This process includes:

- **Trimming Silence:** Silence at the beginning and end of each audio clip is removed using `librosa.effects.trim`. This focuses the model's attention on the relevant parts of the audio signal.
- **Volume Normalization:** The volume of each audio clip is normalized to a consistent level. This is achieved by calculating the root mean square (RMS) of the audio signal, which provides a measure of its power. The audio signal is then scaled so that it achieves a desired RMS value. In our implementation, the desired RMS value is set to 0.1. This step ensures that the loudness of each audio file is consistent, reducing the model's sensitivity to variations in recording levels.

### Current Accuracy

As of the latest model evaluation, the accuracy of the speech recognition system stands at `0.85` on a test dataset of 6000 samples. 

![Screenshot 2024-04-07 at 12.33.16.png](Screenshot%202024-04-07%20at%2012.33.16.png)

### How to Run

1. Download the dataset by running the `download_data.py` script.
2. Calculate some data required for normalization by running `feature_extractor.py`. See above for normalization details.
2. Train the model using the provided training script, `train.py`. A checkpoint will be created.
3. Evaluate the model by running `test.py`

### Dependencies

- PyTorch
- LibROSA
- NumPy
- Requests
