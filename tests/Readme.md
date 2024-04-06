# Voice Command Classification Project

## Overview

This project aims to classify audio files of voice commands into their respective classes using a Long Short-Term Memory (LSTM) neural network. The foundation of this project is the Speech Commands Dataset, provided by Google, which consists of tens of thousands of labeled audio clips of spoken words ([Google Research Blog](https://blog.research.google/2017/08/launching-speech-commands-dataset.html)). The primary goal is to process these audio clips, extract meaningful features, and train an LSTM network for accurate voice command recognition.

## Current Status

As of now, the project is in its development phase, with significant progress made towards setting up the data preprocessing pipeline and preliminary LSTM model implementation. However, we have encountered several challenges related to audio file normalization, spectrogram conversion, and input standardization for the LSTM network.

### Challenges Faced

- **Normalization Problems:** The audio files vary significantly in their amplitude, which affects the consistency of our data inputs. Normalization is required to ensure that the LSTM network receives input data within a standard range.
  
- **Varying Frequency Bins in Spectrograms:** The conversion of audio files to spectrograms has resulted in varying numbers of frequency bins, complicating the uniformity of input data shapes required for training the LSTM model.
  
- **Different Time Lengths in Audio Files:** Audio files in the dataset have varying time lengths, necessitating padding to achieve a consistent input shape for the LSTM model.

## To-Do List

To address the challenges outlined above and advance towards completing the project, the following tasks have been identified:

1. **Normalize Audio Files:**
   - Research and implement an effective method for normalizing the amplitude of audio files to a standard range without distorting the audio quality.
   
2. **Standardize Spectrogram Conversion:**
   - Investigate and apply techniques to ensure a consistent number of frequency bins across all spectrograms, possibly involving adjustments in the FFT (Fast Fourier Transform) parameters or the spectrogram generation process.
   
3. **Implement Padding for Audio Files:**
   - Develop a padding strategy that standardizes the time length of audio inputs without introducing bias or significantly altering the data. Explore zero-padding and other techniques to maintain the integrity of the original audio signals.
   
4. **Convert Audio to Standardized Tensors:**
   - Finalize a preprocessing pipeline that converts audio files into standardized tensors, incorporating the normalization, spectrogram standardization, and padding solutions. This step is crucial for preparing the dataset for LSTM training.
   
5. **Train LSTM Model (Ongoing):**
   - While the LSTM model implementation is not currently a concern, continue refining the model architecture and training process as the preprocessing pipeline is finalized. This includes selecting appropriate hyperparameters, loss functions, and optimization algorithms.

## Contributions

Contributions to this project are welcome, especially in tackling the to-do list items and enhancing the LSTM model's performance. Please feel free to fork the project, make your changes, and submit a pull request with a detailed description of your contributions and improvements.

## Contact

For questions, suggestions, or collaborations, please open an issue in the project's repository or contact the project maintainers directly via [insert contact method].
