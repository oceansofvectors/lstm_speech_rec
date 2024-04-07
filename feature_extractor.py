from os import listdir

import librosa
import numpy as np


class Mfcc:

    def __init__(self):
        pass

    def load_audio(self, filepath):
        """
        Loads an audio file.

        :param filepath: Path to the audio file.
        :return: Waveform and sample rate.
        """
        audio, sr = librosa.load(filepath, sr=16000)  # Explicitly setting the sample rate
        return audio, sr

    def get_mfccs(self, filepath):
        """
        Extracts MFCCs from an audio file and pads them to a fixed size.

        :param filepath: Path to the audio file.
        :return: Padded MFCCs.
        """
        # Extract MFCCs
        audio, sr = self.load_audio(filepath)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # Corrected function call
        # Padding
        mfccs_padded = np.pad(mfccs, ((0, 0), (0, 50 - mfccs.shape[1])), mode='constant', constant_values=0)
        return mfccs_padded.T


if __name__ == "__main__":
    mfcc = Mfcc()
    files = listdir('data/bed')[0:25]
    for file in files:
        output = mfcc.get_mfccs(f"data/bed/{file}")
        print(output.shape)

    files = listdir('data/bird')[0:25]
    for file in files:
        output = mfcc.get_mfccs(f"data/bird/{file}")
        print(output.shape)
