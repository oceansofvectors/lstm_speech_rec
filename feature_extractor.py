from os import listdir

import librosa
import numpy as np


class Mfcc:

    def __init__(self, stats_file=None):
        self.mean = None
        self.std = None
        self.stats_file = stats_file  # File path for saving/loading mean and std
        self.labels = [
            'sheila', 'seven', 'right', 'one', 'house', 'down', 'zero', 'go',
            'yes', 'wow', 'six', 'no', 'three', 'happy', 'bird', 'stop', 'marvin',
            'two', 'five', 'on', 'off', 'four', 'dog', 'up', 'tree', 'cat', 'bed',
            'nine', 'eight', 'left'
        ]
        if stats_file:
            self.load_stats()

    def load_audio(self, filepath):
        audio, sr = librosa.load(filepath, sr=32000)
        # Trim silence
        audio, _ = librosa.effects.trim(audio)
        # Normalize volume
        rms = np.sqrt(np.mean(audio ** 2))
        desired_rms = 0.1
        audio = audio * (desired_rms / rms)
        return audio, sr

    def get_mfccs(self, filepath):
        audio, sr = self.load_audio(filepath)
        # Check and handle NaN or infinite values
        if not np.all(np.isfinite(audio)):
            print(f"Non-finite values found in {filepath}, replacing with zeros.")
            audio = np.nan_to_num(audio)  # Replace NaN and infinite values with zero
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=128)
        mfccs_padded = np.pad(mfccs, ((0, 0), (0, max(0, 65 - mfccs.shape[1]))),
                              mode='constant', constant_values=0)
        # Normalize MFCCs if mean and std are loaded
        mfccs_normalized = self.normalize_mfccs(mfccs_padded.T)
        return mfccs_normalized

    def normalize_mfccs(self, mfccs):
        if self.mean is not None and self.std is not None:
            return (mfccs - self.mean) / self.std
        return mfccs

    def compute_dataset_statistics(self, directory):
        all_mfccs = []
        for label in self.labels:
            print(f"Calculating for {label}")
            files = listdir(f'{directory}/{label}')
            for file in files:
                mfccs = self.get_mfccs(f"{directory}/{label}/{file}")
                all_mfccs.append(mfccs)

        all_mfccs = np.concatenate(all_mfccs, axis=0)
        self.mean = np.mean(all_mfccs, axis=0)
        self.std = np.std(all_mfccs, axis=0)

        # Save mean and std to a file
        self.save_stats()

        return self.mean, self.std

    def save_stats(self):
        if self.stats_file:
            np.savez(self.stats_file, mean=self.mean, std=self.std)

    def load_stats(self):
        try:
            stats = np.load(self.stats_file)
            self.mean = stats['mean']
            self.std = stats['std']
        except IOError:
            print("Stats file not found. Will need to compute from dataset.")


if __name__ == "__main__":
    stats_file = 'mfcc_stats.npz'
    mfcc = Mfcc(stats_file=stats_file)
    directory = 'data'  # Adjust based on your data directory
    # Only compute and save if mean and std are not already loaded
    if mfcc.mean is None or mfcc.std is None:
        print("Computing statistics...")
        mean, std = mfcc.compute_dataset_statistics(directory)
        print(f"Mean: {mean}, Std: {std}")
    else:
        print("Loaded statistics from file.")
