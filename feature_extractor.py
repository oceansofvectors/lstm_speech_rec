import matplotlib.pyplot as plt
import torch
import torchaudio


class Spectrogram:
    def __init__(self, n_fft=400, hop_length=160, win_length=400):
        """
        Initialize the Spectrogram object with necessary parameters for the STFT transformation.

        :param n_fft: Number of FFT components.
        :param hop_length: Number of audio samples between adjacent STFT columns.
        :param win_length: Window size.
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)

    def load_audio(self, filepath):
        """
        Loads an audio file.

        :param filepath: Path to the audio file.
        :return: Waveform and sample rate.
        """
        waveform, sample_rate = torchaudio.load(filepath)
        return waveform.squeeze(0), sample_rate  # Assuming single-channel audio

    def to_tensor(self, waveform):
        """
        Converts an audio waveform to a spectrogram tensor.

        :param waveform: Audio waveform tensor.
        :return: Spectrogram tensor in dB scale.
        """
        spec = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.win_length, window=self.window, center=True,
                          normalized=False, onesided=True, return_complex=True)
        spec_mag = torch.abs(spec)
        spec_mag_db = torch.log10(spec_mag + 1e-6) * 20  # Convert to dB
        return spec_mag_db

    def plot_spectrogram(self, spec_mag_db):
        """
        Plots the spectrogram.

        :param spec_mag_db: Spectrogram tensor in dB scale.
        """
        plt.figure(figsize=(20, 4))
        plt.imshow(spec_mag_db.numpy(), cmap='hot', aspect='auto', origin='lower')
        plt.title('Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
