import pytest
import torch

from feature_extractor import Spectrogram  # Adjust the import path as needed


# Fixture to load audio for tests
@pytest.fixture
def audio_data():
    spectrogram = Spectrogram()
    waveform, sample_rate = spectrogram.load_audio('../data/bed/0a7c2a8d_nohash_0.wav')
    return waveform, sample_rate


def test_load_audio(audio_data):
    waveform, sample_rate = audio_data
    assert waveform.ndim == 1  # Ensure waveform is 1D
    assert sample_rate > 0  # Sample rate should be a positive integer


def test_to_tensor_dimensions(audio_data):
    waveform, _ = audio_data
    spectrogram = Spectrogram()
    spec_mag_db = spectrogram.to_tensor(waveform)
    assert spec_mag_db.ndim == 2  # Spectrogram should be 2D
    assert spec_mag_db.size(0) > 0 and spec_mag_db.size(1) > 0  # Check for non-empty dimensions


def test_to_tensor_value_range(audio_data):
    waveform, _ = audio_data
    spectrogram = Spectrogram()
    spec_mag_db = spectrogram.to_tensor(waveform)
    assert torch.all(spec_mag_db >= -120) and torch.all(spec_mag_db <= 25), "Spectrogram values should be in dB scale"
