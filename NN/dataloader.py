import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
import cupy as cp


def get_rms(audio_signal, frame_length=1024, hop_length=64):
  num_frames = int(cp.ceil(len(audio_signal) / hop_length))
  rms_values = cp.zeros(num_frames)
  for frame_idx in range(num_frames):
    start_index = frame_idx * hop_length
    end_index = min(start_index + frame_length, len(audio_signal))
    frame_data = audio_signal[start_index:end_index]

    window = cp.hanning(len(frame_data))
    frame_data = frame_data * window

    squared_signal = cp.power(frame_data, 2)
    mean_squared = cp.mean(squared_signal)
    rms_value = cp.sqrt(mean_squared)

    rms_values[frame_idx] = rms_value

  return rms_values

def get_zero_crossings(audio_signal, frame_length=1024, hop_length=64):
  num_frames = int(np.ceil(len(audio_signal) / hop_length))
  zero_crossings = np.zeros(num_frames)

  for frame_idx in range(num_frames):
    start_index = frame_idx * hop_length
    end_index = min(start_index + frame_length, len(audio_signal))
    frame_data = audio_signal[start_index:end_index]
    sign_diffs = cp.diff(cp.sign(frame_data))
    zero_crossings[frame_idx] = cp.sum(cp.abs(sign_diffs))

  return zero_crossings

def get_spectral_flux(audio_signal, frame_length=1024, hop_length=64):
  if frame_length % hop_length != 0:
    raise ValueError("Frame length must be a multiple of hop length.")
  num_frames = int(np.ceil(len(audio_signal) / hop_length))
  spectral_flux = np.zeros(num_frames)
  prev_abs_spectrum = np.zeros(frame_length // 2 + 1)

  for frame_idx in range(num_frames):
    start_index = frame_idx * hop_length
    end_index = min(start_index + frame_length, len(audio_signal))
    frame_data = audio_signal[start_index:end_index]

    abs_spectrum = np.abs(np.fft.fft(frame_data))[:frame_length // 2 + 1]
    spectral_flux[frame_idx] = np.sum(np.abs(abs_spectrum - prev_abs_spectrum))
    prev_abs_spectrum = abs_spectrum.copy()

  return spectral_flux

def get_spectral_flatness(audio_signal, frame_length=1024, hop_length=64):
  if frame_length % hop_length != 0:
    raise ValueError("Frame length must be a multiple of hop length.")
  num_frames = int(cp.ceil(len(audio_signal) / hop_length))
  spectral_flatness = cp.zeros(num_frames)

  for frame_idx in range(num_frames):
    start_index = frame_idx * hop_length
    end_index = min(start_index + frame_length, len(audio_signal))
    frame_data = audio_signal[start_index:end_index]

    abs_spectrum = cp.abs(np.fft.fft(frame_data))[:frame_length // 2 + 1]
    spectral_flatness[frame_idx] = cp.exp(np.mean(cp.log(abs_spectrum + cp.spacing(1)))) / cp.mean(abs_spectrum)

  return spectral_flatness

class BenjoDataset(Dataset):
    def __init__(self, data_dir, num_mfccs=13, features='mfcc-2d', device='cuda'):  # Adjust max_length as needed
        self.data_dir = data_dir
        self.features = features
        self.device = device
        self.num_mfccs = num_mfccs
        self.win_length = 1024
        self.hop_size = 64
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".wav")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]

        if self.features == 'params':
            params_array, _ = self.get_benjo_params(index)
            return torch.tensor(params_array, dtype=torch.float32).to(self.device) / 126

        waveform, sample_rate = torchaudio.load(path, normalize=False, format='wav')
        waveform = waveform[0, :].to(self.device)

        if 'bag-of-frames' == self.features:
            MFCC = torchaudio.transforms.MFCC(sample_rate=sample_rate,
                                              n_mfcc=self.num_mfccs,
                                              melkwargs={"n_fft": 1024,
                                                         "win_length": self.win_length,
                                                         "hop_size": self.hop_size,
                                                         "pad": 0,
                                                         "n_mels": 101, "center": False}).to(self.device)
            mfcc = MFCC(waveform) / self.num_mfccs
            
            spectral_centroid = torchaudio.spectral_centroid(sample_rate=sample_rate,
                                                             n_fft=self.win_length,
                                                             win_length=self.win_length,
                                                             hop_size=self.hop_size,
                                                             pad=0)
            cupy_waveform = cp.asarray(waveform)
            
            zero_crossings = torch.as_tensor(get_zero_crossings(cupy_waveform, self.win_length, self.hop_size))
            rms = torch.as_tensor(get_rms(cupy_waveform, self.win_length, self.hop_size))
            spectral_flux = torch.as_tensor(get_spectral_flux(cupy_waveform, self.win_length, self.hop_size))
            spectral_flatness = torch.as_tensor(get_spectral_flatness(cupy_waveform, self.win_length, self.hop_size))

            matrix = np.vstack((mfcc, spectral_centroid, zero_crossings, rms, spectral_flux, spectral_flatness))
            """matrix -= self.means[:, np.newaxis]
            matrix /= self.stds[:, np.newaxis]"""
            summary_matrix = np.zeros((matrix.shape[0], 2))
            for i in range(matrix.shape[0]):
                this_row = matrix[i, :]
                mean = cp.mean(this_row)
                std = cp.std(this_row)
                # skewness = stats.skew(this_row)
                # kurtosis = stats.kurtosis(this_row)
                summary_matrix[i, :] = cp.array([mean, std])
            return summary_matrix

        if 'mfcc' == self.features:
            MFCC = torchaudio.transforms.MFCC(sample_rate=sample_rate,
                                              n_mfcc=self.num_mfccs,
                                              melkwargs={"n_fft": 400,
                                                         "n_mels": 101, "center": False}).to(self.device)
            features = MFCC(waveform)
            features = torch.clip(features, min=-25, max=25)
            shape = features.shape
            features += 25
            features /= 50
            return features[:, :440]

    def get_benjo_params(self, index):
        path = self.files[index]
        params_string = path.removeprefix(self.data_dir + '/').removesuffix('.wav')
        params_list = params_string.split('-')
        params_array = np.array(list(map(int, params_list)))
        return params_array, params_string

