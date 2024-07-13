import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
import cupy as cp
import pickle
import zipfile


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
  num_frames = int(cp.ceil(len(audio_signal) / hop_length))
  spectral_flux = cp.zeros(num_frames)
  prev_abs_spectrum = cp.zeros(frame_length // 2 + 1)

  for frame_idx in range(num_frames):
    start_index = frame_idx * hop_length
    end_index = min(start_index + frame_length, len(audio_signal))
    frame_data = audio_signal[start_index:end_index]

    abs_spectrum = cp.abs(cp.fft.fft(frame_data))[:frame_length // 2 + 1]
    length = min(len(abs_spectrum), len(prev_abs_spectrum))
    spectral_flux[frame_idx] = cp.sum(cp.abs(abs_spectrum[:length] - prev_abs_spectrum[:length]))
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
    spectral_flatness[frame_idx] = cp.exp(cp.mean(cp.log(abs_spectrum))) / cp.mean(abs_spectrum)

  return spectral_flatness

class BenjoDataset(Dataset):
    def __init__(self, data_dir, num_mfccs=13, features='mfcc-2d', device='cuda', 
                 fft_args={"win_length": 1024, "hop_size": 64, "pad": 0},
                 weight_normalization=True,
                 feature_dict={"mfcc": True, "centroid": True, "zcr": True, "rms": True, "flux": True, "flatness": True},
                 stat_dictionary=None):
        self.data_dir = data_dir
        self.features = features
        self.device = device
        self.num_mfccs = num_mfccs
        self.win_length = fft_args["win_length"]
        self.hop_size = fft_args["hop_size"]
        self.pad = fft_args["pad"]
        if stat_dictionary:
          self.stat_dictionary = pickle.load(stat_dictionary)
        else:
           self.stat_dictionary = stat_dictionary
        self.feature_dict = feature_dict
        self.weight_normalization = weight_normalization


        self.zip = zipfile.ZipFile(data_dir)
        self.files = self.zip.namelist()[1:]
        # self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".wav")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]

        if self.features == 'params':
            params_array, _ = self.get_benjo_params(index)
            return torch.tensor(params_array, dtype=torch.float32).to(self.device) / 126
        file = self.zip.open(path)
        waveform, sample_rate = torchaudio.load(file, normalize=True, format='wav')
        waveform = waveform[0, :].to(self.device)
        cupy_waveform = cp.asarray(waveform)

        if 'bag-of-frames' == self.features:
            feature_list = []
            MFCC = torchaudio.transforms.MFCC(sample_rate=sample_rate,
                                              n_mfcc=self.num_mfccs,
                                              melkwargs={"n_fft": 1024,
                                                         "win_length": self.win_length,
                                                         "hop_length": self.hop_size,
                                                         "pad": self.pad,
                                                         "n_mels": 101, "center": False}).to(self.device)
            mfcc = MFCC(waveform)
            mean = torch.mean(mfcc, axis=1).reshape(13, 1)
            std = torch.std(mfcc, axis=1).reshape(13, 1)
            if self.stat_dictionary:
               mean -= self.stat_dictionary["mfcc-mean"][0]
               std -= self.stat_dictionary["mfcc-mean"][1]
               mean /= self.stat_dictionary["mfcc-std"][0]
               std /= self.stat_dictionary["mfcc-std"][1]
            summary_tensor = torch.hstack([mean, std])

            if self.weight_normalization:
              summary_tensor /= self.num_mfccs
            
            if self.feature_dict["centroid"]:
              get_spectral_centroid = torchaudio.transforms.SpectralCentroid(sample_rate=sample_rate,
                                                               n_fft=self.win_length,
                                                               win_length=self.win_length,
                                                               hop_length=self.hop_size,
                                                               pad=self.pad).to(self.device)
              sp_centroid = get_spectral_centroid(waveform)[13:]
              mean, std = torch.nanmean(sp_centroid), torch.std(sp_centroid)
              sp_centroid = torch.tensor([mean, std], device=self.device, dtype=torch.float32)
              if self.stat_dictionary:
                sp_centroid -= self.stat_dictionary["centroid-mean"]
                sp_centroid /= self.stat_dictionary["centroid-std"]
              summary_tensor = torch.vstack([summary_tensor, sp_centroid])
              
            if self.feature_dict["zcr"]:
              zcr = get_zero_crossings(cupy_waveform, self.win_length, self.hop_size)
              mean, std = cp.mean(zcr), cp.std(zcr)
              zcr = torch.tensor([mean, std], device=self.device, dtype=torch.float32)
              if self.stat_dictionary:
                zcr -= self.stat_dictionary["zcr-mean"]
                zcr /= self.stat_dictionary["zcr-std"]
              summary_tensor = torch.vstack([summary_tensor, zcr])
            
            if self.feature_dict["rms"]:
              rms = get_rms(cupy_waveform, self.win_length, self.hop_size)
              mean, std = float(cp.mean(rms)), float(cp.std(rms))
              rms = torch.tensor([mean, std], device=self.device, dtype=torch.float32)
              if self.stat_dictionary:
                rms -= self.stat_dictionary["rms-mean"]
                rms /= self.stat_dictionary["rms-std"]
              summary_tensor = torch.vstack([summary_tensor, rms])

            if self.feature_dict["flux"]:
              spectral_flux = get_spectral_flux(cupy_waveform, self.win_length, self.hop_size)
              mean, std = float(cp.mean(spectral_flux)), float(cp.std(spectral_flux))
              spectral_flux = torch.tensor([mean, std], device=self.device, dtype=torch.float32)
              if self.stat_dictionary:
                spectral_flux -= self.stat_dictionary["flux-mean"]
                spectral_flux /= self.stat_dictionary["flux-std"]
              summary_tensor = torch.vstack([summary_tensor, spectral_flux])


            if self.feature_dict["flatness"]:
              spectral_flatness = get_spectral_flatness(cupy_waveform, self.win_length, self.hop_size)[5:]
              mean, std = float(cp.mean(spectral_flatness)), float(cp.std(spectral_flatness))
              spectral_flatness = torch.tensor([mean, std], device=self.device, dtype=torch.float32)
              if self.stat_dictionary:
                spectral_flatness -= self.stat_dictionary["flatness-mean"]
                spectral_flatness /= self.stat_dictionary["flatness-std"]
              summary_tensor = torch.vstack([summary_tensor, spectral_flatness])
            return summary_tensor

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

