import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
import librosa.feature
import librosa
import scipy.stats as stats


class BenjoDataset(Dataset):
    def __init__(self, data_dir, num_mfccs=40, features='mfcc-2d', device='cuda'):  # Adjust max_length as needed
        self.data_dir = data_dir
        self.features = features
        self.device = device
        self.num_mfccs = num_mfccs
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".wav")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]

        if self.features == 'params':
            params_array, _ = self.get_benjo_params(index)
            return torch.tensor(params_array, dtype=torch.float32).to(self.device) / 126

        waveform, sample_rate = torchaudio.load(path, normalize=True, format='wav')
        waveform = waveform[0, :].to(self.device)

        if 'mel_spectrogram' == self.features:
            epsilon = 0.01
            transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=40).to(self.device)
            mel_spectrogram = transform(waveform)
            log_mel_power = torch.log(torch.abs(mel_spectrogram) + epsilon) - np.log(epsilon)
            return log_mel_power[:, :480]

        if 'bag-of-frames' == self.features:
            y, sample_rate = librosa.load(path)
            spectrogram = np.abs(librosa.stft(y))
            mfcc_librosa = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=self.num_mfccs)
            spectral_centroid = librosa.feature.spectral_centroid(S=spectrogram, sr=sample_rate)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(S=spectrogram, sr=sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(S=spectrogram, sr=sample_rate)
            spectral_contrast = librosa.feature.spectral_contrast(S=spectrogram, sr=sample_rate)
            zero_crossings = librosa.feature.zero_crossing_rate(y=y)
            rms = librosa.feature.rms(y=y)
            matrix = np.vstack((mfcc_librosa, spectral_centroid, spectral_rolloff, spectral_contrast,
                                spectral_bandwidth, zero_crossings, rms))
            """matrix -= self.means[:, np.newaxis]
            matrix /= self.stds[:, np.newaxis]"""
            summary_matrix = np.zeros((matrix.shape[0], 4))
            for i in range(matrix.shape[0]):
                this_row = matrix[i, :]
                mean = np.mean(this_row)
                std = np.std(this_row)
                skewness = stats.skew(this_row)
                kurtosis = stats.kurtosis(this_row)
                summary_matrix[i, :] = np.array([mean, std, skewness, kurtosis])
            return summary_matrix
            # return torch.tensor(summary_matrix, dtype=torch.float32).to(self.device)

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

