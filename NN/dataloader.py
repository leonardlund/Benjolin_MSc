import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
import librosa.feature
import librosa


class BenjoDataset(Dataset):
    def __init__(self, data_dir, num_mfccs=90, features='mfcc-2d', device='cuda'):  # Adjust max_length as needed
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
            transform = torchaudio.transforms.MelSpectrogram(sample_rate).to(self.device)
            mel_spectrogram = transform(waveform)
            log_mel_power = torch.log(torch.abs(mel_spectrogram) + epsilon)
            return log_mel_power

        if 'mfcc' in self.features:
            MFCC = torchaudio.transforms.MFCC(sample_rate=sample_rate,
                                              n_mfcc=self.num_mfccs,
                                              melkwargs={"n_fft": 400, "hop_length": 160,
                                                         "n_mels": 101, "center": False}).to(self.device)
            features = MFCC(waveform)
            features = torch.clip(features, min=-25, max=25)
            shape = features.shape
            features += 25
            features /= 50

            y = waveform.cpu().numpy()
            spectrogram = np.abs(librosa.stft(y))
            mfcc_librosa = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=self.num_mfccs)
            spectral_centroid = librosa.feature.spectral_centroid(S=spectrogram, sr=sample_rate)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(S=spectrogram, sr=sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(S=spectrogram, sr=sample_rate)
            spectral_contrast = librosa.feature.spectral_contrast(S=spectrogram, sr=sample_rate)
            zero_crossings = librosa.feature.zero_crossing_rate(y=y)
            stacked_features = np.hstack((spectral_centroid, spectral_bandwidth,
                                          spectral_rolloff, spectral_contrast, zero_crossings))
            # stacked_delta_1st_order = librosa.feature.delta(stacked_features, axis=1)
            # stacked_delta_2nd_order = librosa.feature.delta(stacked_features, axis=1, order=2)
            #all_features = torch.tensor(np.vstack((stacked_features,
            #                                       stacked_delta_1st_order))).to(self.device)
            return stacked_features
            # features /= 25
            if self.features == 'mfcc-bag-of-frames':
                mean = torch.mean(stacked_features, axis=1)
                std = torch.std(stacked_features, axis=1)
                features = torch.cat((mean, std), dim=0)
                # features = features.reshape((1, 2 * self.num_mfccs))
                return features
            else:
                features = features.reshape((1, shape[0], shape[1]))
                return features

    def get_benjo_params(self, index):
        path = self.files[index]
        params_string = path.removeprefix(self.data_dir + '/').removesuffix('.wav')
        params_list = params_string.split('-')
        params_array = np.array(list(map(int, params_list)))
        return params_array, params_string

