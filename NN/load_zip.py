from torch.utils.data import Dataset
import torchaudio
import zipfile


class BenjoDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.zip = zipfile.ZipFile(data_dir)
        self.files = self.zip.namelist()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]

        file = self.zip.open(path)
        waveform, sample_rate = torchaudio.load(file, normalize=False, format='wav')

        return waveform
