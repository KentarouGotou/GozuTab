import torch.utils.data as data
import os
import numpy as np
from torchvision import transforms

class Mydataset(data.Dataset):
    def __init__(self, data_folder_path="../mydataset", transform=None, context_window_size=9):
        self.data_folder_path = data_folder_path
        self.audio_paths = self.data_folder_path + "/wav_files"
        self.data_paths = self._get_data_paths(data_folder_path)
        self.transform = transform
        self.context_window_size = context_window_size
        self.halfwin = context_window_size // 2

    def __getitem__(self, index):
        path = self.data_paths[index]

        #load data
        data = np.load(path, allow_pickle=True)
        loaded_audio = data["audio"]
        loaded_label = data["tab"]

        #generate data
        audio = []
        full_audio = np.pad(loaded_audio, [(self.halfwin, self.halfwin), (0,0)], "constant")
        for i in range(len(loaded_audio)):
            sample_audio = full_audio[i:i+self.context_window_size]
            sample_audio = np.expand_dims(np.swapaxes(sample_audio, 0, 1), -1)
            audio.append(sample_audio)
        audio = np.array(audio)
        label = loaded_label
        
        full_audio = np.expand_dims(loaded_audio, -1)
        
        return full_audio, label

    def _get_data_paths(self, data_folder_path):
        #get all data paths
        data_paths = []
        for file in os.listdir(data_folder_path):
            data_paths.append(os.path.join(data_folder_path, file))
        return data_paths

    def __len__(self):
        return len(self.data_paths)

#make transform
transform = transforms.Compose([
    transforms.ToTensor()
])

