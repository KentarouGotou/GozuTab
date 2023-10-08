import torch.utils.data as data
import os
import numpy as np
import matplotlib.pyplot as plt

class Dataset(data.Dataset):
    def __init__(self, data_dir_path="preprocessed/", mode="cqt", plot=False):
        self.data_dir_path = data_dir_path
        self.mode = mode
        self.plot = plot
        
        # data paths
        self.tab_data_paths = self.get_tab_data_paths()
        self.audio_data_paths = self.get_audio_data_paths()
        
        # audio max length
        if self.mode == "raw_wave":
            self.full_audio_length = 5000000
        else:
            self.full_audio_length = 8192
        # tab max length
        self.full_tab_length = 512
        # padding tab
        self.padding_tab = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0]
        
    def get_tab_data_paths(self):
        data_paths = []
        tab_dir_path = os.path.join(self.data_dir_path, "tab")
        for file in os.listdir(tab_dir_path):
            data_paths.append(os.path.join(tab_dir_path, file))
        return data_paths
    
    def get_audio_data_paths(self):
        data_paths = []
        audio_dir_path = os.path.join(self.data_dir_path, self.mode)
        for file in os.listdir(audio_dir_path):
            data_paths.append(os.path.join(audio_dir_path, file))
        return data_paths
    
    def __len__(self):
        return len(self.tab_data_paths)
    
    def __getitem__(self, index):
        # data path
        tab_data_path = self.tab_data_paths[index]
        audio_data_path = self.audio_data_paths[index]
        # load data
        loaded_tab_data = np.load(tab_data_path)
        tempo = loaded_tab_data["tempo"]
        tab_data = loaded_tab_data["tab"]
        loaded_audio_data = np.load(audio_data_path)
        
        # generate audio data
        if self.mode == "raw_wave":
            audio_data = np.pad(loaded_audio_data, [(0, self.full_audio_length - loaded_audio_data.shape[0])])
            # expand dimension
            audio_data = np.expand_dims(audio_data, axis=1)
            # transpose
            audio_data = np.transpose(audio_data, (1, 0))
        else:
            audio_data = np.pad(loaded_audio_data, [(0, 0), (0, self.full_audio_length - loaded_audio_data.shape[1])])
            # expand dimension
            audio_data = np.expand_dims(audio_data, axis=2)
            # transpose
            audio_data = np.transpose(audio_data, (2, 0, 1))
        
        # generate tab data
        # padding
        padding_tab_data = [self.padding_tab for i in range(self.full_tab_length - tab_data.shape[0])]
        tab_data = np.concatenate([tab_data, padding_tab_data])
        
        # plot
        if self.plot:
            print("Tab data shape: ", tab_data.shape, "Audio data shape: ", audio_data.shape)
            # Tab data
            print("Tab data: ", tab_data[0])
            # Audio data
            plt.figure(figsize=(16, 9))
            plt.imshow(audio_data[1000:1200], cmap="jet")
            plt.show()
        return tempo, tab_data, audio_data
