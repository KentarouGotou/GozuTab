import torch.utils.data as data
import os
import numpy as np
import matplotlib.pyplot as plt

class Dataset(data.Dataset):
    def __init__(self, data_dir_path="preprocessed/", mode="cqt", context_window=9, plot=False):
        self.data_dir_path = data_dir_path
        self.mode = mode
        self.context_window = context_window
        self.half_context_window = int(self.context_window / 2)
        self.plot = plot
        
        # data paths
        self.tab_data_paths = self.get_tab_data_paths()
        self.audio_data_paths = self.get_audio_data_paths()
        
        # full audio length
        if self.mode == "raw_wave":
            self.full_audio_length = 5000000
        else:
            self.full_audio_length = 8000
        
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
        audio_data = np.empty((self.full_audio_length, loaded_audio_data.shape[0], self.context_window))
        # padding
        full_audio_data = np.pad(loaded_audio_data, ((0, 0), (0, self.full_audio_length - loaded_audio_data.shape[1])))
        padd_audio_data = np.pad(full_audio_data, ((0, 0), (self.half_context_window, self.half_context_window)))
        # insert
        for i in range(self.full_audio_length):
            audio_data[i] = padd_audio_data[:, i:i+self.context_window]
        # expand dim
        audio_data = np.expand_dims(audio_data, -1)
        
        # plot
        print(tab_data.shape, audio_data.shape)
        # Tab data
        print("Tab data: ", tab_data[0])
        # Audio data
        if self.plot:
            plt.figure(figsize=(16, 9))
            plt.imshow(full_audio_data[:,1000:1200], cmap="jet")
            plt.show()
        return tempo, tab_data, audio_data