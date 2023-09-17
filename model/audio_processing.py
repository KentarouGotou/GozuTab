
import os
import sys
import json
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import librosa

class Processing:
    
    def __init__(self, plot=False):
        # directory path to the files
        self.audio_dir_path = "../mydataset/wav_files/"
        self.tab_dir_path = "../mydataset/converted/"
        
        # file name
        self.filename = ""
        
        # prepresentation and its labels storage
        self.output = {}
        
        # preprocessing parameters
        self.sr = 44100
        self.normalize = True
        self.downsample = True
        self.sr_downs = 22050
        
        # STFT parameters
        self.n_fft = 2048
        self.hop_length = 512
        
        # CQT parameters
        self.cqt_n_bins = 192
        self.cqt_bins_per_octave = 24
        
        # plot detail
        self.plot = plot
        if self.plot:
            self.fig = plt.figure(figsize=(20, 16))
            self.ax1 = self.fig.add_subplot(2,2,1)
            self.ax2 = self.fig.add_subplot(2,2,2)
            self.ax3 = self.fig.add_subplot(2,2,3)
            self.ax4 = self.fig.add_subplot(2,2,4)
            
        # effect list
        self.effect_list = {
            # "dead" : "ded",
            # "ghost_note" : "gst",
            "harmonic" : "har",
            "vibrato" : "vib",
            "bend1" : "bn1",
            "bend2" : "bn2",
            "bend3" : "bn3",
            # "slide" : "sld",
            # "hammer" : "h_p",
            "trill" : "trl",
            # "palm_mute" : "brm",
            # "staccato" : "stc",
            # "slap_effect" : "slp"
            "tie" : "tie"
        }
        self.effect_idx = []
        for effect in self.effect_list:
            self.effect_idx.append(self.effect_list[effect])
        
        # save file path
        self.save_dir_path = "preprocessed/"
            
    def get_filename(self, number):
        filenames = sorted(os.listdir(self.tab_dir_path))
        filenames = list(filter(lambda x: x.endswith(".json"), filenames))
        self.filename = filenames[number][:-5]
        return self.filename
    
    def encode_onehot_position(self, data):
        position = np.zeros((6, 21))
        for string in data:
            if data[string] == "-":
                fret = 0
            else:
                fret = int(data[string]) + 1
            position[int(string)-1][fret] = 1
        return position
    
    def encode_onehot_effects(self, data):
        effects = np.zeros((6, 6))
        for string in data:
            if data[string] == "-":
                effect = 0
            else:
                effect = self.effect_idx.index(data[string]) + 1
            effects[int(string)-1][effect] = 1
        return effects
    
    def encode_onehot_ties(self, data):
        ties = np.zeros((6, 2))
        for string in data:
            if data[string] == "-":
                tie = 0
            else:
                tie = 1
            ties[int(string)-1][tie] = 1
        return ties
    
    def load_and_save_tab_file(self):
        with open(self.tab_dir_path + self.filename + ".json", "r") as f:
            tab_json_data = json.load(f)
        header = tab_json_data[0]
        content = tab_json_data[1:]
        all_tab = []
        for tab in content:
            position = self.encode_onehot_position(tab["position"])
            effects = self.encode_onehot_effects(tab["effects"])
            ties = self.encode_onehot_ties(tab["ties"])
            tab = {
                "position" : position,
                "effects" : effects,
                "ties" : ties
            }
            all_tab.append(tab)
        output = [header, all_tab]
        print(output[0], output[1][0])
        return output
        
    def load_and_save_raw_audio_file(self):
        # load the data
        self.sr, data = wavfile.read(self.audio_dir_path + self.filename + ".wav")
        data = data.astype(np.float32)
        # preprocessing
        if self.normalize:
            data = librosa.util.normalize(data)
        if self.downsample:
            data = librosa.resample(data, self.sr, self.sr_downs)
            self.sr = self.sr_downs
        # plot
        if self.plot:
            self.ax1.set_xlabel("normal wave")
            self.ax1.plot(data[2000000:2000100])
        # save the data
        save_path = self.save_dir_path + "raw_wave/"
        np.savez(save_path + self.filename + ".npz", data=data)
        return data
    
    def stft(self, data):
        # stft
        data = librosa.stft(data, n_fft=self.n_fft, hop_length=self.hop_length)
        data = np.abs(data)
        # plot
        if self.plot:
            self.ax2.set_xlabel("stft")
            self.ax2.imshow(librosa.amplitude_to_db(data[:,:200], ref=np.max), cmap="jet")
        # save the data
        save_path = self.save_dir_path + "stft/"
        np.savez(save_path + self.filename + ".npz", data=data)
        return data
    
    def melspectrogram(self, data):
        # melspectrogram
        data = librosa.feature.melspectrogram(data, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        # plot
        if self.plot:
            self.ax3.set_xlabel("melspectrogram")
            self.ax3.imshow(librosa.amplitude_to_db(data[:,:200], ref=np.max), cmap="jet")
        # save the data
        save_path = self.save_dir_path + "melspec/"
        np.savez(save_path + self.filename + ".npz", data=data)
        return data
    
    def cqt(self, data):
        # cqt
        data = librosa.cqt(data, hop_length=self.hop_length, sr=self.sr, n_bins=self.cqt_n_bins, bins_per_octave=self.cqt_bins_per_octave)
        data = np.abs(data)
        # plot
        if self.plot:
            self.ax4.set_xlabel("constant-q transform")
            self.ax4.imshow(data[:,:200], cmap="jet")
        # save the data
        save_path = self.save_dir_path + "cqt/"
        np.savez(save_path + self.filename + ".npz", data=data)
        return data
    
    def load_and_save_reprocessed_file(self, number):
        # get the file name
        filename = self.get_filename(number)
        tab = self.load_and_save_tab_file()
        raw_audio = self.load_and_save_raw_audio_file()
        sftf = self.stft(raw_audio)
        melspec = self.melspectrogram(raw_audio)
        cqt = self.cqt(raw_audio)
        if self.plot:
            plt.show()
        
        
def main(args):
    generator = Processing(plot=False)
    generator.load_and_save_reprocessed_file(0)
    # for i in range(6):
    #     if i == 0:
    #         generator = Processing(plot=True)
    #         generator.load_and_save_reprocessed_file(i)
    #     else:
    #         generator = Processing(plot=False)
    #         generator.load_and_save_reprocessed_file(i)
    
    
if __name__ == "__main__":
    args = sys.argv
    main(args)