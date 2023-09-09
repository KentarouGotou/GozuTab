
import os
import sys
import json
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import librosa

class Processing:
    
    def __init__(self, preproc_mode="cqt"):
        # directory path to the files
        self.audio_dir_path = "../mydataset/wav_files/"
        self.tab_dir_path = "../mydataset/converted/2/"
        
        # prepresentation and its labels storage
        self.output = {}
        
        #preprocessing modes
        self.preproc_mode = preproc_mode
        self.downsample = True
        self.normalize = True
        self.sr_downs = 22050
        
        # CQT parameters
        self.cqt_n_bins = 192
        self.cqt_bins_per_octave = 24
        
        # save file path
        if self.preproc_mode == "cqt":
            self.save_path = "preprocessed/cqt_repr/"
            
    def get_filename(self, number):
        filenames = sorted(os.listdir(self.tab_dir_path))
        filenames = list(filter(lambda x: x.endswith(".json"), filenames))
        return filenames[number][:-5]
    
    def load_tab_file(self, filename):
        with open(self.tab_dir_path + filename + ".json", "r") as f:
            tab = json.load(f)
        return tab
        
    def load_and_save_reprocessed_file(self, number):
        # get the file name
        filename = self.get_filename(number)
        tab = self.load_tab_file(filename)
        for i in tab:
            print(i)
        
        
def main(args):
    mode = args[0]
    generator = Processing(mode)
    generator.load_and_save_reprocessed_file(0)
    
    
if __name__ == "__main__":
    args = sys.argv
    main(args)