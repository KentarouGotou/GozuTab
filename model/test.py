import getdataset as gd

dataset = gd.Dataset(plot=True)
tempo, tab_data, audio_data = dataset[1]
print(tempo)
print(tab_data.shape)
print(audio_data.shape)