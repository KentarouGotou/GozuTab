from getdataset import Mydataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torch.optim as optim
# from pytorch_memlab import profile
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 7, (3, 1))
        self.conv2 = nn.Conv2d(16, 16, 7, (3, 1))
        self.conv3 = nn.Conv2d(16, 32, 7, (3, 1))
        self.conv4 = nn.Conv2d(32, 32, 5, 2)
        self.conv5 = nn.Conv2d(32, 64, 5, 2)
        self.conv6 = nn.Conv2d(64, 64, 3, 1)
        
        self.pool = nn.MaxPool2d(2, 2)

        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        
    # @profile

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))
        x = f.relu(self.conv4(x))
        x = f.relu(self.conv5(x))
        # x = f.relu(self.conv6(x))
        # print(x.shape)
        x = self.pool(x)
        x = self.drop1(x)
        x = self.drop2(x)
        x = x.reshape(384, x.shape[0], -1)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=3023)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=384, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 4096)
        self.fc2 = nn.Linear(4096, vocab_size)
        self.flatten = nn.Flatten()

    def forward(self, sequence, x):
        embedding = self.embeddings(sequence)
        x, state = self.rnn(embedding, x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x, state

embedding_dim = 200
hidden_dim = 120
vocab_size = 3024

#set gpu
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

encoder = Encoder().to(device)
decoder = Decoder(vocab_size, embedding_dim, hidden_dim).to(device)

#loss
criterion = nn.CrossEntropyLoss()

#optimizer
encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)

dataset = Mydataset()

dataloader = DataLoader(
    dataset = dataset,
    batch_size=3
    )
    
for epoch in range(3):
    epoch_loss = 0

    for data, label in dataloader:
        
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        input_tensor = torch.tensor(data, dtype=torch.float32, device=device)
        output_tensor = torch.tensor(label, dtype=torch.int64, device=device)
        
        encoder_state = encoder(input_tensor)
        
        source = output_tensor[:, :-1]
        target = output_tensor[:, 1:]
        
        loss = 0
        
        decoder_output, _ = decoder(source, encoder_state)
        
        for j in range(decoder_output.size()[1]):
            loss += criterion(decoder_output[:, j, :], target[:, j])
            
        epoch_loss += loss.item()
        
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
    
    print("Epoch %d: %.2f" % (epoch, epoch_loss))