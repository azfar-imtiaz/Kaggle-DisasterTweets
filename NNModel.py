from torch import nn
import torch.nn.functional as F


class TextClassifierNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(TextClassifierNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, num_classes)
        self.init_weights()

    def init_weights(self):
        init_range = 0.5
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc1.weight.data.uniform_(-init_range, init_range)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-init_range, init_range)
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.uniform_(-init_range, init_range)
        self.fc3.bias.data.zero_()

    def forward(self, text): #, offsets):
        # emb = self.embedding(text, offsets)
        emb = self.embedding(text)
        x = self.fc1(emb)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

