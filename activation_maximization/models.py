from typing import List
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CNNClassifierConfig:
    vocab_dim: int
    embedding_dim: int
    kernels_num: int
    kernels_sizes: List[int]
    dropout_rate: float
    hidden: int


class CNNClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = nn.Embedding(config.vocab_dim, config.embedding_dim)
        
        self.convs = nn.ModuleList([
            nn.Conv2d(1, config.kernels_num, (sz, config.embedding_dim)) for sz in config.kernels_sizes
        ])
            
        self.fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(len(config.kernels_sizes) * config.kernels_num, config.hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, 1)
        )
    
    def forward(self, x):
        # x.shape = (batch_size, length, vocab_size)
        
        x = x * self.mask[None, None, :]
        # x.shape = (batch_size, length, vocab_size)
        
        x = x @ self.embeddings.weight
        # x.shape = (batch_size, length, embedding_dim)

        x = [F.relu(conv(x.unsqueeze(1))).squeeze(-1) for conv in self.convs]
        # x.shape = [(batch_size, kernel_num, length)] * len(kernels_sizes)

        x = [F.max_pool1d(conved, conved.size(2)).squeeze(2) for conved in x]
        # x.shape = [(batch_size, kernels_num)] * len(kernels_sizes)

        x = torch.cat(x, dim = 1)
        # x.shape = (batch_size, kernels_num * len(kernels_sizes))

        logit = self.fc(x)
        # logit.shape = (batch_size, 1)

        return logit
