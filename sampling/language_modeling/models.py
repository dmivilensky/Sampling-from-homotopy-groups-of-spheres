from dataclasses import dataclass

import torch
from torch import nn



@dataclass(frozen=True)
class LSTMModelConfig:
    vocab_dim: int
    embedding_dim: int
    hidden_dim: int
    layers_num: int
    bias: bool
    lstm_dropout_rate: float
    ff_dropout_rate: float



class LSTMLanguageModel(nn.Module):
    def __init__(self, config):
        
        self.embedding = nn.Embedding(config.vocab_dim, config.embedding_dim)
                    
        self.rnn = nn.LSTM(
            input_size      = config.embedding_dim,
            hidden_size     = config.hidden_dim,
            num_layers      = config.layers_num,
            bias            = config.bias,
            bidirectional   = False,
            dropout         = config.lstm_dropout_rate,
        )

        self.ff = nn.Sequential(
            nn.Dropout(config.ff_dropout_rate),
            nn.Linear(config.hidden_dim, config.vocab_dim),
        )

    def forward(self, x):
        # x.shape = (length, batch_size)
        
        x = self.embedding(x)
        # x.shape = (length, batch_size, embeding_dim)

        x, _ = self.rnn(x)
        # x.shape = (length, batch_size, hidden_dim)

        x = self.ff(x)
        # x.shape = (length, batch_size, vocab_dim)

        return x


class SumJointLanguageModel(nn.Module):
    def __init__(self, models, weights):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.register_buffer('weights', torch.tensor(weights))

    def forward(self, x):
        # x.shape = (length, batch_size)

        outputs = [weight * model(x) for model, weight in zip(self.models, self.weights)]
        # outputs.shape = [(length, batch_size, vocab_dim)] * len(models)

        outputs = torch.stack(outputs)
        # outputs.shape = (models, length, batch_size, vocab_dim)

        scores = outputs.sum(dim = 0)
        # scores.shape = (length, batch_size, vocab_dim)

        return scores
