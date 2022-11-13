from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from tqdm import tqdm

class TokenSampler(nn.Module):
    def __init__(self): super().__init__()

class ArgmaxTokenSampler(TokenSampler):
    def __init__(self): super().__init__()

    def forward(self, scores):
        return torch.argmax(scores, dim = -1)

class TemperatureTokenSampler(TokenSampler):
    def __init__(self, temperature, samples_num, keepdim=False):
        super().__init__()
        self.temperature = temperature
        self.samples_num = samples_num
        self.keepdim = keepdim

    def forward(self, scores):
        probas = F.softmax(scores / self.temperature)

        sampled = torch.multinomial(probas, num_samples=self.samples_num, replacement=True)

        if self.keepdim:
            return sampled
        return sampled.squeeze(dim = -1)

class TopKTokenSampler(TokenSampler):
    def __init__(self, delegate: 'TokenSampler', k: int): 
        super().__init__()
        self.k = k

    def scores(self, scores):
        _, idx = torch.topk(scores, self.k, dim = -1, sorted = False)
        scores = torch.scatter(scores, dim = -1, idx = idx, src = -torch.inf)
        
        return self.delegate(scores)

class SuppressTokenSampler(TokenSampler):
    def __init__(self, delegate: 'TokenSampler', suppress_tokens: List[int]):
        self.delegate = delegate
        self.suppress_tokens = torch.tensor(suppress_tokens, dtype=torch.long)

    def forward(self, scores):
        scores = torch.scatter(scores, dim = -1, idx = self.suppress_tokens, src = -torch.inf)

        return self.delegate(scores)


def left_to_right_word_sampling(
    model: 'nn.Module',
    sampler: 'TokenSampler',
    input: 'torch.Tensor',
    length: int,
    device: 'torch.device',
    verbose: bool
):
    input_length, batch_size = input.shape
    sampled = torch.zeros((length, batch_size), device = device)
    sampled[:input_length, :] = input

    idx_iter = range(input_length, length)
    if verbose:
        idx_iter = tqdm(idx_iter)

    for t in idx_iter:
        scores = model(sampled[:t])
        sampled[t] = sampler(scores[-1])
    
    return sampled
