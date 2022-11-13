#!/usr/bin/env python3

import os
os.chdir('..')

import torch
import itertools
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from know_how_optimizer.lr_scheduler import WarmupMultiStepLR
from model.encoder import EncoderLSTM, EncoderConvolution
from utils.free_group import pairwise_distances
from data import GroupDatasetRandom
import argparse


def collate_wrapper(batch):
    return pad_sequence([
        torch.IntTensor(sen['sequence'][0])
        for sen in batch
    ], batch_first=True, padding_value=0), [sen['sequence'][1] for sen in batch]


epochs = 40
batch_size = 10
sample_count = 100
steps = 30
generators = 2
group_dataset = GroupDatasetRandom(sample_count, generators)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Train embeddings for Cayley graph of the free group.")
parser.add_argument("arch", type=str, help="Architecture of encoder (lstm, conv)", default="lstm")
args = parser.parse_args()

if args.arch == "lstm":
    model = EncoderLSTM(generators).to(device)
elif args.arch == "conv":
    model = EncoderConvolution(generators).to(device)
else:
    raise AttributeError("Undefined encoder achitecture `" + args.arch + "`")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = WarmupMultiStepLR(optimizer, warmup_iters=10)
criterion = torch.nn.MSELoss().to(device)

for _ in range(epochs):
    epoch_loss = 0.0
    iters = 0

    model.train()
    data_loader = itertools.islice(DataLoader(
        group_dataset, batch_size=batch_size,
        shuffle=True, num_workers=0,
        collate_fn=collate_wrapper
    ), batch_size*steps)

    for batch in data_loader:
        optimizer.zero_grad()
        sequences, sequences_lengths = batch

        embeddings = model(sequences, sequences_lengths)
        loss = criterion(
            torch.cdist(embeddings, embeddings, p=2).unsqueeze(0),
            pairwise_distances(sequences).unsqueeze(0)
        ) / (batch_size**2)
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        iters += 1

    print('loss:', epoch_loss / iters)
