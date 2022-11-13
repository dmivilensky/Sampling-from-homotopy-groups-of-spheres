#!/usr/bin/env python3

import os
os.chdir('..')

import torch
# torch.autograd.set_detect_anomaly(True)
import itertools
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from know_how_optimizer.lr_scheduler import WarmupMultiStepLR
from model import distance, Stacked
from model.distance.hyperbolic import HyperbolicHead
from model.encoder import EncoderLSTM
from model.distance import EuclideanHead
from utils.free_group import pairwise_distances
from data import GroupDatasetRandom
import argparse


def collate_wrapper(batch):
    return pad_sequence([
        torch.IntTensor(sen['sequence'][0])
        for sen in batch
    ], batch_first=True, padding_value=0), [sen['sequence'][1] for sen in batch]


dim = 256
generators = 2
group_dataset = GroupDatasetRandom(100, generators)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(
    description="Train embeddings with distances for Cayley graph of the free group.")
parser.add_argument(
    "dist", type=str, help="Distance type (euclid, hyp)", default="lstm")
args = parser.parse_args()

encoder = EncoderLSTM(generators, dimension=dim)

if args.dist == "euclid":
    head = EuclideanHead(dimension=dim, lp=2)
elif args.dist == "hyp":
    head = HyperbolicHead(dimension=dim, method="poincare")
else:
    raise AttributeError("Undefined distance notion `" + args.dist + "`")

model = Stacked(encoder, head).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss().to(device)

batch_size = 10
for _ in range(10):
    epoch_loss = 0.0
    iters = 0

    model.train()
    data_loader = itertools.islice(DataLoader(
        group_dataset, batch_size=batch_size,
        shuffle=True, num_workers=0,
        collate_fn=collate_wrapper
    ), batch_size*50)

    for batch in data_loader:
        optimizer.zero_grad()
        sequences, sequences_lengths = batch

        dists = model(sequences, sequences_lengths)
        loss = criterion(
            dists, pairwise_distances(sequences).unsqueeze(0)
        ) / (batch_size**2)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        iters += 1

    print('loss:', epoch_loss / iters)
