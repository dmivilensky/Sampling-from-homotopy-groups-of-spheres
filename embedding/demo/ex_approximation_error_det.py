#!/usr/bin/env python3

import os
os.chdir('..')

import torch
import numpy as np
import itertools
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector
from know_how_optimizer.lr_scheduler import WarmupMultiStepLR
from model.encoder import EncoderLSTM
from utils.free_group import pairwise_distances
from data import GroupDatasetBounded
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os


def collate_wrapper(batch):
    return pad_sequence([
        torch.IntTensor(sen['sequence'][0])
        for sen in batch
    ], batch_first=True, padding_value=0), [sen['sequence'][1] for sen in batch]


FIGURES_PATH = "./figures/"
epochs = 100
generators = 2
dimension = 256
small = 5

if not os.path.isfile(FIGURES_PATH + f'approx_det_{epochs}_{generators}_{small}.data'):
    group_dataset = GroupDatasetBounded(small, generators)
    batch_size = len(group_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Train embeddings for Cayley graph of the free group.")
    parser.add_argument("arch", type=str, help="Architecture of encoder (lstm, conv)", default="lstm")
    args = parser.parse_args()

    if args.arch == "lstm":
        model = EncoderLSTM(generators, dimension=dimension, dropout=0).to(device)
    else:
        raise AttributeError("Undefined encoder achitecture `" + args.arch + "`")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = WarmupMultiStepLR(optimizer, warmup_iters=0, gamma=0.95)
    criterion = torch.nn.MSELoss().to(device)

    for _ in tqdm(range(epochs)):
        epoch_loss = 0.0
        iters = 0

        model.train()
        data_loader = DataLoader(
            group_dataset, batch_size=batch_size,
            shuffle=True, num_workers=0,
            collate_fn=collate_wrapper)
        
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

    losses = []
    for big in tqdm(range(2, small + 3)):
        if big > small:
            group_dataset_test = GroupDatasetBounded(big, generators, small)
        else:
            group_dataset_test = GroupDatasetBounded(big, generators)
        batch_size = len(group_dataset_test)

        epoch_loss = 0.0
        iters = 0

        model.eval()
        data_loader_test = DataLoader(
            group_dataset_test, batch_size=batch_size,
            shuffle=True, num_workers=0,
            collate_fn=collate_wrapper)

        for batch in data_loader_test:
            sequences, sequences_lengths = batch

            embeddings = model(sequences, sequences_lengths)
            loss = criterion(
                torch.cdist(embeddings, embeddings, p=2).unsqueeze(0),
                pairwise_distances(sequences).unsqueeze(0)
            ) / (batch_size**2)

            epoch_loss += loss.item()
            iters += 1

        losses.append(epoch_loss / iters)

    with open(FIGURES_PATH + f'approx_det_{epochs}_{generators}_{small}.data', 'wb') as f:
        pickle.dump(losses, f)

else:
    with open(FIGURES_PATH + f'approx_det_{epochs}_{generators}_{small}.data', 'rb') as f:
        losses = pickle.load(f)

plt.rcParams.update({'font.size': 16})

fig = plt.figure()
ax = plt.axes()
ax.set_yscale("log")

ax.plot(list(range(2, small + 3)), losses)
ax.set_xlabel('$R$')
ax.set_ylabel('$f_R(x^N)$')
ax.set_title(f'$d = {dimension}, gen = {generators}, r = {small}$')
ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(FIGURES_PATH + f'approx_det_{epochs}_{generators}_{small}.pdf')
plt.show()
