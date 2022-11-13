#!/usr/bin/env python3

import os
os.chdir('..')

import torch
import numpy as np
import itertools
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from know_how_optimizer.lr_scheduler import WarmupMultiStepLR
from model.encoder import EncoderLSTM, EncoderConvolution
from utils.free_group import pairwise_distances
from data import GroupDatasetRandom
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
epochs = 10
generators = 2
attempts = 10

if not os.path.isfile(FIGURES_PATH + f'dims_{epochs}_{generators}.data'):
    batch_size = 10
    sample_count = 100
    steps = 50
    group_dataset = GroupDatasetRandom(sample_count, generators)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Train embeddings for Cayley graph of the free group.")
    parser.add_argument("arch", type=str, help="Architecture of encoder (lstm, conv)", default="lstm")
    args = parser.parse_args()

    dimensions = [
        4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 
        128, 256, 512, 768, 1024, 1536, 2048, 4096
    ]
    plateaus = []
    stds = []

    for d in dimensions:
        problem_losses = []

        for attempt in range(attempts):
            if args.arch == "lstm":
                model = EncoderLSTM(generators, dimension=d).to(device)
            elif args.arch == "conv":
                model = EncoderConvolution(generators).to(device)
            else:
                raise AttributeError("Undefined encoder achitecture `" + args.arch + "`")

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = WarmupMultiStepLR(optimizer, warmup_iters=10)
            criterion = torch.nn.MSELoss().to(device)

            losses = []

            for _ in tqdm(range(epochs)):
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

                losses.append(epoch_loss / iters)
            
            problem_losses.append(losses[-1])

        print('dimension:', d, 'loss:', np.mean(problem_losses), np.std(problem_losses))
        plateaus.append(np.mean(problem_losses))
        stds.append(np.std(problem_losses))

    with open(FIGURES_PATH + f'dims_{epochs}_{generators}.data', 'wb') as f:
        pickle.dump((dimensions, plateaus, stds), f)

else:
    with open(FIGURES_PATH + f'dims_{epochs}_{generators}.data', 'rb') as f:
        dimensions, plateaus, stds = pickle.load(f)

plt.rcParams.update({'font.size': 16})

fig = plt.figure()
ax = plt.axes()
ax.set_xscale("log")
ax.set_yscale("log")

ax.errorbar(dimensions, plateaus, yerr=stds)
ax.set_xlabel('$d$')
ax.set_ylabel('$f(x^N)$')
ax.set_yticks([0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xticks([4, 8, 16, 32, 64, 256, 1024, 4096])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_title(f'$N = {epochs}, gen = {generators}$')
ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(FIGURES_PATH + f'dims_{epochs}_{generators}.pdf')
plt.show()
