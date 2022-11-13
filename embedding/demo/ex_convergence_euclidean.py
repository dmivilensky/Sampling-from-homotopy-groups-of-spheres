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
epochs = 1000
generators = 2
dimension = 256

if not os.path.isfile(FIGURES_PATH + f'conv_{epochs}_{generators}.data'):
    batch_size = 10
    sample_count = 100
    steps = 50
    group_dataset = GroupDatasetRandom(sample_count, generators)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Train embeddings for Cayley graph of the free group.")
    parser.add_argument("arch", type=str, help="Architecture of encoder (lstm, conv)", default="lstm")
    args = parser.parse_args()


    if args.arch == "lstm":
        model = EncoderLSTM(generators, dimension=dimension).to(device)
    elif args.arch == "conv":
        model = EncoderConvolution(generators).to(device)
    else:
        raise AttributeError("Undefined encoder achitecture `" + args.arch + "`")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = WarmupMultiStepLR(optimizer, warmup_iters=10)
    criterion = torch.nn.MSELoss().to(device)

    losses = []
    dists = []
    grads = []

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
        dists.append(parameters_to_vector(model.parameters()).detach().numpy())
        grad = 0.0
        for param in model.parameters():
            grad += torch.norm(param.grad).detach().item()**2
        grad = grad**(1/2)
        grads.append(grad)

    dists = np.array(dists)
    dists = dists - parameters_to_vector(model.parameters()).detach().numpy()
    dists = np.apply_along_axis(np.linalg.norm, 1, dists)

    with open(FIGURES_PATH + f'conv_{dimension}_{generators}.data', 'wb') as f:
        pickle.dump((losses, dists, grads), f)

else:
    with open(FIGURES_PATH + f'conv_{dimension}_{generators}.data', 'rb') as f:
        losses, dists, grads = pickle.load(f)

plt.rcParams.update({'font.size': 16})

fig = plt.figure()
ax = plt.axes()
ax.set_yscale("log")

ax.plot(losses)
ax.set_xlabel('$N$')
ax.set_ylabel('$f(x^N)$')
ax.set_title(f'$d = {dimension}, gen = {generators}$')
ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(FIGURES_PATH + f'conv_{epochs}_{generators}.pdf')
plt.show()

fig = plt.figure()
ax = plt.axes()
ax.set_yscale("log")

ax.plot(grads)
ax.set_xlabel('$N$')
ax.set_ylabel('$\\|\\nabla f(x^N)\\|_2$')
ax.set_title(f'$d = {dimension}, gen = {generators}$')
ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(FIGURES_PATH + f'conv_grad_{epochs}_{generators}.pdf')
plt.show()

fig = plt.figure()
ax = plt.axes()
ax.set_yscale("log")

ax.plot(dists)
ax.set_xlabel('$N$')
ax.set_ylabel('$\\|x^N - x^*\\|_2$')
ax.set_title(f'$d = {dimension}, gen = {generators}$')
ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(FIGURES_PATH + f'conv_arg_{epochs}_{generators}.pdf')
plt.show()
