from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

from pathlib import Path
from json import loads

from .models import CNNClassifierConfig, CNNClassifier

import torch
from torch.nn import functional as F
from torchmetrics import MetricTracker, CatMetric, Accuracy
from torch.utils.data import DataLoader

from tqdm import tqdm

@dataclass(frozen = True)
class ClassifierTrain:
    save_path: Path

    generators_number: int

    training_dataset_path: Path
    validation_dataset_path: Path

    model_config: 'CNNClassifierConfig'
    optimizer_config: Tuple[str, Dict[str, Any]]
    scheduler_config: Tuple[str, Dict[str, Any]]

    training_batch_size: int
    validation_batch_size: int

    epochs_num: int

    device_name: str
    dtype_name: str


    def __post_init__(self):

        tokens = ['[PAD]'] + [str(x) for x in range(-self.generators_number, self.generators_number + 1) if x != 0]
        object.__setattr__(self, 'tokens', tokens)
        
        vocab = {key: i for (i, key) in enumerate(tokens)}
        object.__setattr__(self, 'vocabulary', vocab)

        object.__setattr__(self, 'training_dataset', loads(self.training_dataset_path.read_text()))
        object.__setattr__(self, 'validation_dataset', loads(self.validation_dataset_path.read_text()))

        model = CNNClassifier(self.model_config)
        object.__setattr__(self, 'model', model)
        
        name, args = self.optimizer_config
        optimizer_factory = object.__getattribute__(torch, name)
        object.__setattr__(self, 'optimizer', optimizer_factory(self.model.parameters(), **args))
        
        name, args = self.scheduler_config
        scheduler_factory = object.__getattribute__(torch, name)
        object.__setattr__(self, 'scheduler', scheduler_factory(self.optimizer, **args))

        object.__setattr__(self, 'device', torch.device(self.device_name))
        object.__setattr__(self, 'dtype', torch.__getattribute__(torch, self.dtype_name))

        loss_metric = MetricTracker(CatMetric())
        object.__setattr__(self, 'training_loss_metric', loss_metric)
        
        train_acc_metric = MetricTracker(Accuracy(num_classes=self.num_classes, ignore_index = vocab['[PAD]']))
        object.__setattr__(self, 'training_accuracy_metric', train_acc_metric)

        val_acc_metric = MetricTracker(Accuracy(num_classes=self.num_classes, ignore_index = vocab['[PAD]']))
        object.__setattr__(self, 'validation_accuracy_metric', val_acc_metric)




def collate_fn_factory(vocab: Dict[str, int]):
    def collate_fn(batch: List[List[int]]):
        max_length = max(map(len, batch))
        batch = map(lambda word: [str(f) for f in word] + ['[PAD]'] * (max_length - len(word)), batch)
        batch = map(lambda word: [vocab[f] for f in word], batch)
        batch = map(lambda word: torch.tensor(word, dtype = torch.long))
        batch = torch.stack(list(batch))

        return batch[:, :-1]

    return collate_fn


def validate(config):
    device = config.device

    model       = config.model
    dataset     = config.validation_dataset
    batch_size  = config.validation_batch_size
    collate_fn  = collate_fn_factory(config.vocabulary)
    
    metric = config.validation_accuracy_metric

    data_iter = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
        collate_fn = collate_fn,
    )

    metric.increment()
    for batch in tqdm(data_iter):
        input, target = map(lambda v: v.to(device), batch)

        pred = model(input).squeeze(-1)
        metric(pred, target)


def train(config):
    device = config.device
    dtype  = config.dtype

    model       = config.model.to(device = device, dtype = dtype)
    optimizer   = config.optimizer
    scheduler   = config.scheduler
    
    dataset     = config.training_dataset
    batch_size  = config.training_batch_size
    collate_fn  = collate_fn_factory(config.vocabulary)

    loss_metric = config.traininig_loss_metric.to(device)
    acc_metric  = config.training_accuracy_metric.to(device)

    for epoch in range(1, config.epochs_num + 1):
        model.train()

        loss_metric.increment()
        acc_metric.increment()

        data_iter = DataLoader(dataset, batch_size, shuffle = True, collate_fn = collate_fn)

        for batch in tqdm(data_iter):
            input, target = map(lambda v: v.to(device), batch)

            pred = model(input).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_metric(loss.detach())
            acc_metric(pred, target)
        
        scheduler.step()

        validate(config)

        checkpoint_path = config.save_path / 'checkpoints'
        checkpoint_path.mkdir(exists_ok=True, parents=True)

        state = {
            'optimizer': optimizer.state_dict(), 
            'scheduler': scheduler.state_dict(),
            'model': model.state_dict()
        }

        torch.save(state, checkpoint_path / f'{epoch:03d}.pt')

        for attr in filter(lambda attr: attr.endswith('metric'), dir(config)):
            value = object.__getattribute__(config, attr).compute()
            if attr == 'train_loss_metric':
                value = value.mean()
            print(f'{attr}: {value.item()}', end=' | ')

