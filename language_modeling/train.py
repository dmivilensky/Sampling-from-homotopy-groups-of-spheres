from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from json import loads

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torchmetrics import CatMetric, MetricTracker, Accuracy, MeanMetric

from tqdm import tqdm

from group_tools.reduced_words import is_from_singleton_normal_closure, normalize
from group_tools.utils import print_word
from sampling.language_model_sampler import left_to_right_word_sampling, SuppressTokenSampler, TemperatureTokenSampler

from itertools import chain, combinations

from .models import LSTMModelConfig, LSTMLanguageModel


@dataclass(frozen=True)
class ModuleConfig:
    name: str
    arguments: Dict[str, Any]
    

@dataclass(frozen=True)
class LanguageModelingConfig:
    save_path: Path

    generators_number: int

    training_dataset_path: Path
    validation_dataset_path: Path

    model_config: 'LSTMModelConfig'
    optimizer_config: Tuple[str, Dict[str, Any]]
    scheduler_config: Tuple[str, Dict[str, Any]]

    training_batch_size: int
    validation_batch_size: int

    epochs_num: int

    evaluation_temperature: float
    evaluation_batch_size: int
    evaluation_iterations: int
    evaluation_length: int
    evaluation_prefix_threshold: int

    device_name: str
    dtype_name: str


    def __post_init__(self):

        tokens = ['[PAD]', '[BEGIN]', '[END]'] +\
             [str(x) for x in range(-self.generators_number, self.generators_number + 1) if x != 0]
        object.__setattr__(self, 'tokens', tokens)
        
        vocab = {key: i for (i, key) in enumerate(tokens)}
        object.__setattr__(self, 'vocabulary', vocab)

        object.__setattr__(self, 'training_dataset', loads(self.training_dataset_path.read_text()))
        object.__setattr__(self, 'validation_dataset', loads(self.validation_dataset_path.read_text()))

        model = LSTMLanguageModel(self.model_config)
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

        bases = [[i] for i in range(1, self.generators_number + 1)] +\
            [list(range(1, self.generators_number + 1))]
        object.__setattr__(self, 'bases', bases)
        for subset_bases in chain.from_iterable(
            combinations(bases, r) for r in range(1, self.generators_number + 1 + 1)
        ):
            metric = MetricTracker(MeanMetric())
            object.__setattr__(self, f'evaluation_{"_".join(map(str), subset_bases)}_metric', metric)
        






def collate_fn_factory(vocab: Dict[str, int]):
    def collate_fn(batch: List[List[int]]):
        max_length = max(map(len, batch))
        batch = map(lambda word: ['[BEGIN]'] + [str(f) for f in word] + ['[END]'] + ['[PAD]'] * (max_length - len(word)), batch)
        batch = map(lambda word: [vocab[f] for f in word], batch)
        batch = map(lambda word: torch.tensor(word, dtype = torch.long))
        batch = torch.stack(list(batch))

        return batch[:, :-1], batch[:, :-1]

    return collate_fn

def uncollate_fn_factory(tokens: List[str]):
    def uncollate_fn(batch: 'torch.Tensor'):
        batch = batch.tolist()
        batch = map(lambda word: [tokens[f] for f in word], batch)
        batch = map(lambda word: [int(f) for f in word if f not in ['[PAD]', '[BEGIN]', '[END]']], batch)
        batch = map(normalize, batch)

        return list(batch)
    return uncollate_fn


def validate(config):
    device = config.device

    model = config.model
    
    dataset = config.validation_dataset
    batch_size = config.validation_batch_size

    data_iter = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
        collate_fn = collate_fn_factory(config.vocabulary)
    )

    metric = config.validation_accuracy_metric
    metric.increment()

    model.eval()
    for batch in tqdm(data_iter):
        input, target = map(lambda v: v.to(device), batch)

        pred = model(input.permute(1, 0)).permute(1, 2, 0)
        metric(pred, target)


def evaluate(config):
    device = config.device

    model = config.model
    model.eval()

    sampler = SuppressTokenSampler(
        delegate = TemperatureTokenSampler(
            temperature = config.evaluation_temperature
        ),
        suppress_tokens = [config.vocabulary[t] for t in ['[PAD]', '[BEGIN]', '[END]']]
    ).to(device)

    uncollate_fn = uncollate_fn_factory(config.tokens)

    for _ in range(config.evaluation_iterations):
        input = torch.full(
            (1, config.evaluation_batch_size),
            config.vocabulary['[BEGIN]'],
            device = device,
            dtype = torch.long
        )

        sampled = left_to_right_word_sampling(model, sampler, input, config.evaluation_length, device, verbose = True)
        sampled = uncollate_fn(sampled.permute(1, 0))

        bases = config.bases
        uniques = [set() for _ in range(len(bases))]
        for sequence in sampled:
            word = print_word(sequence, verbose = False)

            for unique, base in zip(uniques, bases):
                for current_end in range(config.evaluation_prefix_threshold, config.evaluation_length):
                    if is_from_singleton_normal_closure([base], sequence[:(current_end + 1)]):
                        unique.add(word[:(current_end + 1)])
        
        for subset in chain.from_iterable(
            combinations(list(zip(bases, uniques)), r) for r in range(1, len(bases) + 1)
        ):
            subset_bases, subset_uniques = zip(*subset)
            metric = object.__getattribute__(config, f'evaluation_{"_".join(map(str), subset_bases)}_metric')
            metric(len(set.intersection(*subset_uniques)))


def train(config):
    device = config.device
    dtype  = config.dtype

    vocab = config.vocabulary

    model = config.model.to(device = device, dtype = dtype)
    optimizer = config.optimizer
    scheduler = config.scheduler

    epochs = config.epochs_num
    dataset = config.traininig_dataset
    batch_size = config.training_batch_size

    loss_metric = MetricTracker(CatMetric())
    acc_metric = MetricTracker(Accuracy(num_classes = len(vocab)))

    for epoch in range(1, epochs + 1):
        
        model.train()

        loss_metric.increment()
        acc_metric.increment()

        data_iter = DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = True,
            collate_fn = collate_fn_factory(vocab)
        )   

        for batch in tqdm(data_iter):
            input, target = map(lambda v: v.to(device), batch)
            # input.shape = (batch_size, length)
            # target.shape = (batch_size, length)

            input = input.permute(1, 0)
            # input.shape = (length, batch_size)
            
            pred  = model(input).permute(1, 2, 0)
            # pred.shape = (batch_size, vocab_dim, length)

            loss = F.cross_entropy(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_metric(loss.detach().cpu())
            acc_metric(pred.cpu(), target.cpu())

        scheduler.step()

        validate(config)

        evaluate(config)

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
