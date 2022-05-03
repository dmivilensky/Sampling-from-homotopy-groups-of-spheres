import torch
from itertools import islice
from torch.utils.data import Dataset
from free_group import free_group_bounded, is_from_singleton_normal_closure


def cross_encoding(word, generators_number):
    encoded_word = torch.zeros((len(word), generators_number))
    for i, factor in enumerate(word):
        encoded_word[i, abs(factor) - 1] = factor/abs(factor)
    return encoded_word


def pad(word, max_length):
    return torch.cat((word, torch.zeros((max_length - word.shape[0], word.shape[1]))), 0).to(torch.double)


class NormalClosure(Dataset):
    def __init__(self, generator, dataset_length, 
                 input_preprocessing=None, 
                 output_preprocessing=None, 
                 generators_number=2, max_length=10):
        
        baseline_group = free_group_bounded(generators_number=generators_number, max_length=max_length)
        condition = lambda word: is_from_singleton_normal_closure(generator, word)
        self.class_1 = list(islice(filter(condition, baseline_group), dataset_length // 2))
        self.class_0 = list(islice(filter(lambda word: not condition(word), baseline_group), dataset_length // 2))
        
        if input_preprocessing is None:
            input_preprocessing = lambda word: pad(cross_encoding(word, generators_number=generators_number), max_length).t()
        if output_preprocessing is None:
            output_preprocessing = lambda y: torch.Tensor([y]).to(torch.double)

        self.input_preprocessing = input_preprocessing
        self.output_preprocessing = output_preprocessing
        
        self.dataset_length = dataset_length

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, i):
        if i < self.dataset_length // 2:
            word = self.class_1[i]
            indicator = 1
        else:
            word = self.class_0[i - self.dataset_length // 2]
            indicator = 0
        return self.input_preprocessing(word), self.output_preprocessing(indicator)


if __name__ == "__main__":
    generators_number = 2
    max_length = 10

    dataset = NormalClosure([[1]], 1, generators_number=generators_number, max_length=max_length)
    print(dataset[0])
