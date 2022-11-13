import numpy
import random
from torch.utils.data import Dataset


class GroupDatasetBounded(Dataset):
    def __init__(self, max_length, generators, min_length=1):
        self.max_length = max_length
        self.min_length = min_length
        self.generators = generators

    def __len__(self):
        return int(self.generators * ((2 * self.generators - 1) ** self.max_length - 1) / (self.generators - 1)) - int(self.generators * ((2 * self.generators - 1) ** (self.min_length - 1) - 1) / (self.generators - 1))

    def __getitem__(self, idx):
        length = max(self.min_length, int(round(self.max_length * numpy.sqrt(numpy.random.random()))))
        sequence = [random.choice(list(range(1, 1 + 2 * self.generators)))]
        for _ in range(length - 1):
            sequence.append(random.choice(list(
                set(range(1, 1 + 2 * self.generators)) -
                set([1 + (sequence[-1] + 1) % 4])
            )))
        sequence = "".join(map(str, sequence))

        return {'sequence': (list(map(int, list(sequence))), length)}
