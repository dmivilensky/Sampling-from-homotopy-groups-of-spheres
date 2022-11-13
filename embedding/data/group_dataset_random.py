import numpy
import random
from torch.utils.data import Dataset


class GroupDatasetRandom(Dataset):
    def __init__(self, sample_count, generators):
        self.sample_count = sample_count
        self.generators = generators

    def __len__(self):
        return self.sample_count

    def __getitem__(self, idx):
        length = max(1, int(numpy.random.poisson(lam=5.0)))
        sequence = [random.choice(list(range(1, 1 + 2 * self.generators)))]
        for _ in range(length - 1):
            sequence.append(random.choice(list(
                set(range(1, 1 + 2 * self.generators)) -
                set([1 + (sequence[-1] + 1) % 4])
            )))
        sequence = "".join(map(str, sequence))

        """
        If you want to guarantee the non-reducability

        while any([str(i) + str(1 + (i + 1) % 4) in sequence for i in range(1, 1 + 2 * self.generators)]):
            for i in range(1, 1 + 2 * self.generators):
                sequence = sequence.replace(str(i) + str(1 + (i + 1) % 4), "")
        """

        return {'sequence': (list(map(int, list(sequence))), length)}
