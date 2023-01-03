from typing import Iterable, List, Callable
from group_tool.utils import Word

import math
from numpy import random

from itertools import islice
from functools import reduce as freduce
from tqdm import tqdm

def random_length(radius, method="uniform"):
    if method == "uniform":
        # https://arxiv.org/pdf/1805.08207.pdf 6.3 Uniform sampling in hyperbolic space
        return max(1, int(round(math.acosh(1 + random.random() * (math.cosh(radius) - 1)))))
    elif method == "almost_uniform":
        return max(1, int(round(math.asinh(random.random() * math.cosh(radius - 1)))))
    elif method == "uniform_radius":
        return max(1, int(round(random.random() * radius)))


def iterable_from_batches(sampler: Callable[[], List[Word]]) -> Iterable[Word]:
    while True:
        for word in sampler():
            yield word


def unique(iterable: Iterable[Word]) -> Iterable[Word]:
    seen = set()
    for el in iterable:
        if not tuple(el) in seen:
            seen.add(tuple(el))
            yield el


def subset(iterable: Iterable[List[Word]]) -> Iterable[List[Word]]:
    for el in iterable:
        result, subset = [], random.randint(0, 2 ** (len(el)))
        for i, w in enumerate(el):
            if subset & (1 << i):
                result.append(w)
        yield result


def shuffle(iterable: Iterable[List[Word]]) -> Iterable[List[Word]]:
    return map(lambda x: random.sample(x, len(x)) , iterable)


def join(*iterables: Iterable[Word]) -> Iterable[List[Word]]:
    return zip(*iterables)


def random_union(*iterables: Iterable[Word]) -> Iterable[Word]:
    while True:
        yield from random.choice(*iterables)


def append(iterable: Iterable[Word], iterables: Iterable[List[Word]]) -> Iterable[List[Word]]:
    for els, el in zip(iterables, iterable):
        els.append(el)
        yield els


def reduce(fn: Callable[[Word, Word], Word], iterables: Iterable[List[Word]]) -> Iterable[Word]:
    return map(lambda l: freduce(fn, l) if l else [], iterables)


def take_unique(take: int, iterable: Iterable[Word], verbose = False) -> Iterable[Word]:
    iterable = islice(unique(iterable), take)
    return tqdm(iterable, total=take) if verbose else iterable


def prefixes(iterable: Iterable[Word]) -> Iterable[Word]:
    for el in iterable:
        for t in range(len(el)):
            yield el[:t + 1]
