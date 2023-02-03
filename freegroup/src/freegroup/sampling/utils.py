from typing import Iterable, List, Callable
from freegroup.tools import Word

import math
import random

from itertools import islice, repeat
from functools import reduce as freduce
from tqdm import tqdm

def random_length(radius, method="uniform"):
    if not isinstance(method, str):
        return method()
    
    if method == "uniform":
        # https://arxiv.org/pdf/1805.08207.pdf 6.3 Uniform sampling in hyperbolic space
        return max(1, int(round(math.acosh(1 + random.random() * (math.cosh(radius) - 1)))))
    elif method == "almost_uniform":
        return max(1, int(round(math.asinh(random.random() * math.cosh(radius - 1)))))
    elif method == "uniform_radius":
        return max(1, int(round(random.random() * radius)))


def iterable_from_batches(batch_sampler: Callable[[], List[Word]], num_tries: int = None) -> Iterable[Word]:
    for _ in repeat(None) if num_tries is None else repeat(None, num_tries):
        for word in batch_sampler():
            yield word


def unique(iterable: Iterable[Word], key = None) -> Iterable[Word]:
    seen = set()

    for el in iterable:
        test = el if key is None else key(el)
        if test not in seen:
            seen.add(test)
            yield el


def subset(iterable: Iterable[List[Word]], empty: bool = False) -> Iterable[List[Word]]:
    for el in iterable:
        result, subset = [], random.randint(1 if not empty else 0, 2 ** (len(el)) - 1)
        for i, w in enumerate(el):
            if subset & (1 << i):
                result.append(w)
        yield result


def shuffle(iterable: Iterable[List[Word]]) -> Iterable[List[Word]]:
    return map(lambda x: random.sample(x, len(x)) , iterable)


def random_union(*iterables: Iterable[Word]) -> Iterable[Word]:
    while True:
        yield from random.choice(*iterables)


def reduce(fn: Callable[[Word, Word], Word], iterables: Iterable[List[Word]]) -> Iterable[Word]:
    return map(lambda l: freduce(fn, l) if l else [], iterables)


def take_unique(take: int, iterable: Iterable[Word], key = tuple, verbose = False) -> Iterable[Word]:
    iterable = islice(unique(iterable, key=key), take)
    return tqdm(iterable, total=take) if verbose else iterable


def prefixes(iterable: Iterable[Word]) -> Iterable[Word]:
    for el in iterable:
        for t in range(len(el)):
            yield el[:t + 1]
