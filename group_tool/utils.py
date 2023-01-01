import math
import random
from tqdm import tqdm
from itertools import islice
from numpy import array, pad
from functools import reduce as freduce
from typing import List, Iterable, Callable


def is_sublist(t: List, s: List):
    if len(s) == 0:
        return False
    for i in range(len(s) - len(t)):
        if all(map(lambda v: v[0] == v[1], zip(t, s[i:i+len(t)]))):
            return True
    return False


def random_length(radius, method="uniform"):
    if method == "uniform":
        # https://arxiv.org/pdf/1805.08207.pdf 6.3 Uniform sampling in hyperbolic space
        return max(1, int(round(math.acosh(1 + random.random() * (math.cosh(radius) - 1)))))
    elif method == "almost_uniform":
        return max(1, int(round(math.asinh(random.random() * math.cosh(radius - 1)))))
    elif method == "uniform_radius":
        return max(1, int(round(random.random() * radius)))


LETTERS = "xyzpqrstuvwklmn"


def print_word(word, verbose=True):
    result = []
    for factor in word:
        if type(factor) is list:
            result.append("[" + ",".join(print_word(factor, verbose=False)) + "]")
        else:
            result.append(LETTERS[abs(factor) - 1] + ("⁻¹" if factor < 0 else ""))
    
    if verbose:
        print("".join(result))
    return result


def print_words(words, verbose=True):
    for word in words:
        yield print_word(word, verbose)


def parse_word(string, order=None):
    letters = LETTERS[:order]
    i = 0
    word = []
    while i < len(string):
        if string[i] != "⁻":
            word.append(letters.index(string[i]) + 1)
            i += 1
        else:
            word[-1] = -word[-1]
            i += 2
    return word


Word = List[int]


def to_numpy(words: Iterable[Word]):
    words = list(words)
    max_length = max(map(len, words))
    return array(list(map(lambda v: pad(v, (0, max_length - len(v))), words)))


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
