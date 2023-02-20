from typing import List, Tuple, Iterable

import math

from numpy import random
from random import sample, randint
import freegroup.tools as tools
from itertools import repeat


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
    elif method == "constant":
        return radius


def free_group_bounded(generators_number=2, max_length=5, random_length_method = "uniform"):
    generators = set(range(1, generators_number + 1)) | set(range(-generators_number, 0))

    while True:
        length = random_length(max_length, method = random_length_method)
        word = sample(generators, 1)
        for _ in range(length-1):
            factor = sample(generators - set([-word[-1]]), 1)[0]
            word.append(factor)

        yield word


def normal_closure(generator, fg_dimension: int, method: str = 'conjugation', **params):
    if method == 'conjugation':
        return normal_closure_conjugation(generator, fg_dimension, **params)
    
    if method == 'brackets':
        return normal_closure_brackets(generator, fg_dimension, **params)

    raise ValueError('Unknown method')


def normal_closure_conjugation(generator, generators_number=2, max_length=5):
    while True:
        length = random_length(max_length)
        word = []

        while True:
            factor = generator if random.random() > 0.5 else tools.reciprocal(generator)

            conjugator = next(free_group_bounded(
                generators_number=generators_number, 
                max_length=(length - len(word) - len(factor)) // 2
            ))
            new_word = word + tools.conjugation(factor, conjugator)
            new_word = tools.normalize(new_word)
            if len(new_word) > max_length:
                break
            word = new_word

        yield word


def random_bracket_sequence(n):
    """Generates a balanced sequence of n +1s and n -1s corresponding to correctly nested brackets."""
    # "Generating binary trees at random", Atkinson & Sack, 1992

    # Generate a randomly shuffled sequence of n +1s and n -1s
    # These are steps 1 and 2 of the algorithm in the paper
    seq = [-1, 1] * n
    random.shuffle(seq)

    # This now corresponds to a balanced bracket sequence (same number of
    # opening and closing brackets), but it might not be well-formed
    # (brackets closed before they open). Fix this up using the bijective
    # map in the paper (step 3).
    prefix = []
    suffix = []
    word = []
    partial_sum = 0
    for s in seq:
        word.append(s)
        partial_sum += s
        if partial_sum == 0: # at the end of an irreducible balanced word
            if s == -1: # it was well-formed! append it.
                prefix += word
            else:
                # it was not well-formed! fix it.
                prefix.append(1)
                suffix = [-1] + [-x for x in word[1:-1]] + suffix
            word = []

    return prefix + suffix


def random_from_identities(depth: int, identites: List[Tuple]):
    seq = random_bracket_sequence(depth)

    match, stack = [None] * len(seq), []

    for i, c in enumerate(seq):
        stack.append((i, c))
        if len(stack) < 2:
            continue
        (i1, c1), (i2, c2) = stack[-2], stack[-1]
        if c1 == -c2:
            del stack[-2:]
            match[i1] = i2
            match[i2] = i1

    sampled = [None] * len(seq)

    for idx, match_idx in enumerate(match):
        sampled[idx], sampled[match_idx] = identites[random.choice(len(identites))]
        if random.random() > 0.5:
            sampled[idx], sampled[match_idx] = sampled[match_idx], sampled[idx]
    
    return sum(sampled, [])


def normal_closure_brackets(generator, free_group_dimension: int, max_depth: int, random_depth_method: str = 'uniform'):
    identities = [([-x], [x]) for x in range(1, free_group_dimension + 1)]
    base, i_base = generator, tools.reciprocal(generator)
    for t in range(0, len(base)):
        identities.append((base[:t], base[t:]))
        identities.append((i_base[:t], i_base[t:]))

    while True:
        yield random_from_identities(random_length(max_depth, random_depth_method), identities)


def _random_commutator(words: Iterable):
    words = list(words)
    if len(words) == 0:
        raise ValueError
    if len(words) == 1:
        return words[0]
    if len(words) == 2:
        return tuple(words)
    if len(words) >= 2:
        split_idx = randint(1, len(words) - 1)
        return (_random_commutator(words[:split_idx]), _random_commutator(words[split_idx:]))


def random_order_commutant(closures: List[Iterable]):
    return map(_random_commutator, zip(*closures))

