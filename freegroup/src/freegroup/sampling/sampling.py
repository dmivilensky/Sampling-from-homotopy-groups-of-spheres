from typing import List, Tuple

from numpy import random
from random import sample
from .utils import (
    random_length,
    join,
    shuffle,
    reduce,
    subset,
)
import freegroup.tools as tools
from itertools import repeat


def free_group_bounded(generators_number=2, max_length=5):
    generators = set(range(1, generators_number + 1)) | set(range(-generators_number, 0))

    while True:
        length = random_length(max_length)
        word = sample(generators, 1)
        for _ in range(length-1):
            factor = sample(generators - set([-word[-1]]), 1)[0]
            word.append(factor)

        yield word


def normal_closure(generator: tools.Word, fg_dimension: int, method: str = 'conjugation', **params):
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


# def symmetric_commutant(generators_number=2, max_length=5):
#     closures =\
#         [normal_closure([[i]], generators_number, max_length) for i in range(1, generators_number + 1)] +\
#         [normal_closure([list(range(1, generators_number + 1))], generators_number, max_length)]
#     yield from filter(lambda x: len(x) > 0, 
#         map(normalize, 
#         utils.reduce(commutator,
#         utils.shuffle(
#         zip(*closures)
#     ))))



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


def random_from_identities(depth: int, identites: List[Tuple[tools.Word, tools.Word]]):
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


def normal_closure_brackets(generator: tools.Word, free_group_dimension: int, max_depth: int):
    identities = [([-x], [x]) for x in range(1, free_group_dimension + 1)]
    base, i_base = generator, tools.reciprocal(generator)
    for t in range(0, len(base)):
        identities.append((base[:t], base[t:]))
        identities.append((i_base[:t], i_base[t:]))

    while True:
        yield random_from_identities(random_length(max_depth), identities)


def symmetric_commutant(
    generators: List[tools.Word],
    fg_dimension: int,
    n_multipliers: int,
    closure_method: str = 'conjugation',
    **closure_params,
):
    closures = [normal_closure(g, fg_dimension, closure_method, **closure_params) for g in generators]
    g = reduce(tools.commutator, shuffle(join(*closures)))
    g = reduce(tools.multiply, subset(join(*repeat(g, n_multipliers))))
    yield from filter(lambda x: len(x) > 0, map(tools.normalize, g))
