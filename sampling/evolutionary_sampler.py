import time
import random
import warnings

from group_tool.reduced_words import *
from group_tool.utils import print_word


def distance_to_singleton_normal_closure(word, generators, approximation="reduction"):
    if len(generators) != 1:
        raise NotImplementedError('`generators` must contain only one generator ;)')

    if approximation == "reduction":
        contained_smth_to_reduce = True
        generator = generators[0]
        generator_len = len(generator)

        doubled_generator  = generator * 2
        doubled_reciprocal = reciprocal(generator) * 2

        while contained_smth_to_reduce:
            contained_smth_to_reduce = False
            new_word = []

            i = 0
            while i <= len(word) - generator_len:
                subword = word[i:i + generator_len]
                if occurs(subword, doubled_generator) or occurs(subword, doubled_reciprocal):
                    contained_smth_to_reduce = True
                    i += generator_len
                else:
                    new_word.append(word[i])
                    i += 1
            
            if i < len(word):
                new_word += word[-(len(word)-i):]
            word = normalize(new_word)

        return len(word)
    else:
        raise NotImplementedError('unknown `approximation`')


def better_base(singleton_generators, joint_generator, word, previous_function, first=None):
    if first:
        current_function = 0
        for gen in singleton_generators[:first]:
            current_function += distance_to_singleton_normal_closure(word, gen)
            if current_function >= previous_function:
                return False
        return True

    current_function = 0
    for gen in singleton_generators:
        current_function += distance_to_singleton_normal_closure(word, gen)
        if current_function >= previous_function:
            return False
    current_function += distance_to_singleton_normal_closure(word, joint_generator)
    if current_function >= previous_function:
        return False
    return True


def dist_base(singleton_generators, joint_generator, word, first=None):
    if first:
        return sum(distance_to_singleton_normal_closure(word, gen) for gen in singleton_generators[:first])
    return distance_to_singleton_normal_closure(word, joint_generator) + \
           sum(distance_to_singleton_normal_closure(word, gen) for gen in singleton_generators)


def optimize(
    word, dist, better, mutation_rate=0.1, generators_number=2, 
    max_iters=10, method='gemmate', fixed_size=False, verbose=True):

    # https://arxiv.org/pdf/1703.03334.pdf 3.2 (1 + 1) EA
    # https://arxiv.org/pdf/1812.11061.pdf 2.2 (\mu + \lambda) EA

    if method == 'gemmate' and fixed_size:
        warnings.warn('gemmate mutation method is not compatible with `fixed_size` set to True')

    generators = set(range(1, generators_number + 1)) | set(range(-generators_number, 0))
    def mutate(word, method='gemmation'):
        mutated_word = word.copy()
        if method == 'gemmate':
            i = random.randint(0, len(word))
            if random.random() < mutation_rate:
                mutated_word.insert(i, random.sample(
                    generators - set([mutated_word[i-1]]) if i == len(word)
                    else generators - set([mutated_word[i], mutated_word[i-1]])
                , 1)[0])
            else:
                mutated_word.pop(min(i, len(word)-1))
        elif method == 'edit':
            for i in range(len(mutated_word)):
                if random.random() < mutation_rate:
                    mutated_word[i] = random.sample(generators - set([mutated_word[i]]), 1)[0]
        else:
            raise NotImplementedError('unknown `method`')
        return mutated_word

    current_function = dist(word)

    if verbose:
        print('INFO: optimization started')

    for _ in range(max_iters):
        new_word = mutate(word, method)
        normalized = normalize(new_word)

        if len(normalized) == 0:
            continue

        if better(normalized, current_function):
            word = (new_word if fixed_size else normalized).copy()
            current_function = dist(normalized)

            if verbose:
                print(f'INFO: f value = {current_function}')
                print_word(normalized)

            if current_function == 0:
                break

    if verbose:
        print(
            f'INFO: optimization finished,', 
            'reached intersection' if current_function == 0 else 'reached max_iters', '\n'
            )

    return normalize(word), current_function == 0


class EvolutionarySampler:
    def __init__(
        self, generators_number=2, max_length=10, 
        exploration_rate=None, baseline="free", first=None, **kwargs):
        
        singleton_generators = [[[x]] for x in range(1, generators_number+1)]
        joint_generator = [[x for x in range(1, generators_number+1)]]

        self.generators_number = generators_number
        self.max_length = max_length
        self.exploration_rate = exploration_rate

        if baseline == "free":
            self.baseline_group = free_group_bounded(
                generators_number=generators_number, max_length=max_length)
        elif baseline == "joint":
            self.baseline_group = normal_closure(joint_generator, 
            generators_number=generators_number, max_length=max_length)
        elif baseline == "singleton":
            self.baseline_group = normal_closure(singleton_generators[0], 
            generators_number=generators_number, max_length=max_length)
        else:
            raise NotImplementedError('unknown `baseline`')

        if baseline in ["free", "joint", "singleton"]:
            self.dist = lambda word: dist_base(singleton_generators, joint_generator, word, first=first)
            self.better = lambda word, previous_function: better_base(singleton_generators, joint_generator, word, previous_function, first=first)
            if not first:
                self.condition = lambda word: all(
                    is_from_singleton_normal_closure(gen, word) 
                    for gen in singleton_generators
                    ) and is_from_singleton_normal_closure(joint_generator, word)
            else:
                self.condition = lambda word: all(
                    is_from_singleton_normal_closure(gen, word) 
                    for gen in singleton_generators[:first])
        else:
            raise NotImplementedError()

        self.kwargs = kwargs

    def __iter__(self):
        return self

    def __next__(self):
        success = False
        while not success:
            word = next(self.baseline_group)
            if self.condition(word):
                return word
            if random.random() > self.exploration_rate:
                continue
            word, success = optimize(
                word, self.dist, self.better, 
                generators_number=self.generators_number, **self.kwargs)

        return word


if __name__ == "__main__":
    sampler = EvolutionarySampler(
        baseline="singleton", generators_number=5, 
        max_length=60, max_iters=100, mutation_rate=0.8, 
        exploration_rate=1., verbose=False,
        method='gemmate', fixed_size=False, first=3)
    
    start = time.time()
    for i in range(1000):
        print_word(next(sampler))
    print(time.time() - start, 's')
