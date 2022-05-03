import time
import random
from math import exp
from free_group import is_cyclic_permutation, reciprocal, normalize, free_group_bounded, normal_closure, print_word


def distance_to_singleton_normal_closure(word, generators, approximation="reduction"):
    if len(generators) != 1:
        raise NotImplementedError('`generators` must contain only one generator ;)')

    if approximation == "reduction":
        contained_smth_to_reduce = True
        generator = generators[0]
        generator_len = len(generator)

        while contained_smth_to_reduce:
            contained_smth_to_reduce = False
            new_word = []

            i = 0
            while i <= len(word) - generator_len:
                subword = word[i:i + generator_len]
                if is_cyclic_permutation(subword, generator) or is_cyclic_permutation(subword, reciprocal(generator)):
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


def optimize(initial_word, function, p=0.1, generators_number=2, max_length=10, early_stop=None, max_iters=100):
    # https://arxiv.org/pdf/1703.03334.pdf 3.2 (1 + 1) EA
    # https://arxiv.org/pdf/1812.11061.pdf 2.2 (\mu + \lambda) EA

    generators = set(range(1, generators_number + 1)) | set(range(-generators_number, 0))
    def mutate(word):
        mutated_word = word.copy()
        for i in range(len(mutated_word)):
            if random.random() < p:
                mutated_word[i] = random.sample(generators - set([mutated_word[i]]), 1)[0]
        return normalize(mutated_word)
    
    word = initial_word.copy()

    for _ in range(max_iters):
        if early_stop(word):
            break

        new_word = mutate(word)

        if len(new_word) == 0:
            continue

        if function(new_word) < function(word):
            word = new_word.copy()

    return word, early_stop(word)


class EvolutionarySampler:
    def __init__(self, generators_number=2, max_length=10, 
                 baseline="free", regularization=None, 
                 regularization_coeff=1e-1, max_iters=50,
                 mutation_rate=0.1):
        singleton_generators = [[[x]] for x in range(1, generators_number+1)]
        joint_generator = [[x for x in range(1, generators_number+1)]]

        self.generators_number = generators_number
        self.max_length = max_length
        self.max_iters = max_iters
        self.mutation_rate = mutation_rate

        if baseline == "free":
            self.baseline_group = free_group_bounded(generators_number=generators_number, max_length=max_length)
            base_function = lambda word: sum(distance_to_singleton_normal_closure(word, gen) for gen in singleton_generators) + distance_to_singleton_normal_closure(word, joint_generator)
        elif baseline == "joint":
            self.baseline_group = normal_closure(joint_generator, generators_number=generators_number, max_length=max_length)
            base_function = lambda word: sum(distance_to_singleton_normal_closure(word, gen) for gen in singleton_generators)
        elif baseline == "singleton":
            self.baseline_group = normal_closure(singleton_generators[0], generators_number=generators_number, max_length=max_length)
            base_function = lambda word: sum(distance_to_singleton_normal_closure(word, gen) for gen in singleton_generators[1:]) + distance_to_singleton_normal_closure(word, joint_generator)
        else:
            raise NotImplementedError('unknown `baseline`')

        self.early_stop = lambda word: base_function(word) == 0

        if regularization is not None:
            if regularization == "exp":
                self.function = lambda word: base_function(word) + regularization_coeff * exp(1/(len(word) + 1e-2))
            else:
                raise NotImplementedError('unknown `regularization`')
        else:
            self.function = base_function

    def __iter__(self):
        return self

    def __next__(self):
        success = False
        while not success:
            initial_word = next(self.baseline_group)
            resulting, success = optimize(
                initial_word, self.function, 
                generators_number=self.generators_number, 
                max_length=self.max_length, early_stop=self.early_stop,
                p=self.mutation_rate, max_iters=self.max_iters)
        return resulting


if __name__ == "__main__":
    sampler = EvolutionarySampler(baseline="free", generators_number=3, max_length=100)
    for i in range(1):
        start = time.time()
        print_word(next(sampler))
        print(time.time() - start, 's')
