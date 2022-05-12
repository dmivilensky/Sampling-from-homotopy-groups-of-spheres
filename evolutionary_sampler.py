import time
import random
from functools import partial
from free_group import occurs, reciprocal, normalize, free_group_bounded, normal_closure, print_word, is_from_singleton_normal_closure, parse_word


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


def better_base(singleton_generators, joint_generator, word, previous_function):
    current_function = 0
    for gen in singleton_generators:
        current_function += distance_to_singleton_normal_closure(word, gen)
        if current_function >= previous_function:
            return False
    current_function += distance_to_singleton_normal_closure(word, joint_generator)
    if current_function >= previous_function:
        return False
    return True


def dist_base(singleton_generators, joint_generator, word):
    return distance_to_singleton_normal_closure(word, joint_generator) + sum(distance_to_singleton_normal_closure(word, gen) for gen in singleton_generators)


def optimize(word, dist, better, p=0.1, generators_number=2, max_length=10, max_iters=10):
    # https://arxiv.org/pdf/1703.03334.pdf 3.2 (1 + 1) EA
    # https://arxiv.org/pdf/1812.11061.pdf 2.2 (\mu + \lambda) EA

    generators = set(range(1, generators_number + 1)) | set(range(-generators_number, 0))
    def mutate(word):
        mutated_word = word.copy()
        for i in range(len(mutated_word)):
            if random.random() < p:
                mutated_word[i] = random.sample(generators - set([mutated_word[i]]), 1)[0]
        return mutated_word

    current_function = dist(word)

    for _ in range(max_iters):
        new_word = mutate(word)
        normalized = normalize(new_word)

        if len(normalized) == 0:
            continue

        if better(normalized, current_function):
            word = new_word.copy()
            current_function = dist(normalized)
            if current_function == 0:
                break

    return normalize(word), current_function == 0


class EvolutionarySampler:
    def __init__(self, generators_number=2, max_length=10, 
                 baseline="free", max_iters=5, 
                 mutation_rate=0.1, exploration_rate=None):
        singleton_generators = [[[x]] for x in range(1, generators_number+1)]
        joint_generator = [[x for x in range(1, generators_number+1)]]

        self.generators_number = generators_number
        self.max_length = max_length
        self.max_iters = max_iters
        self.mutation_rate = mutation_rate
        if exploration_rate is None:
            self.exploration_rate = 1/max_iters
        else:
            self.exploration_rate = exploration_rate

        if baseline == "free":
            self.baseline_group = free_group_bounded(generators_number=generators_number, max_length=max_length)
        elif baseline == "joint":
            self.baseline_group = normal_closure(joint_generator, generators_number=generators_number, max_length=max_length)
        elif baseline == "singleton":
            self.baseline_group = normal_closure(singleton_generators[0], generators_number=generators_number, max_length=max_length)
        else:
            raise NotImplementedError('unknown `baseline`')

        self.dist = partial(dist_base, singleton_generators, joint_generator)
        self.better = partial(better_base, singleton_generators, joint_generator)
        self.condition = lambda word: all(is_from_singleton_normal_closure(gen, word) for gen in singleton_generators) and is_from_singleton_normal_closure(joint_generator, word)

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
                generators_number=self.generators_number, 
                max_length=self.max_length,
                p=self.mutation_rate, max_iters=self.max_iters)
        print('evolution succeed')
        return word


if __name__ == "__main__":
    word = parse_word("x⁻¹z⁻¹y⁻¹zyxy⁻¹x⁻¹z⁻¹yzy⁻¹xy")
    print(len(word))
    print(is_from_singleton_normal_closure([[1]], word), is_from_singleton_normal_closure([[2]], word), is_from_singleton_normal_closure([[3]], word), is_from_singleton_normal_closure([[1, 2, 3]], word))
    print(dist_base([[[1]], [[2]], [[3]]], [[1, 2, 3]], word))

    sampler = EvolutionarySampler(baseline="joint", generators_number=3, max_length=25, max_iters=50, mutation_rate=0.1, exploration_rate=1.)
    for i in range(1):
        start = time.time()
        print_word(next(sampler))
        print(time.time() - start, 's')
