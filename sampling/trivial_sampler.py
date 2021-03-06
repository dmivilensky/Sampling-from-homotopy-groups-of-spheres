import time
from group_tool.reduced_words import *
from group_tool import print_word


class TrivialSampler:
    def __init__(self, generators_number=2, max_length=10, baseline="free"):
        singleton_generators = [[[x]] for x in range(1, generators_number+1)]
        joint_generator = [[x for x in range(1, generators_number+1)]]

        if baseline == "free":
            baseline_group = free_group_bounded(
                generators_number=generators_number, max_length=max_length)
            condition = lambda word: all(
                is_from_singleton_normal_closure(gen, word) 
                for gen in singleton_generators) and is_from_singleton_normal_closure(joint_generator, word)
        elif baseline == "joint":
            baseline_group = normal_closure(joint_generator, 
                generators_number=generators_number, max_length=max_length)
            condition = lambda word: all(
                is_from_singleton_normal_closure(gen, word) 
                for gen in singleton_generators)
        elif baseline == "singleton":
            baseline_group = normal_closure(singleton_generators[0], 
                generators_number=generators_number, max_length=max_length)
            condition = lambda word: all(
                is_from_singleton_normal_closure(gen, word) 
                for gen in singleton_generators[1:]) and is_from_singleton_normal_closure(joint_generator, word)
        else:
            raise NotImplementedError('unknown `baseline`')

        self.intersection = filter(condition, baseline_group)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.intersection)


if __name__ == "__main__":
    word = [-2, 3, 2, -3, -2, -1, -2, 1, 2, 2, 1, 1, 1, -2, -2, -1, 2, 1, 2, 3, -2, -3, 2, -1, -1, -1, 3, -1, -2, -1, 3, 3, 1, 2, 1, -3, -3, 1, 1, 1, -2, 3, 2, -3, -2, -1, -2, 1, 2, 2, -1, -1, -1, -2, -2, -1, 2, 1, 2, 3, -2, -3, 2, 3, 3, -1, -2, -1, -3, -3, 1, 2, 1, -3]
    print(len(word))
    print(
        is_from_singleton_normal_closure([[1]], word), 
        is_from_singleton_normal_closure([[2]], word), 
        is_from_singleton_normal_closure([[3]], word), 
        is_from_singleton_normal_closure([[1, 2, 3]], word))

    sampler = TrivialSampler(baseline="joint", generators_number=3, max_length=25)
    for i in range(1):
        start = time.time()
        print_word(next(sampler))
        print(time.time() - start, 's')
