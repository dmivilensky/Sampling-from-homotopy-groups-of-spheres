import time
from freegroup.tools import (is_from_singleton_normal_closure, to_string)
from freegroup.sampling import (free_group_bounded, normal_closure_conjugation as normal_closure)


class TrivialSampler:
    def __init__(self, generators_number=2, max_length=10, baseline="free", first=None):
        singleton_generators = [[x] for x in range(1, generators_number+1)]
        joint_generator = [x for x in range(1, generators_number+1)]

        if baseline == "free":
            baseline_group = free_group_bounded(
                generators_number=generators_number, max_length=max_length)
            if not first:
                condition = lambda word: all(
                    is_from_singleton_normal_closure(gen, word) 
                    for gen in singleton_generators) and is_from_singleton_normal_closure(joint_generator, word)
            else:
                condition = lambda word: all(
                    is_from_singleton_normal_closure(gen, word) 
                    for gen in singleton_generators[:first])
        elif baseline == "joint":
            baseline_group = normal_closure(joint_generator, 
                generators_number=generators_number, max_length=max_length)
            condition = lambda word: all(
                is_from_singleton_normal_closure(gen, word) 
                for gen in singleton_generators)
            if first:
                raise NotImplementedError('`first` disagrees with joint baseline')
        elif baseline == "singleton":
            baseline_group = normal_closure(singleton_generators[0], 
                generators_number=generators_number, max_length=max_length)
            if not first:
                condition = lambda word: all(
                    is_from_singleton_normal_closure(gen, word) 
                    for gen in singleton_generators[1:]) and is_from_singleton_normal_closure(joint_generator, word)
            else:
                condition = lambda word: all(
                    is_from_singleton_normal_closure(gen, word) 
                    for gen in singleton_generators[1:first])
        else:
            raise NotImplementedError('unknown `baseline`')

        self.intersection = filter(condition, baseline_group)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.intersection)


if __name__ == "__main__":
    sampler = TrivialSampler(baseline="joint", generators_number=2, max_length=25)
    start = time.time()
    for i in range(1000):
        print(to_string(next(sampler), method='su'))
    print(time.time() - start, 's')
