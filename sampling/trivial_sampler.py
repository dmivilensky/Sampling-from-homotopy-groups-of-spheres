import time
from group_tool.reduced_words import *
from group_tool.utils import print_word, parse_word


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
        # Only for tests!
        elif baseline == "first_one":
            baseline_group = normal_closure(singleton_generators[0], 
                generators_number=generators_number, max_length=max_length)
            condition = lambda _: True
        # Only for tests!
        elif baseline == "first_two":
            baseline_group = normal_closure(singleton_generators[0], 
                generators_number=generators_number, max_length=max_length)
            condition = lambda word: is_from_singleton_normal_closure(singleton_generators[1], word) 
        # Only for tests!
        elif baseline == "first_three":
            baseline_group = normal_closure(singleton_generators[0], 
                generators_number=generators_number, max_length=max_length)
            condition = lambda word: is_from_singleton_normal_closure(singleton_generators[1], word) and is_from_singleton_normal_closure(singleton_generators[2], word) 
        else:
            raise NotImplementedError('unknown `baseline`')

        self.intersection = filter(condition, baseline_group)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.intersection)


if __name__ == "__main__":
    # word = [-2, 3, 2, -3, -2, -1, -2, 1, 2, 2, 1, 1, 1, -2, -2, -1, 2, 1, 2, 3, -2, -3, 2, -1, -1, -1, 3, -1, -2, -1, 3, 3, 1, 2, 1, -3, -3, 1, 1, 1, -2, 3, 2, -3, -2, -1, -2, 1, 2, 2, -1, -1, -1, -2, -2, -1, 2, 1, 2, 3, -2, -3, 2, 3, 3, -1, -2, -1, -3, -3, 1, 2, 1, -3]
    # print(len(word))
    # print(
    #     is_from_singleton_normal_closure([[1]], word), 
    #     is_from_singleton_normal_closure([[2]], word), 
    #     is_from_singleton_normal_closure([[3]], word), 
    #     is_from_singleton_normal_closure([[1, 2, 3]], word))
    # word = parse_word("xy⁻¹xy⁻¹xy⁻¹xy⁻¹xyyx⁻¹yx⁻¹yx⁻¹x⁻¹x⁻¹x⁻¹y⁻¹y⁻¹xyy")
    # print(
    #     is_from_singleton_normal_closure([[1]], word), 
    #     is_from_singleton_normal_closure([[2]], word),
    #     is_from_singleton_normal_closure([[1, 2]], word))
    # quit()

    sampler = TrivialSampler(baseline="joint", generators_number=2, max_length=25)
    start = time.time()
    for i in range(1000):
        print_word(next(sampler))
    print(time.time() - start, 's')
