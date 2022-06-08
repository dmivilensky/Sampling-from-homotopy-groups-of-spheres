import random
from group_tool.utils import random_length


def commutators(generators_number=2):
    commutator = [1]
    yield commutator

    while True:
        element_to_increase = len(commutator) - 1
        while element_to_increase > 0 and commutator[element_to_increase] == generators_number:
            element_to_increase -= 1

        if element_to_increase == -1:
            commutator.insert(0, 1)
            raise NotImplementedError()
        else:
            commutator[element_to_increase] += 1
            raise NotImplementedError()

        yield commutator


def random_commutator_power(commutator_index):
    return random.randint(-3, 3)


def free_group_bounded(generators_number=2, max_length=5):
    while True:
        length = random_length(max_length)
        current_length = 0
        word = []

        for i, commutator in enumerate(commutators(generators_number=generators_number)):
            power = random_commutator_power(i)

            if current_length + len(commutator)*power > length:
                break

            word.append(power)
            current_length += len(commutator)*power
        
        if current_length > 0:
            yield word