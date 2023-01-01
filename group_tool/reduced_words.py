import random 
import group_tool.utils as utils


def free_group_bounded(generators_number=2, max_length=5):
    generators = set(range(1, generators_number + 1)) | set(range(-generators_number, 0))

    while True:
        length = utils.random_length(max_length)
        word = [random.sample(generators, 1)[0]]

        for _ in range(length-1):
            factor = random.sample(generators - set([-word[-1]]), 1)[0]
            word.append(factor)

        yield word


def reciprocal(word):
    return [-factor for factor in word[::-1]]


def conjugation(word, conjugator):
    inverted_conjugator = reciprocal(conjugator)

    i = 0
    while i < min(len(inverted_conjugator), len(word)) and inverted_conjugator[-(i+1)] + word[i] == 0:
        i += 1

    j = 0
    while j < min(len(word), len(conjugator)) and word[-(j+1)] + conjugator[j] == 0:
        j += 1
    
    return inverted_conjugator[:(-i if i != 0 else len(inverted_conjugator)+1)] + word[i:(-j if j != 0 else len(word)+1)] + conjugator[j:]


def commutator(x, y):
    return reciprocal(x) + reciprocal(y) + x + y


def multiply(x, y): 
    return x + y


def normalize(word):
    normalized = []

    for factor in word:
        if factor == 0:
            continue
        if len(normalized) == 0:
            normalized.append(factor)
            continue

        if factor == -normalized[-1]:
            normalized.pop()
        else:
            normalized.append(factor)

    return normalized


def normal_closure(subgroup, generators_number=2, max_length=5):
    while True:
        length = utils.random_length(max_length)
        word = []

        while True:
            factor = random.sample(subgroup, 1)[0]
            if random.random() > 0.5:
                factor = reciprocal(factor)

            conjugator = next(free_group_bounded(
                generators_number=generators_number, 
                max_length=(length - len(word) - len(factor)) // 2
            ))
            new_word = word + conjugation(factor, conjugator)
            new_word = normalize(new_word)
            if len(new_word) > max_length:
                break
            word = new_word

        yield word


def symmetric_commutant(generators_number=2, max_length=5):
    closures =\
        [normal_closure([[i]], generators_number, max_length) for i in range(1, generators_number + 1)] +\
        [normal_closure([list(range(1, generators_number + 1))], generators_number, max_length)]
    yield from filter(lambda x: len(x) > 0, 
        map(normalize, 
        utils.reduce(commutator,
        utils.shuffle(
        zip(*closures)
    ))))


def occurs(a, b):
    len_a = len(a)
    for i in range(len(b) - len_a):
        if b[i] == a[0] and b[i:i + len_a] == a:
            return True
    return False


def is_from_singleton_normal_closure(generators, word):
    if len(generators) != 1:
        raise NotImplementedError('`generators` must contain only one generator ;)')

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
    
    return len(word) == 0
