import math
import random


def random_length(radius):
    # https://arxiv.org/pdf/1805.08207.pdf 6.3 Uniform sampling in hyperbolic space
    return max(1, int(round(math.acosh(1 + random.random() * (math.cosh(radius) - 1)))))


def free_group_bounded(generators_number=2, max_length=5):
    generators = set(range(1, generators_number + 1)) | set(range(-generators_number, 0))

    while True:
        length = random_length(max_length)
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


def normal_closure(subgroup, generators_number=2, max_length=5):
    while True:
        length = random_length(max_length)
        word = []

        while len(word) < length:
            factor = random.sample(subgroup, 1)[0]
            if random.random() > 0.5:
                factor = reciprocal(factor)

            conjugator = next(free_group_bounded(
                generators_number=generators_number, 
                max_length=(length - len(word) - len(factor)) // 2
            ))
            word += conjugation(factor, conjugator)
            word = normalize(word)

        yield word


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


def is_cyclic_permutation(a, b):
    if len(a) != len(b):
        return False

    double_b = b * 2
    for i in range(2 * len(b)):
        if double_b[i] == a[0] and double_b[i:i + len(a)] == a:
            return True
    return False


def is_from_normal_closure(generator, word):
    contained_smth_to_reduce = True
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
    
    return len(word) == 0


def print_word(word):
    letters = "xyzpqrstuvwklmn"
    print("".join(map(lambda factor: letters[abs(factor) - 1] + ("⁻¹" if factor < 0 else ""), word)))


def print_words(words):
    for word in words:
        print_word(word)
