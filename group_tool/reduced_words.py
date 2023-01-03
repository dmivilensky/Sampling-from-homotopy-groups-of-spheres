from typing import List, Iterable
from numpy import array, pad


Word = List[int]


LETTERS = "xyzpqrstuvwklmn"

def print_word(word, verbose=True):
    result = []
    for factor in word:
        if type(factor) is list:
            result.append("[" + ",".join(print_word(factor, verbose=False)) + "]")
        else:
            result.append(LETTERS[abs(factor) - 1] + ("⁻¹" if factor < 0 else ""))
    
    if verbose:
        print("".join(result))
    return result


def print_words(words, verbose=True):
    for word in words:
        yield print_word(word, verbose)


def parse_word(string, order=None):
    letters = LETTERS[:order]
    i = 0
    word = []
    while i < len(string):
        if string[i] != "⁻":
            word.append(letters.index(string[i]) + 1)
            i += 1
        else:
            word[-1] = -word[-1]
            i += 2
    return word


def to_numpy(words: Iterable[Word]):
    words = list(words)
    max_length = max(map(len, words))
    return array(list(map(lambda v: pad(v, (0, max_length - len(v))), words)))


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
