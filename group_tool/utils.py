import math
import random


def random_length(radius, method="uniform"):
    if method == "uniform":
        # https://arxiv.org/pdf/1805.08207.pdf 6.3 Uniform sampling in hyperbolic space
        return max(1, int(round(math.acosh(1 + random.random() * (math.cosh(radius) - 1)))))
    elif method == "almost_uniform":
        return max(1, int(round(math.asinh(random.random() * math.cosh(radius - 1)))))
    elif method == "uniform_radius":
        return max(1, int(round(random.random() * radius)))


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
