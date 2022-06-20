from group_tool.reduced_words import normalize
from group_tool.utils import print_word, parse_word


def commutator(x, y):
    if type(x) is int:
        return [x, y]
    else:
        return x + [y]


def unpack(x):
    if type(x) is not int and len(x) == 1:
        return x[0]
    else:
        return x


def weight(x):
    if type(x) is int:
        return 1
    else:
        return weight(unpack(x[:-1])) + weight(x[-1])


def compare(x, y):
    wx, wy = weight(x), weight(y)
    if wx != wy:
        return 1 if wx > wy else -1
    elif wx == 1:
        return 1 if x > y else -1
    else:
        c1 = compare(unpack(x[:-1]), unpack(y[:-1]))
        return compare(x[-1], y[-1]) if c1 == 0 else c1


def hall(word):
    w = normalize(word)
    i = 0
    while i < len(w) - 1:
        j = 0
        changed = False
        while j < len(w) - i - 1:
            y = w[j]
            x = w[j+1]
            if compare(y, x) == 1:
                changed = True
                w[j], w[j+1] = x, y
                w.insert(j+2, commutator(y, x))
                j += 2
            else:
                j += 1
        i += 1
        if not changed:
            break
        print_word(w)
    return w


if __name__ == "__main__":
    word = parse_word("zyx")
    # word = parse_word("x⁻¹z⁻¹y⁻¹zyxy⁻¹x⁻¹z⁻¹yzy⁻¹xy")
    print_word(word)
    hall(word)