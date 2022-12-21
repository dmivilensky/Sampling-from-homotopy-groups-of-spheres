import math

def is_prime(p):
    for i in range(2, int(math.sqrt(p))+1):
        if p % i == 0:
            return False
    return True

def allowable_set(n, k, p=2, start_from=None, i=1, wave=False):
    if p == 2:
        if k == 1:
            for i in range(2**(i-1) if wave else 1, n+1):
                yield [2*i - 1]
        else:
            for first in range(2**(i-1) if wave else 1, 2*n+1):
                for tail in allowable_set(first, k-1, i=2):
                    yield [first] + tail
    else:
        if k == 1:
            for i in range(1, start_from):
                yield [("lambda", i)]
        else:
            ranges = range(math.floor((p**(i-1)-1)/2) + 1 if wave else 1, n+1) if start_from is None else range(math.floor((p**(i-1)-1)/2) + 1 if wave else 1, start_from)
            for symbol in ["mu", "lambda"]:
                for first in ranges:
                    for tail in allowable_set(first, k-1, p=p, start_from=p*first + (1 if symbol == "mu" else 0), i=2):
                        yield [(symbol, first)] + tail

def allowable_set_filtration(n, k, p, j):
    if is_prime(p):
        if p == 2:
            if j == 1:
                yield from allowable_set(n, k)
            else:
                head = [2**i * n for i in range(1, j)]
                for tail in allowable_set(head[-1], k-j+1, i=j):
                    yield head + tail
        else:
            if j == 1:
                yield from allowable_set(n, k, p=p)
            else:
                head = [("mu", p**(i-1) * n) for i in range(1, j)]
                for tail in allowable_set(n, k-j+1, p=p, start_from=p*head[-1][1]+1, i=j):
                    yield head + tail
    else:
        pass

for seq in allowable_set_filtration(2, 2, 3, 2):
    print(seq)
