import math

def is_prime(p):
    if p == 2:
        return True
    for i in range(2, int(math.sqrt(p))+1):
        if p % i == 0:
            return False
    return True

def is_power_of_prime(p):
    number = 1
    power = 0
    for i in range(2, p+1):
        if is_prime(i) and p % i == 0:
            while number < p:
                number *= i
                power += 1
            if number == p:
                return i, power
            else:
                return False, False

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
            for i in range(1, n+1):
                yield [("lambda", i)]
        else:
            ranges = range(math.floor((p**(i-1)-1)/2) + 1 if wave else 1, n+1) if start_from is None else range(math.floor((p**(i-1)-1)/2) + 1 if wave else 1, start_from)
            for symbol in ["mu", "lambda"]:
                for first in ranges:
                    for tail in allowable_set(first, k-1, p=p, start_from=p*first + (1 if symbol == "mu" else 0), i=2):
                        yield [(symbol, first)] + tail

def allowable_set_filtration(n, k, p, j):
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

def derived_functor_dimension(i, p, k, n):
    return len(derived_functor_generators(i, p, k, n))

def derived_functor_generators(i, p, k, n):
    if p == 2:
        return list(filter(lambda seq: i == 2*n + sum(seq), allowable_set(n, k, p=2)))
    else:
        return list(filter(lambda seq: i == 2*n + (2*p - 2)*sum(map(lambda pair: pair[1], seq)) - len(list(filter(lambda pair: pair[0] == "lambda", seq))), allowable_set(n, k, p=p)))

if __name__ == "__main__":
    for i in [2, 3]:
        for s in range(2, 9):
            p, k = is_power_of_prime(s)
            if p:
                dim = derived_functor_dimension(i, p, k, 1)
                if dim == 0:
                    print(f"L_{i} Lie^({p}^{k}) (Z, 2) = 0")
                elif dim == 1:
                    print(f"L_{i} Lie^({p}^{k}) (Z, 2) = Z/{p}")
                else:
                    print(f"L_{i} Lie^({p}^{k}) (Z, 2) = (Z/{p})^{dim}")
