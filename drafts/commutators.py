from freegroup.tools import normalize, print_word, parse_word


def factorize(w):
    if len(w) == 2:
        return w
    if len(w) == 1:
        return w[0]
    i = 1
    while i < len(w) - 1:
        if print_word(w[:i], verbose=False) < print_word(w[i:], verbose=False):
            break
    return [factorize(w[:i]), factorize(w[i:])]


if __name__ == "__main__":
    # Python implementation of
    # the above approach
    
    maxlen = 5
    generators = 2

    S = list(range(1, generators+1))
    k = len(S)
    
    # To store the indices
    # of the characters
    w = [-1]
        
    # Loop till w is not empty
    while w:
        
        # Incrementing the last character
        w[-1] += 1
        m = len(w)
        print(factorize(w))
        # print_word(map(lambda i: S[i], w))
            
        # Repeating w to get a
        # n-length string
        while len(w) < maxlen:
            w.append(w[-m])
            
        # Removing the last character
        # as long it is equal to
        # the largest character in S
        while w and w[-1] == k - 1:
            w.pop()
