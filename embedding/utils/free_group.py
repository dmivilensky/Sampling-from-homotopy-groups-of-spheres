import torch
import numpy


def lcp(strs):
    if len(strs) == 0:
        return ""
    current = strs[0]
    for i in range(1, len(strs)):
        temp = ""
        if len(current) == 0:
            break
        for j in range(len(strs[i])):
            if j < len(current) and current[j] == strs[i][j]:
                temp += current[j]
            else:
                break
        current = temp
    return current


def pairwise_distances(sequences):
    with torch.no_grad():
        batch_size = sequences.shape[0]
        result = numpy.zeros(shape=(batch_size, batch_size))
        for i in range(batch_size):
            for j in range(i + 1):
                s1 = "".join(map(lambda x: str(x.item()).strip("0"), sequences[i]))
                s2 = "".join(map(lambda x: str(x.item()).strip("0"), sequences[j]))
                result[i, j] = result[j, i] =\
                    len(s1) + len(s2) - 2 * len(lcp([s1, s2]))
        return torch.Tensor(result)
