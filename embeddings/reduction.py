import numpy as np
from group_tool.reduced_words import free_group_bounded, is_from_singleton_normal_closure

dt = np.int32

Sanov_x = np.matrix(
    [[1, 2],
     [0, 1]], dtype=dt)
Sanov_y = np.matrix(
    [[1, 0],
     [2, 1]], dtype=dt)

Bl0 = np.array(
    [[[1, 0, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0]],
     [[0, 1, 0, 0],
      [0, 0, 0, 1],
      [0, 0, 0, 0],
      [0, 0, 0, 0]],
     [[0, 0, 0, 0],
      [0, 0, 0, 0],
      [1, 0, 0, 0],
      [0, 0, 1, 0]],
     [[0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 0, 1]]])

def commutators_embedding(x_power, y_power, straightforward=False):
    if straightforward:
        return np.linalg.matrix_power(np.linalg.inv(Sanov_x), x_power) @\
               np.linalg.matrix_power(np.linalg.inv(Sanov_y), y_power) @\
               np.linalg.matrix_power(Sanov_x, x_power) @\
               np.linalg.matrix_power(Sanov_y, y_power)
    return np.matrix(
        [[16*(x_power**2)*(y_power**2) + 4*x_power*y_power + 1, 8*(x_power**2)*y_power],
         [-8*x_power*(y_power**2), 1-4*x_power*y_power]], dtype=dt)

assert np.allclose(commutators_embedding(10, 5), commutators_embedding(10, 5, straightforward=True))

def generators_embedding(i):
    assert 1 <= i
    return commutators_embedding(1, i)

def one_hot(generators_number, word):
    result = []
    for letter in word:
        line = np.zeros(shape=2*generators_number+1)
        line[abs(letter) + (generators_number if letter < 0 else 0)] = 1
        result.append(line)
    return np.matrix(result)

def get_embedding_map(n, i):
    remove_gen_A = np.eye(2*n+1)
    remove_gen_A[i] *= 0
    remove_gen_A[i][0] = 1
    remove_gen_A[n+i] *= 0
    remove_gen_A[n+i][0] = 1

    embed_A = np.vstack([np.eye(2).flatten()] + [generators_embedding(i).flatten() for i in range(1, n+1)] + [np.linalg.inv(generators_embedding(i)).flatten() for i in range(1, n+1)])

    return remove_gen_A @ embed_A

def distance_from_normal_closure(generators_number, generator, oh_word):
    assert len(generator) == 1
    l = oh_word.shape[0]

    embedding = np.array(oh_word @ get_embedding_map(generators_number, generator[0]))

    zero_first_row = np.eye(l)
    zero_first_row[0] *= 0
    first_row = np.zeros(l)
    first_row[0] = 1
    left = np.zeros(shape=(l,l))
    left[0][0] = 1

    for s in range(1, l):
        right = np.zeros(shape=(l,l))
        right[s][0] = 1
        embedding = zero_first_row @ embedding + first_row @ (left @ embedding @ Bl0 @ embedding.T @ right).T
    
    embedding = first_row @ embedding - np.array([1, 0, 0, 1])
    return np.linalg.norm(embedding)

n = 3
gen = free_group_bounded(generators_number=n, max_length=5)

for _ in range(100):
    word = next(gen)
    oh_word = one_hot(n, word)
    print("in fact:", is_from_singleton_normal_closure([[1]], word), "we say:", distance_from_normal_closure(n, [1], oh_word) <= 1e-10)
