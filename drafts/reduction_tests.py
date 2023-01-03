import numpy as np
from freegroup.tools import free_group_bounded

dt = np.int32

Sanov_x = np.matrix(
    [[1, 2],
     [0, 1]], dtype=dt)
Sanov_y = np.matrix(
    [[1, 0],
     [2, 1]], dtype=dt)

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

def word_embedding(word):
    result = np.matrix(
        [[1, 0],
         [0, 1]], dtype=dt)
    for letter in word:
        letter_embedding = generators_embedding(abs(letter))
        if letter < 0:
            letter_embedding = np.linalg.inv(letter_embedding)
        result = result @ letter_embedding
    return result

def one_hot(generators_number, word):
    result = []
    for letter in word:
        line = np.zeros(shape=2*generators_number+1)
        line[abs(letter) + (generators_number if letter < 0 else 0)] = 1
        result.append(line)
    return np.matrix(result)

n = 3
i = 1
gen = free_group_bounded(generators_number=n, max_length=5)
word = next(gen)
l = len(word)

remove_gen_A = np.eye(2*n+1)
remove_gen_A[i] *= 0
remove_gen_A[i][0] = 1
remove_gen_A[n+i] *= 0
remove_gen_A[n+i][0] = 1

embed_A = np.vstack([np.eye(2).flatten()] + [generators_embedding(i).flatten() for i in range(1, n+1)] + [np.linalg.inv(generators_embedding(i)).flatten() for i in range(1, n+1)])

embedding_map = remove_gen_A @ embed_A

oh_word = one_hot(n, word)
embedding = np.array(oh_word @ embedding_map)
print(embedding)

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

a = Sanov_x.flatten()
b = Sanov_y.flatten().T
assert np.allclose(a @ Bl0 @ b, (Sanov_x @ Sanov_y).flatten().T)

a = np.random.random((2, 2)).flatten()
b = np.random.random((2, 2)).flatten().T
assert np.allclose(a @ Bl0 @ b, (a.reshape(2,2) @ b.T.reshape(2,2)).flatten().T)

zero_first_row = np.eye(l)
zero_first_row[0] *= 0
first_row = np.zeros(l)
first_row[0] = 1
left = np.zeros(shape=(l,l))
left[0][0] = 1

real_product = np.eye(2)
for i, m in enumerate(embedding):
    real_product = real_product @ m.reshape(2, 2)
print("real:", real_product.flatten())

for s in range(1, l):
    right = np.zeros(shape=(l,l))
    right[s][0] = 1
    embedding = zero_first_row @ embedding + first_row @ (left @ embedding @ Bl0 @ embedding.T @ right).T

print(first_row @ embedding)
