from autograd import numpy as np
from autograd import grad, jacobian
from functools import partial
from group_tool.reduced_words import free_group_bounded, is_from_singleton_normal_closure

dt = np.int32

Sanov_x = np.array(
    [[1, 2],
     [0, 1]], dtype=dt)
Sanov_y = np.array(
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
    return np.array(
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
    return np.array(result)

def get_embedding_map(n, i):
    remove_gen_A = np.eye(2*n+1)
    remove_gen_A[i] *= 0
    remove_gen_A[i][0] = 1
    remove_gen_A[n+i] *= 0
    remove_gen_A[n+i][0] = 1

    embed_A = np.vstack([np.eye(2).flatten()] + [generators_embedding(i).flatten() for i in range(1, n+1)] + [np.linalg.inv(generators_embedding(i)).flatten() for i in range(1, n+1)])

    return np.matmul(remove_gen_A, embed_A)

def distance_from_normal_closure(generators_number, generator, oh_word):
    assert len(generator) == 1
    l = oh_word.shape[0]

    embedding = np.matmul(oh_word, get_embedding_map(generators_number, generator[0]))

    zero_first_row = np.eye(l)
    zero_first_row[0] *= 0
    first_row = np.zeros(l)
    first_row[0] = 1
    left = np.zeros(shape=(l,l))
    left[0][0] = 1

    for s in range(1, l):
        right = np.zeros(shape=(l,l))
        right[s][0] = 1
        embedding = np.matmul(zero_first_row, embedding) + np.matmul(first_row, (np.matmul(np.matmul(np.matmul(np.matmul(left, embedding), Bl0), embedding.T), right)).T)
        # embedding = np.matmul(zero_first_row, embedding) + np.matmul(first_row, np.matmul(np.matmul(np.matmul(right.T, np.matmul(embedding, Bl0.swapaxes(1, 2))), embedding.T), left.T).transpose(1, 2, 0))
        
    embedding = np.matmul(first_row, embedding) - np.array([1, 0, 0, 1])
    return np.linalg.norm(embedding)**2

n = 3
l = 10

gen = free_group_bounded(generators_number=n, max_length=l)
f_pure = lambda oh_word: sum(partial(distance_from_normal_closure, n, [i])(oh_word) for i in range(1, n+1)) / n

penalty = lambda oh_word: np.linalg.norm(oh_word[:, 0])**2 + 1e+1/np.std(np.sum(softmax(x.reshape(l, 2*n+1)) * np.vstack([np.linspace(1, 2*n+1, 2*n+1)]*l), axis=1))

f = lambda oh_word: f_pure(oh_word) + penalty(oh_word)

# for _ in range(100):
#     word = next(gen)
#     oh_word = one_hot(n, word)
#     print("in fact:", is_from_singleton_normal_closure([[1]], word), "we say:", f(oh_word) <= 1e-10)

def softmax(x, beta=10):
    e_x = np.exp(beta * (x - np.matmul(np.expand_dims(np.max(x, axis=1), 1), np.ones(shape=(1, x.shape[1])))))
    return e_x / np.matmul(np.expand_dims(np.sum(e_x, axis=1), 1), np.ones(shape=(1, x.shape[1])))

f_flat_pure = lambda x: f_pure(softmax(x.reshape(l, 2*n+1)))
f_flat = lambda x: f(softmax(x.reshape(l, 2*n+1)))

df = lambda x: grad(f_flat)(x)
d2f = lambda x: jacobian(grad(f_flat))(x)

def pad(word, length):
    return word + [0] * (l - len(word))

x = one_hot(n, pad(next(gen), l)).flatten()

for t in range(100):
    # x = x - np.linalg.inv(d2f(x) + lambd * np.eye(x.shape[0])) @ df(x)
    g = df(x)
    x = x - (g / np.linalg.norm(g)) / np.sqrt(t+1)
    print(f_flat_pure(x))
    word = np.argmax(softmax(x.reshape(l, 2*n+1)), axis=1)
    word[word >= n+1] = n - word[word >= n+1]
    print(word)
