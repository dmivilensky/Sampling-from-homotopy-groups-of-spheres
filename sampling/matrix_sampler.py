import time
from autograd import numpy as np
from autograd import grad
from functools import partial
from freegroup.tools import is_from_singleton_normal_closure, normalize, print_word


dot_product = np.array(
    [[[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]],
     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]], dtype=np.float64)


def softmax(x, beta=10):
    e_x = np.exp(beta * (x - np.matmul(np.expand_dims(np.max(x, axis=1), 1), np.ones(shape=(1, x.shape[1])))))
    return e_x / np.matmul(np.expand_dims(np.sum(e_x, axis=1), 1), np.ones(shape=(1, x.shape[1])))


def one_hot(generators_number, word):
    result = []
    for letter in word:
        line = np.zeros(shape=2*generators_number+1)
        line[abs(letter) + (generators_number if letter < 0 else 0)] = 1
        result.append(line)
    return np.array(result)


class MatrixSampler:
    def __init__(
        self, generators_number=2, max_length=10, 
        beta_1=500, beta_2=500, eps=1e-80, maxiter=400, minimum_upper_bound=1, 
        dtype=np.float64, first=None, verbose=False):
        
        self.eps = eps
        self.dtype = dtype
        self.beta_2 = beta_2
        self.maxiter = maxiter
        self.minimum_upper_bound = minimum_upper_bound

        self.max_length = max_length
        self.generators_number = generators_number

        self.Sanov_x = np.array([[1, 2], [0, 1]], dtype=dtype)
        self.Sanov_y = np.array([[1, 0], [2, 1]], dtype=dtype)
        
        if not first:
            first = generators_number

        self.f_pure = lambda x: sum(
                partial(MatrixSampler.distance_from_normal_closure, generators_number, [i])(softmax(x.reshape(max_length, 2*generators_number+1), beta_1))
                for i in range(1, min(first, generators_number)+1)) / min(first, generators_number)
        self.penalty = lambda x: 1/MatrixSampler.distance_from_normal_closure(generators_number, [], softmax(x.reshape(max_length, 2*generators_number+1), beta_2))
        self.f = lambda x: self.f_pure(x) + self.penalty(x)

        self.first = first
        self.verbose = verbose

        self.gen = self.sample_word()

    @staticmethod
    def generators_embedding(i, dtype=np.float64):
        return np.array([[16*(i**2) + 4*i + 1, 8*i], [-8*(i**2), 1-4*i]], dtype=dtype)

    @staticmethod
    def get_substitute_and_embed(generators_number, i):
        embed = np.vstack([np.eye(2).flatten()] +\
            [MatrixSampler.generators_embedding(j).flatten() for j in range(1, generators_number+1)] +\
            [np.linalg.inv(MatrixSampler.generators_embedding(j)).flatten() for j in range(1, generators_number+1)])

        if i is None:
            return embed

        remove_generator = np.eye(2*generators_number+1)
        remove_generator[i] *= 0
        remove_generator[i][0] = 1
        remove_generator[generators_number+i] *= 0
        remove_generator[generators_number+i][0] = 1

        return np.matmul(remove_generator, embed)

    @staticmethod
    def distance_from_normal_closure(generators_number, generator, one_hot, dtype=np.float64, distance=True):
        l = one_hot.shape[0]

        if len(generator) <= 1:
            embedding = np.matmul(one_hot, MatrixSampler.get_substitute_and_embed(generators_number, generator[0] if len(generator) == 1 else None))
        else:
            raise NotImplementedError('`generators` must contain no more than one generator ;)')

        zero_first_row = np.eye(l)
        zero_first_row[0] *= 0
        first_row = np.zeros(l)
        first_row[0] = 1
        left = np.zeros(shape=(l,l))
        left[0][0] = 1

        for s in range(1, l):
            right = np.zeros(shape=(l,l))
            right[s][0] = 1
            embedding = np.matmul(zero_first_row, embedding) + np.matmul(first_row, (np.matmul(np.matmul(np.matmul(np.matmul(left, embedding), dot_product), embedding.T), right)).T)
            
        if distance:
            embedding = np.matmul(first_row, embedding) - np.array([1, 0, 0, 1], dtype=dtype)
            return np.linalg.norm(embedding)**2
        else:
            return np.matmul(first_row, embedding)

    def sample_word(self):
        while True:
            x = np.random.random(self.max_length*(2*self.generators_number+1))
            for _ in range(self.maxiter):
                g = grad(self.f)(x)
                g_norm = np.linalg.norm(g)
                x = x - g / max(g_norm, self.eps)
                f_val = self.f_pure(x)

                if g_norm < self.eps and f_val > self.minimum_upper_bound:
                    if self.verbose:
                        print("poor local minimum")
                    break

                word = np.argmax(softmax(x.reshape(self.max_length, 2*self.generators_number+1), self.beta_2), axis=1)
                word[word >= self.generators_number+1] = self.generators_number - word[word >= self.generators_number+1]

                if all(is_from_singleton_normal_closure([[i]], word) for i in range(1, min(self.first, self.generators_number)+1)):
                    if len(normalize(word)) > 0:
                        if self.verbose:
                            print("%.2e %.2e %.2e" % (f_val, self.penalty(x), g_norm))
                        yield normalize(word)
                        break
                    elif self.verbose:
                        print("degenerate word generated")

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.gen)


if __name__ == "__main__":
    sampler = MatrixSampler(generators_number=2, max_length=25)
    start = time.time()
    for i in range(1000):
        print_word(next(sampler))
    print(time.time() - start, 's')
