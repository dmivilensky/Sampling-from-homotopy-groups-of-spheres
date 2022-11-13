import numpy as np
import matplotlib.pyplot as plt
from group_tool.reduced_words import free_group_bounded

plt.figure(figsize=(10, 8))
for dt in [np.int16, np.int32, np.int64, np.float32]:
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

    # assert np.allclose(commutators_embedding(10, 5), commutators_embedding(10, 5, straightforward=True))

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

    x_values = []
    y_values = []

    if dt in [np.float16, np.float32]:
        sz = np.finfo(dt).max
    else:
        sz = np.iinfo(dt).max

    for n in range(2, 100):
        i = 2
        while True:
            gen = free_group_bounded(generators_number=n, max_length=i)
            max_element = 0
            for attempt in range(500):
                max_element = max(max_element, np.max(word_embedding(next(gen))))
                if max_element > sz:
                    break
            if max_element > sz:
                break
            i += 1
        print("maximal length for n = %d: %d" % (n, i-1))
        x_values.append(n)
        y_values.append(i-1)

    plt.plot(x_values, np.minimum.accumulate(y_values), label="type: " + str(dt) + ", limit: %e" % sz + ", maximal n: %e" % np.floor(np.roots([16, 4, 1 - sz]).max()))

plt.xlabel("generators number $n$")
plt.ylabel("maximal length $l$")
plt.title("limits")
plt.grid(alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("limits.pdf")
plt.show()