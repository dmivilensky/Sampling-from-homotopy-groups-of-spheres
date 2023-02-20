# Definitions

- **generator** is a free group *letter*, word of length 1, i. e. `1` is a free group word `x` and `-1` is a free group word `X`
- **word** is a `list` of either **generator** or **commutator**, i.e. `list(1, -2, 3)` is a free group word `xYz`
- iterable **commutator** is a `tuple` of **word**s, i. e. `tuple(x, y, z)` is a free group commutator `[[x, y], z]`

# `freegroup.tools`
All methods have `batch_` version that accepts list of `word`s
- method `reciprocal(word)`. Intverts the given **word**.
  ```py
  from freegroup.tools import reciprocal
  assert reciprocal([(1, 2), 4, 3]) == [-3, -4, (2, 1)]
  ```
- method `flatten(word)`. Removes **commutator**s from the given **word**
  ```py
  from freegroup.tools import flatten
  assert flatten([(1, 2), 4, 3]) == [-1, -2, 1, 2, 4, 3]
  ```
- method `conjugate(word, conjugator)`. Conjugate the given **word** with a conjugator, which is also a **word**
  ```py
  from freegroup.tools import conjugate
  assert conjugate((1, 2), 1) == ([-1, 1, 1], [-1, 2, 1])
  assert conjugate([1, 2], [1]) == [[-1], 1, 2, [1]]
  ```
 - method `normalize(word)`. Normalizes the given **word**, i. e. reduces `i` and `-i`, make words like `[[1], 2, 3]` flat, ...
    ```py
    from freegroup.tools import normalize
    assert normalize([[-1], 1, 2, [1]]) == [2, 1]
    ```
 - method `reduce_modulo_singleton_normal_closure(word, closure)`. Reduces and removes all trivial words by modulo of `closure`, which is a `list` of **generator**s
    ```py
    from freegroup.tools import reduce_modulo_singleton_normal_closure
    assert reduce_modulo_singleton_normal_closure([2, 1, 1, 2, 3, -2, 2, 3, 1, 1], [1, 2, 3]) == [2, 1, -2, 1]
    ```
- method `is_from_singleton_normal_closure(word, closure)`. Checks wether the given word is from normal closure.
  ```py
  from freegroup.tools import is_from_singleton_normal_closure
  assert is_from_singleton_normal_closure([-3, 1, -2, -1, 2, 3], [1]) == True
  ```
- method `to_string(word, method = Either 'lu', 'integer', 'superscript')`. Converts the given word to a string using the given `method`.
  - 'lu'. The number `i` equals to a lowercase latin letter and `-i` equals to uppercase latin letter
  - 'integer'. The number `i` equals to a string `i`
  - `superscript`. The number `i` equals to a lowercase latin letter and `-i` equals to the same lowercase latin letter with a superscript -1
- method `from_string(string, method = Either 'lu', 'integer', 'superscript')`. Converts the given string to a **word** using the given `method`

# `freegroup.sampling`
This module helps to build **word** samplers for generating datasets
- `random_length(radius, method = Either 'uniform', 'uniform_radius', 'constant', custom_function)`. Returns a number from the given distribution. One can pass custom distribuiton in `method` parameter.
- `freegroup_bounded(freegroup_dimension, max_length, random_length_method)`. Infinite generator of non-reducible words from free group on `freegroup_dimension` **generator**s
- `normal_closure(closure, freegroup_dimension, method = Either 'conjugation' or 'brackets', params = Either max_length or max_depth, random_depth_method)`. Infinite generator of words from the normal closure `<closure>`
  ```py
  from freegroup.sampling import normal_closure
  generator = normal_closure([1], 4, method = 'brackets', max_depth = 10, max_depth_method = 'uniform_radius')  
  ```
  `generator` will produce **word**s from `<x>` with uniform length from 2 to 20
 - `random_order_commutant(words)`. Accepts `list` of Infinite Generatots. Returns Infinite Generator of random order commutators.
  ```py
  from freegroup.sampling import freegroup_bounded, random_order_commutant
  iterables = [it1, it2, it3, it4,]
  iterable = random_order_commutant(zip(*iterables))
  ```
  `iterable` will produce expressions like `[[next(it1), next(it2)], [next(it1), next(it2)]]` or `[[[next(it1), next(it2)], next(it3)], next(it4)]`, ...
  
  ## Example dataset
  ```py
  from freegroup.sampling import freegroup_bounded, random_order_commutant
  from iteration_utilities import unique_everseen
  from itertools import islice
  from tqdm import tqdm
  from random import randint, sample
  
  def iterable():
    commutee = freegroup_bounded(4, radius = 10, method = 'uniform_radius')

    def leaves():
      while True: yield [next(commuteee) for _ in range(randint(2, 5))]

    iterable = leaves()
    # iterable = map(lambda x: sample(x, len(x)), iterable), if you want random permutation of `leaves`
    iterable = random_order_commutant(iterable)
    return unique_everseen(iterable, key = tuple)
  
  size = int(10 ** 2)
  dataset = tqdm(islice(iterable(), size), total = size) # You can omit tqdm, if you don't want output anything
  ```
