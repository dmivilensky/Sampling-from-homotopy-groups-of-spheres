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
All samplers have `_generator` version for infinite iterable
This module helps to build **word** samplers for generating datasets
- `random_length(method = Either ['uniform', 'uniform_radius', 'constant'] or custom_function, **params)`. Returns a number from the given distribution. One can pass custom distribuiton in `method` parameter.
- `freegroup(freegroup_dimension, length_method, length_parameters)`. Infinite generator of non-reducible words from free group on `freegroup_dimension` **generator**s
- `normal_closure(method = ['conjugation', 'brackets'], closure, freegroup_dimension, **params)`. Random word from the normal closure `<closure>`
  ```py
  from freegroup.sampling import normal_closure
  generator = normal_closure('brackets', [1], 4, depth_method = 'uniform', depth_parameters = {'radius': 10})  
  ```
  `generator` will produce **word**s from `<x>` with uniform length from 2 to 20
