# Sampling from $\pi_n(S^2)$. Application of optimization and machine learning methods to problems of algebraic topology

## Problem statement

There is Wu formula [1] for the homotopy groups of the two-dimensional sphere:
$$\pi_n(S^2) = \frac{R_0 \cap ... \cap R_{n-1}}{[[R_0, ..., R_{n-1}]]},$$
where $R_i = \langle x_i \rangle \subset F$ is a subgroup of free group $F$ generated by $x_i$ ( $i=1, ..., n-1$ ), $R_0 = \langle x_1 x_2 ... x_{n-1} \rangle \subset F$, and $[[R_0, ..., R_{n-1}]] = \Pi_{\pi \in S_n} [R_{\pi(0)}, ..., R_{\pi(n-1)}]$ is a symmetric commutant. Following this formula, we're trying to solve the problem of sampling elements from homotopy group, represented by some elements of free group $F$, which can be expressed by words in the alphabet $\{ x_1, ..., x_{n-1}, x_1^{-1}, ..., x_{n-1}^{-1} \}$, pretty comfortable object to apply computational algorithms. At the same time, sampling elements from $R_i$ (checking that element is in $R_i$) is a relatively simple procedure. It turns out to be significantly difficult to sample elements from the intersection of $R_i$'s (formula's numerator) and to check if they are in symmetric commutant (formula's denominator), there is no explicit algorithm for these problems. We propose several approximate algorithms, using a wide variety of approaches from optimization theory and application of neural networks to NLP problems.

- [1] Jie Wu. "Combinatorial descriptions of homotopy groups of certain spaces". В: Mathematical Proceedings of the Cambridge Philosophical Society. Т. 130. 3. Cambridge University Press. 2001, с. 489—513.
- [2] Ralph Fox. "Free Differential Calculus, I: Derivation in the Free Group Ring". Annals of Mathematics. Т. 57. 3. 1953, с. 547–560.
- [3] Roman Mikhailov. "Homotopy theory of Lie functors". arXiv preprint arXiv:1808.00681. 2018.

## Progress

### Group tools

#### Module's implemented

- Generation of the words, representing elements from free group, normal subgroups; reduction; checking if element is in normal subgroup ```freegroup.tools```, ```freegroup.sampling```;
- Automatic calculation of Fox derivatives [2] ```freegroup.derivatives```;
- Calculation of dimension of derived functor of Lie functor [3] ```lie-derived-functors```.

#### Implementation's paused

- Generation of expressions of commutators, Hall's commutator collecting process.

### Sampling from the intersection of normal subgroups (Wu formula's numerator)

#### Method's implemented

- Random sampling of the works from free group and filtration by checking if elements are in necessary normal subgroups ```sampling.trivial_sampler```;
- Global optimization using $(1+1)$-evolutionary algorithm of the approximate distance to the intersection of subgroups $\Sigma_{i=0}^{n-1} d(x, R_i)$, where $d(x, R_i)$ is the length of reduced word after substitution $x_i \to e$ or $x_1 \to x_{n-1}^{-1} \dots x_2^{-1}$ ```sampling.evolutionary_sampler```;
- Global continuous optimization using multistart method and gradient descent with clipping of the distance to unit in Sanov embeddings after corresponding substitutions ```sampling.matrix_sampler```;
- Обучение ансамбля нейронных сетей архитектуры LSTM генерированию следующей буквы в слове, минимизирующему дивергенцию Кульбака-Лейблера с эмпирическим распределением ```sampling.language_model_sampler```;
- Обучение нейросети архитектуры Transformer генерированию следующей буквы в слове, минимизирующему дивергенцию Кульбака-Лейблера с эмпирическим распределением ```gpt.GPT_2_Words_generation```;
- Обучение с подкреплением генеративно-состязательной нейронной сети генерированию слов, с высокой вероятностью принадлежащих нормальной подгруппе ```seqGAN.SeqGAN_Generation_words```.

#### Research's paused

- Непрерывная оптимизация расстояния до пересечения подгрупп при квазиизометрическом вложении графа Кэли свободной группы в гиперболическое пространство ```embedding```;
- Максимизация активации нейронной сети, обученной проверять принадлежность слова нормальной подгруппе ```sampling.activation_maximization_sampler```.

### Filtering for the factorization by symmetric commutant (Wu formula's denominator)

#### Method's implemented

- 

## Installation

To install module `freegroup`, run the following commands:

- `python -m pip install -r freegroup/requirements.txt`
- `python -m pip install ./freegroup`

After this, you'll be able to use packages `freegroup.tools`, `freegroup.sampling`, `freegroup.derivatives`.

## Rights

Copyright © Fedor Pavutnitskiy, Dmitrii Vilensky-Pasechnyuk, Kirill Brilliantov and German Magai
