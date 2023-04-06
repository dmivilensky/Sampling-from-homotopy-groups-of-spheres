from typing import Callable, List
from dataclasses import dataclass
from functools import reduce

class Expr:
    pass

@dataclass
class Comm(Expr):
    children: List[Expr]
        
@dataclass
class Mult(Expr):
    children: List[Expr]

class Visitor:
    def __call__(self, word):
        if isinstance(word, list): return self.visit_word(word)
        if isinstance(word, Comm): return self.visit_comm(word.children)
        if isinstance(word, Mult): return self.visit_mult(word.children)

    def batch_visit(self, words):
        return [self(x) for x in words]
    
    def visit_word(self, generator): pass

    def visit_comm(self, commutator): pass

    def visit_mult(self, multiplication): pass


class Invert(Visitor):
    def visit_word(self, word):
        return [-f for f in word[::-1]]

    def visit_comm(self, children):
        return Comm([x for x in reversed(children)])

    def visit_mult(self, children):
        return Mult([self(x) for x in children[::-1]])

reciprocal = Invert()
batch_invert = reciprocal.batch_visit


class Flatten(Visitor):
    def visit_word(self, word): return word

    def visit_comm(self, children):
        return reduce(lambda x, y: reciprocal(x) + reciprocal(y) + x + y, map(self, children))

    def visit_mult(self, children):
        return reduce(lambda x, y: x + y, map(self, children))

flatten = Flatten()
batch_flatten = flatten.batch_visit

class Conjugate(Visitor):
    def __init__(self, conjugator):
        self.conjugator = conjugator

    def visit_word(self, word):
        return reciprocal(self.conjugator) + word + self.conjugator

    def visit_comm(self, children):
        return Comm([self(x) for x in childnre])

    def visit_mult(self, children):
        if len(children) == 1 and isinsatnce(children[0], Comm):
            return Mult([self(children[0])])
        else:
            return Mult([reciprocal(self.conjugator)] + children + [self.conjugator])

def conjugate(word, conjugator):
    return Conjugate(conjugator)(word)

def batch_conjugate(words, conjugator):
    return Conjugate(conjugator).batch_visit(words)


class Clone(Visitor):
    def visit_word(self, word): return word[::]

    def visit_comm(self, children): Comm([self(x) for x in children])

    def visit_mult(self, children): return Mult([self(x) for x in multiplication])

clone = Clone()
batch_clone = clone.batch_visit


class Normalize(Visitor):
    
    
    @staticmethod
    def _trim_commutees(left, right):

        '''
        [xy, x] = [y, x]
        [xy, X] = [y, X]
        [xy, Y] = [y, x]
        [x, xy] = [x, y]
        [X, xy] = [X, y]
        [Y, xy] = [x, y]
        '''

        min_length = min(len(left), len(right))

        if min_length == 0:
            return left, right

        is_right_min = len(right) == min_length
        if not is_right_min:
            left, right = right, left

        if left[:min_length] == right:
            left, right = Normalize._trim_commutees(left[min_length:], right)
        if left[:min_length] == reciprocal(right):
            left, right = Normalize._trim_commutees(left[min_length:], right)
        if left[-min_length:] == reciprocal(right):
            left, right = Normalize._trim_commutees(left[-min_length:], left[:-min_length])

        if not is_right_min:
            return right, left
        return left, right
    
    def visit_word(self, word): return reduce_modulo_singleton_normal_closure(word)

    def visit_comm(self, children):
        children = list(map(self, children))
        if isinstance(children[0], list) and isinstance(children[1], list):
            children[0], children[1] = Normalize._trim_commutees(children[0], children[1])
        if any(map(lambda x: isinstance(x, list) and len(x) == 0, children)): return []
        return Comm(children)

    def visit_mult(self, children):
        result = []
        
        for x in map(self, children):
            if isinstance(x, list) and result and isinstance(result[-1], list):
                result[-1] = reduce_modulo_singleton_normal_closure(result[-1] + x)
            else:
                result.append(x)
                
        if len(result) == 1:
            return result[0]
        
        return Mult(result)

normalize = Normalize()
batch_normalize = normalize.batch_visit


from numpy import array, pad

def to_numpy(word):
    return array(flatten(word), int)

def batch_to_numpy(words):
    words = flatten.batch_visit(words)
    max_length = max(map(len, words))
    return array(list(map(lambda v: pad(v, (0, max_length - len(v))), words)))


def is_trivial(word, closure = None):
    if closure is None: return not word

    doubled_closure = closure * 2
    _doubled_closure = reciprocal(closure) * 2

    for idx in range(len(word)):
        if any(map(lambda x: word == x[idx:len(word) + idx], [doubled_closure, _doubled_closure])):
            return True
    return False

def batch_is_trivial(words, closure = None):
    return [is_trivial(x, closure) for x in words]


def _reduce_modulo_singleton_normal_closure_step(reduced, token, closure = None):
    reduced.append(token)

    if len(reduced) >= 2 and reduced[-2] == -reduced[-1]:
        del reduced[-2:]
    
    if not closure is None and len(reduced) >= len(closure):
        if is_trivial(reduced[-len(closure):], closure):
            del reduced[-len(closure):]

def reduce_modulo_singleton_normal_closure(word, closure = None):
    word = flatten(word)

    reduced = []
    for token in word:
        _reduce_modulo_singleton_normal_closure_step(reduced, token, closure)
            
    return reduced

def batch_reduce_modulo_singleton_normal_closure(words, closure = None):
    return [reduce_modulo_singleton_normal_closure(x, closure) for x in words]


def is_from_singleton_normal_closure(word, closure = None):
    return len(reduce_modulo_singleton_normal_closure(word, closure)) == 0


def batch_is_from_singleton_normal_closure(words, closure = None):
    return [is_from_singleton_normal_closure(x, closure) for x in words]


@dataclass
class ToString(Visitor):

    begin_commutator_token: str                         = '['
    end_commutator_token: str                           = ']'
    sep_commutator_token: str                           = ','
    generator_representation_fn: Callable[[int], str]   = str
    begin_multiplication_token: str                     = ''
    sep_multiplication_token: str                       = ' '
    end_multiplication_token: str                       = ''

    def visit_word(self, word):
        return self.sep_multiplication_token.join(map(self.generator_representation_fn, word))

    def visit_comm(self, children):
        return f'{self.begin_commutator_token}{self.sep_commutator_token.join(map(self, children))}{self.end_commutator_token}'

    def visit_mult(self, children):
        return f'{self.begin_multiplication_token}{self.sep_multiplication_token.join(map(self, children))}{self.end_multiplication_token}'


import parsec


@dataclass
class FromString():
    begin_commutator_token: str             = '['
    end_commutator_token: str               = ']'
    sep_commutator_token: str               = ','
    generator_representation_regex: str     = None
    generator_representation_map: Callable  = None
    begin_multiplication_token: str         = None
    end_multiplication_token: str           = None

    def __post_init__(self):

        assert not self.generator_representation_regex is None and \
            not self.generator_representation_map is None 

        within_spaces = lambda x: parsec.spaces() >> x << parsec.spaces()
        token         = lambda x: within_spaces(parsec.string(x))
        
        @parsec.generate
        def generator():
            raw_generators = yield parsec.many1(within_spaces(parsec.regex(self.generator_representation_regex)))
            return list(map(self.generator_representation_map, raw_generators))
            
        @parsec.generate
        def commutator():
            if not self.begin_commutator_token is None:
                _ = yield token(self.begin_commutator_token)

            children = yield parsec.sepBy1(multiplication, token(self.sep_commutator_token))
            
            if not self.end_commutator_token is None:
                _ = yield token(self.end_commutator_token)
            
            return Comm(children)

        @parsec.generate
        def multiplication():
            if not self.begin_multiplication_token is None:
                _ = yield token(self.begin_multiplication_token)
            
            multipliers = yield parsec.many1(within_spaces(parsec.try_choice(commutator, generator)))
            
            if not self.end_multiplication_token is None:
                _ = yield token(self.end_multiplication_token)

            return Mult(multipliers)
        
        self.test = generator.parse_strict
        self.parse = multiplication.parse_strict
    
    def __call__(self, string: str):
        return normalize(self.parse(string))



_LETTERS = "xyzpqrstuvwklmn"

def _to_integer_representation(generator):
    return str(generator)

def _from_integer_representation(generator):
    return int(generator)

to_integer_representation = ToString(
    generator_representation_fn = _to_integer_representation
)
from_integer_representation = FromString(
    generator_representation_regex  = r'(-[0-9]+|[0-9]+)',
    generator_representation_map    = _from_integer_representation
)

def _to_lower_upper_representation(generator):
    return _LETTERS[abs(generator) - 1].upper() if generator < 0 else _LETTERS[generator - 1]

def _from_lower_upper_representation(generator):
    return (-1 if generator.isupper() else 1) * (_LETTERS.index(generator.lower()) + 1)

to_lower_upper_representation = ToString(
    generator_representation_fn = _to_lower_upper_representation,
    sep_multiplication_token    = '',
)
from_lower_upper_representation = FromString(
    generator_representation_regex  = r'([a-zA-Z])',
    generator_representation_map    = _from_lower_upper_representation
)

def _to_superscript_representation(generator):
    return _LETTERS[abs(generator) - 1] + ('' if generator > 0 else '⁻¹')

def _from_superscript_representation(generator):
    sgn = len(generator) <= 1
    return (1 if sgn else -1) * (_LETTERS.index(generator[0]) + 1)

to_superscript_representation = ToString(
    generator_representation_fn = _to_superscript_representation,
    sep_multiplication_token    = '',
)

from_superscript_representation = FromString(
    generator_representation_regex  = r'[a-zA-Z](⁻¹)?',
    generator_representation_map    = _from_superscript_representation,
)

def to_string(word, method = 'tokenizer'):
    if not isinstance(method, str) and not isinstance(method, dict):
        raise ValueError('You should specify either arguemnts for `ToString` or `string` as `method`')
    if not isinstance(method, str):
        return ToString(**method)(word)
    if method in ['int', 'integer', 'tokenizer']:
        return to_integer_representation(word)
    if method in ['lu', 'lower_upper', 'lower-upper']:
        return to_lower_upper_representation(word)
    if method in ['superscript', 'su']:
        return to_superscript_representation(word)
    raise ValueError('Unknown representation method')

def from_string(word, method = 'tokenizer'):
    if not isinstance(method, str) and not isinstance(method, dict):
        raise ValueError('You should specify either arguemnts for `FromString` or `string` as `method`')
    if not isinstance(method, str):
        return FromString(**kwargs)(word)
    if method in ['int', 'integer', 'tokenizer']:
        return from_integer_representation(word)
    if method in ['lower_upper', 'lower-upper', 'lu']:
        return from_lower_upper_representation(word)
    if method in ['superscript', 'su']:
        return from_superscript_representation(word)
    raise ValueError('Unknown representation method')
    
def batch_to_string(words, **kwargs):
    return list(map(lambda x: to_string(x, **kwargs), words))

def batch_from_string(words, **kwargs):
    return list(map(lambda x: from_string(x, **kwargs), words))

def wu_closure(freegroup_dimension, index):
    if index == 0: return list(range(1, freegroup_dimension + 1))
    return [index]
