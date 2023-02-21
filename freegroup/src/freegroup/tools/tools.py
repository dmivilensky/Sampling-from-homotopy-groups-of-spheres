from typing import Callable
from dataclasses import dataclass

class Visitor:
    def __call__(self, word):
        if isinstance(word, int):
            return self.visit_generator(word)  
        if isinstance(word, list):
            return self.visit_multiplication(word)
        if isinstance(word, tuple):
            return self.visit_commutator(word)

    def batch_visit(self, words):
        return [self(x) for x in words]
    
    def visit_generator(self, generator): pass

    def visit_commutator(self, commutator): pass

    def visit_multiplication(self, multiplication): pass


class Invert(Visitor):
    def visit_generator(self, generator):
        return -generator

    def visit_commutator(self, commutator):
        return tuple(x for x in reversed(commutator))

    def visit_multiplication(self, multiplication):
        return [self(x) for x in multiplication[::-1]]

reciprocal = Invert()
batch_invert = reciprocal.batch_visit


class Flatten(Visitor):
    def visit_generator(self, generator): return [generator]

    def visit_commutator(self, commutator):
        commutator = tuple(self(x) for x in commutator)
        result = commutator[0]
        for idx in range(1, len(commutator)):
            result = reciprocal(result) + reciprocal(commutator[idx]) + result + commutator[idx]
        return result

    def visit_multiplication(self, multiplication):
        result = []
        for x in multiplication: 
            result.extend(self(x))
        return result

flatten = Flatten()
batch_flatten = flatten.batch_visit

class Conjugate(Visitor):
    def __init__(self, conjugator):
        self.conjugator = conjugator

    def visit_generator(self, generator):
        return [reciprocal(self.conjugator)] + [generator] + [self.conjugator]

    def visit_commutator(self, commutator):
        return tuple(self(x) for x in commutator)

    def visit_multiplication(self, multiplication):
        if len(multiplication) == 1 and isinstance(multiplication[0], tuple):
            return [self.visit_commutator(multiplication[0])]
        else:
            return [reciprocal(self.conjugator)] + multiplication + [self.conjugator]

def conjugate(word, conjugator):
    return Conjugate(conjugator)(word)

def batch_conjugate(words, conjugator):
    return Conjugate(conjugator).batch_visit(words)


class Clone(Visitor):
    def visit_generator(self, generator): return generator

    def visit_commutator(self, commutator): tuple(self(x) for x in commutator)

    def visit_multiplication(self, multiplication): return [self(x) for x in multiplication]

clone = Clone()
batch_clone = clone.batch_visit


class Normalize(Visitor):
    def visit_generator(self, generator): return generator

    def visit_commutator(self, commutator): return tuple(self(x) for x in commutator)

    def visit_multiplication(self, mutliplication):
        children = []

        def handle(x):
            if isinstance(x, list):
                for y in x: handle(y)
            elif children and isinstance(x, int) and isinstance(children[-1], int) and children[-1] == -x:
                del children[-1:]
            else:
                children.append(x)
    
        for x in map(self, mutliplication): handle(x)

        if len(children) == 1 and not isinstance(children[0], int):
            return children[0]
        
        return children

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


def reduce_modulo_singleton_normal_closure(word, closure = None):
    word = flatten(word)
    closure_len = len(closure)

    reduced = []
    for factor in word:
        reduced.append(factor)

        if len(reduced) >= 2 and reduced[-2] == -reduced[-1]:
            del reduced[-2:]

        if len(reduced) >= closure_len:
            if is_trivial(word[-closure_len:], closure):
                del reduced[-closure_len:]
            
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

    def visit_generator(self, generator):
        return self.generator_representation_fn(generator)

    def visit_commutator(self, commutator):
        children = (self(x) for x in commutator)
        return f'{self.begin_commutator_token}{self.sep_commutator_token.join(children)}{self.end_commutator_token}'

    def visit_multiplication(self, mult):
        children = [self(x) for x in mult]
        return f'{self.begin_multiplication_token}{self.sep_multiplication_token.join(children)}{self.end_multiplication_token}'


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
            raw_generator = yield parsec.regex(self.generator_representation_regex)
            return self.generator_representation_map(raw_generator)
            

        @parsec.generate
        def commutator():
            if not self.begin_commutator_token is None:
                _ = yield token(self.begin_commutator_token)

            children = yield parsec.sepBy1(multiplication, token(self.sep_commutator_token))
            
            if not self.end_commutator_token is None:
                _ = yield token(self.end_commutator_token)
            
            return tuple(children)

        @parsec.generate
        def multiplication():
            if not self.begin_multiplication_token is None:
                _ = yield token(self.begin_multiplication_token)
            
            multipliers = yield parsec.many1(within_spaces(parsec.try_choice(commutator, generator)))
            
            if not self.end_multiplication_token is None:
                _ = yield token(self.end_multiplication_token)

            return multipliers
        
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

def to_string(word, method = None, **kwargs):
    if len(kwargs) == 0 and method is None:
        raise ValueError('You should specify either kwargs for `ToString` or `method`')
    if method is None:
        return ToString(**kwargs)(word)
    if method in ['int', 'integer', 'tokenizer']:
        return to_integer_representation(word)
    if method in ['lu', 'lower_upper', 'lower-upper']:
        return to_lower_upper_representation(word)
    if method in ['superscript', 'su']:
        return to_superscript_representation(word)
    raise ValueError('Unknown representation method')

def from_string(word, method = None, **kwargs):
    if len(kwargs) == 0 and method is None:
        raise ValueError('You should specify either kwargs for `FromString` or `method`')
    if method is None:
        return FromString(**kwargs)(word)
    if method in ['int', 'integer', 'tokenizer']:
        return from_integer_representation(word)
    if method in ['lower_upper', 'lower-upper', 'lu']:
        return from_lower_upper_representation(word)
    if method in ['superscript', 'su']:
        return from_superscript_representation(word)
    raise ValueError('Unknown representation method')
