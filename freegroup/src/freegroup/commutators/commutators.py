from typing import List, Callable
from dataclasses import dataclass

from freegroup import tools
from functools import reduce


class Expr:
    def __init__(self): pass
    
@dataclass
class Commutator(Expr):
    left: Expr
    right: Expr

@dataclass
class Multiplication(Expr):
    children: List[Expr]




class Visitor:
    def __call__(self, expr):
        if isinstance(expr, list): 
            return self.visit_word(expr)
        if isinstance(expr, Commutator):
            return self.visit_commutator(expr)
        if isinstance(expr, Multiplication):
            return self.visit_mult(expr)
        raise ValueError('Unknown expr type')

    def visit_word(self, word): pass

    def visit_commutator(self, commutator):
        return list(map(self, [commutator.left, commutator.right]))

    def visit_mult(self, mult):
        return list(map(self, mult.children))


class Normalize(Visitor):
    def visit_word(self, word):
        return tools.normalize(word)

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
        if left[:min_length] == tools.reciprocal(right):
            left, right = Normalize._trim_commutees(left[min_length:], right)
        if left[-min_length:] == tools.reciprocal(right):
            left, right = Normalize._trim_commutees(left[-min_length:], left[:-min_length])
        
        if not is_right_min:
            return right, left
        return left, right

    def visit_commutator(self, commutator):
        left, right = super().visit_commutator(commutator)

        if isinstance(left, list) and isinstance(right, list):
            left, right = Normalize._trim_commutees(left, right)

        for child in [left, right]:
            if isinstance(child, list) and not child:
                return []

        return Commutator(left, right)

    def visit_mult(self, mult):
        children = []
        for child in super().visit_mult(mult):
            if isinstance(child, Multiplication):
                children.extend(child.children)
            elif children and isinstance(children[-1], list) and isinstance(child, list):
                merged = tools.multiply(children.pop(), child)
                children.append(self.visit_word(merged))
            elif isinstance(child, list) and not child:
                continue
            else:
                children.append(child)
        if not children:
            return []
        return Multiplication(children) if len(children) > 1 else children[0]

normalize = Normalize()
        

class ToFreegroup(Visitor):
    def visit_word(self, word):
        return word

    def visit_commutator(self, commutator):
        left, right = super().visit_commutator(commutator)
        return tools.commutator(left, right)

    def visit_mult(self, mult):
        return reduce(tools.multiply, super().visit_mult(mult), [])

to_freegroup = ToFreegroup()


@dataclass
class ToString(Visitor):

    begin_commutator_token: str     = '['
    end_commutator_token: str       = ']'
    sep_commutator_token: str       = ','
    begin_word_token: str           = ''
    sep_word_token: str             = ' '
    end_word_token: str             = ''
    letter_fn: Callable[[int], str] = str
    begin_multiplication_token      = ''
    sep_multiplication_token        = ''
    end_multiplication_token        = ''

    def visit_word(self, word):
        return f'{self.begin_word_token}{self.sep_word_token.join(map(self.letter_fn, word))}{self.end_word_token}'

    def visit_commutator(self, commutator):
        left, right = super().visit_commutator(commutator)
        return f'{self.begin_commutator_token}{left}{self.sep_commutator_token}{right}{self.end_commutator_token}'

    def visit_mult(self, mult):
        return f'{self.begin_multiplication_token}{self.sep_multiplication_token.join(super().visit_mult(mult))}{self.end_multiplication_token}'


import parsec


@dataclass
class FromString():
    begin_commutator_token: str     = '['
    end_commutator_token: str       = ']'
    sep_commutator_token: str       = ','
    begin_word_token: str           = None
    sep_word_token: str             = None
    end_word_token: str             = None
    letter_regex: str               = r'(-[0-9]+|[0-9]+)'
    letter_fn: Callable[[str], int] = int
    begin_multiplication_token: str = None
    end_multiplication_token: str   = None
    sep_multiplication_token: str   = None

    def __post_init__(self):

        within_spaces = lambda x: parsec.spaces() >> x << parsec.spaces()
        token         = lambda x: within_spaces(parsec.string(x))
        
        @parsec.generate
        def word():
            
            if not self.begin_word_token is None:
                _ = yield token(self.begin_word_token)
            
            if not self.sep_word_token is None:
                raw_word = yield parsec.sepBy1(
                    within_spaces(parsec.regex(self.letter_regex)),
                    token(self.sep_word_token)
                )
            else:
                raw_word = yield parsec.many1(
                    within_spaces(parsec.regex(self.letter_regex))
                )
            
            if not self.end_word_token is None:
                _ = yield within_spaces(self.end_word_token)

            return list(map(self.letter_fn, raw_word))

        @parsec.generate
        def commutator():
            
            if not self.begin_commutator_token is None:
                _ = yield token(self.begin_commutator_token)
            
            left = yield within_spaces(multiplication)
            
            if not self.sep_commutator_token is None:
                _ = yield token(self.sep_commutator_token)
            
            right = yield within_spaces(multiplication)
            
            if not self.end_commutator_token is None:
                _ = yield token(self.end_commutator_token)
            
            return Commutator(left, right)

        @parsec.generate
        def multiplication():
            if not self.begin_multiplication_token is None:
                _ = yield token(self.begin_multiplication_token)
            
            multipliers = yield parsec.many1(within_spaces(parsec.try_choice(commutator, word)))
            
            if not self.end_multiplication_token is None:
                _ = yield token(self.end_multiplication_token)

            return Multiplication(multipliers)
        
        self.parse = multiplication.parse_strict
        self.normalize = Normalize()
    
    def __call__(self, string: str):
        return self.normalize(self.parse(string))

def _to_lu_repr(number):
    letters = 'xyzpqrst'
    return letters[abs(number) - 1].upper() if number < 0 else letters[number - 1]

def _from_lu_repr(letter):
    letters = 'xyzpqrst'
    return (-1 if letter.isupper() else 1) * (letters.index(letter.lower()) + 1)

to_tokenizer = ToString()
from_tokenizer = FromString()

to_lu = ToString(letter_fn=_to_lu_repr, sep_word_token='')
from_lu = FromString(letter_regex=r'[a-zA-Z]{1}', letter_fn=_from_lu_repr)
