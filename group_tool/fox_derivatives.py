def augmentation(expr):
    if isinstance(expr, int):
        return expr
    return expr.augmentation

class Expression:
    def __init__(self, augmentation):
        self.augmentation = augmentation

class GeneratorExpression(Expression):
    def __init__(self, generator):
        super().__init__(1)
        self.generator = generator

class SumExpression(Expression):
    @staticmethod
    def create(lhs, rhs):
        if isinstance(lhs, int) and isinstance(rhs, int):
            return lhs + rhs
        if lhs == 0:
            return rhs
        if rhs == 0:
            return lhs
        return SumExpression(lhs, rhs)

    def __init__(self, lhs, rhs):
        super().__init__(augmentation(lhs) + augmentation(rhs))
        self.lhs = lhs
        self.rhs = rhs

class MultiplicativeExpression(Expression):
    @staticmethod
    def create(lhs, rhs):
        if isinstance(lhs, int) and isinstance(rhs, int):
            return lhs * rhs
        if isinstance(lhs, int) and isinstance(rhs, MultiplicativeExpression) and isinstance(rhs.lhs, int):
            return MultiplicativeExpression(rhs.lhs * lhs, rhs.rhs)
        if lhs == 0:
            return 0
        if rhs == 0:
            return 0
        if lhs == 1:
            return rhs
        if rhs == 1:
            return lhs
        return MultiplicativeExpression(lhs, rhs)

    def __init__(self, lhs, rhs):
        super().__init__(augmentation(lhs) * augmentation(rhs))
        self.lhs = lhs
        self.rhs = rhs



from math import ceil
from string import ascii_lowercase

def expression_of_word(word):
    if len(word) == 1:
        return GeneratorExpression(word[0])
    m = ceil(len(word) / 2)
    return MultiplicativeExpression(expression_of_word(word[:m]), expression_of_word(word[m:]))

def print_expression(expression, depth = 0):
    if isinstance(expression, int):
        print(' ' * depth * 2, expression)
    elif isinstance(expression, GeneratorExpression):
        letter = ascii_lowercase[abs(expression.generator) - 1]
        print(' ' * depth * 2, letter if expression.generator > 0 else letter.upper())
    elif isinstance(expression, SumExpression):
        print(' ' * depth * 2, 'SUM')
        print_expression(expression.lhs, depth + 1)
        print_expression(expression.rhs, depth + 1)
    elif isinstance(expression, MultiplicativeExpression):
        print(' ' * depth * 2, 'MULT')
        print_expression(expression.lhs, depth + 1)
        print_expression(expression.rhs, depth + 1)




def derivative(u, wrt):

    if isinstance(u, int):    
        return 0

    if isinstance(u, GeneratorExpression):
        if abs(u.generator) != abs(wrt):
            return 0
        if u.generator != wrt:
            return MultiplicativeExpression.create(-1, u)
        return 1
        
    if isinstance(u, SumExpression):
        ui = derivative(u.lhs, wrt)
        uj = derivative(u.rhs, wrt)
        return SumExpression.create(ui, uj)

    if isinstance(u, MultiplicativeExpression):
        ui = derivative(u.lhs, wrt)
        uj = derivative(u.rhs, wrt)

        res_lhs = MultiplicativeExpression.create(augmentation(u.rhs), ui)
        res_rhs = MultiplicativeExpression.create(u.lhs, uj)   
        
        return SumExpression.create(res_lhs, res_rhs)
    
    raise ValueError(f'Unknown expression type {u}')


'''
derivatives, labels = calculate_all_derivatives([-1, -2, 1, 2], modulo_length=3, n_generators=2)

for coef, label in zip(derivatives, labels):
    print(f'{label}: {augmentation(coef)}')

> []: 1
> [1]: 0
> [2]: 0
> [1, 1]: 0
> [2, 1]: -1
> [1, 2]: 1
> [2, 2]: 0

'''
def calculate_all_derivatives(word, modulo_length, n_generators):
    derivatives = [[expression_of_word(word)]]
    labels = [[[]]]
    for _ in range(modulo_length - 1):
        derivatives.append([])
        labels.append([])
        for u, label in zip(derivatives[-2], labels[-2]):
            for gen in range(1, n_generators + 1):
                derivatives[-1].append(derivative(u, gen))
                labels[-1].append([gen] + label)
    return sum(derivatives, []), sum(labels, [])
