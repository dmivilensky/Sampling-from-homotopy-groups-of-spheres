import pickle
from itertools import islice
from argparse import ArgumentParser
from freegroup.sampling import symmetric_commutant, take_unique
from itertools import repeat

parser = ArgumentParser(description='Generate dataset of elements from symmetric commutant')
parser.add_argument('generators_number', type=int, help='number of generators (x, y, z, ...)')
parser.add_argument('max_length', type=int, help = 'maximal length of generated words')
parser.add_argument('part_max_length', type=int, help='maximal length of word from one normal closure (generated to be included in result')
parser.add_argument('max_number_multipliers', type=int, help='maximal number of symmetric commutant multipliers')
parser.add_argument('size', type=int, help='desired number of words in the dataset')
args = parser.parse_args()

generators = [[i] for i in range(1, args.generators_number + 1)] + \
    [list(range(1, args.generators_number + 1))]

g = symmetric_commutant(generators,
    args.generators_number,
    args.max_number_multipliers,
    max_length = args.part_max_length
)
g = filter(lambda x: len(x) < args.max_length, g)
dataset = list(take_unique(args.size, g))

print('average length =', int(sum(map(len, dataset)) / len(dataset)))
real_max_length = max(map(len, dataset))
print('max length =', real_max_length)

with open(f"datasets/symcom_n={args.generators_number}_l={real_max_length}.pkl", "wb") as file:
    pickle.dump(dataset, file)
