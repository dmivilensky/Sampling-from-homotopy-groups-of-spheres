import pickle
from argparse import ArgumentParser
from freegroup.sampling import normal_closure_conjugation as normal_closure, random_order_commutant
from itertools import islice
from iteration_utilities import unique_everseen

parser = ArgumentParser(description='Generate dataset of elements from symmetric commutant')
parser.add_argument('generators_number', type=int, help='number of generators (x, y, z, ...)')
parser.add_argument('max_length', type=int, help = 'maximal length of generated words')
parser.add_argument('part_max_length', type=int, help='maximal length of word from one normal closure (generated to be included in result')
parser.add_argument('max_number_multipliers', type=int, help='maximal number of symmetric commutant multipliers')
parser.add_argument('size', type=int, help='desired number of words in the dataset')
args = parser.parse_args()

generators = [[i] for i in range(1, args.generators_number + 1)] + \
    [list(range(1, args.generators_number + 1))]

closures = [normal_closure(g, generators_number=args.generators_number, max_length=args.part_max_length) for g in generators]

g = random_order_commutant(closures)

g = filter(lambda x: len(x) < args.max_length, g)
dataset = list(islice(unique_everseen(g, key = tuple), args.size))

print('average length =', int(sum(map(len, dataset)) / len(dataset)))
real_max_length = max(map(len, dataset))
print('max length =', real_max_length)

with open(f"datasets/symcom_n={args.generators_number}_l={real_max_length}.pkl", "wb") as file:
    pickle.dump(dataset, file)
