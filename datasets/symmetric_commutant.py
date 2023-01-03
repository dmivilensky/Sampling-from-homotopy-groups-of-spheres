import pickle
from itertools import islice
from argparse import ArgumentParser
import freegroups.sampling as smp
from freegroups.tools import commutator, multiply, normalize
from itertools import repeat

parser = ArgumentParser(description='Generate dataset of elements from symmetric commutant')
parser.add_argument('generators_number', type=int, help='number of generators (x, y, z, ...)')
parser.add_argument('max_length', type=int, help = 'maximal length of generated words')
parser.add_argument('part_max_length', type=int, help='maximal length of word from one normal closure (generated to be included in result')
parser.add_argument('max_number_multipliers', type=int, help='maximal number of symmetric commutant multipliers')
parser.add_argument('size', type=int, help='desired number of words in the dataset')
args = parser.parse_args()

gs = [smp.normal_closure([[i]], args.generators_number, max_length=args.part_max_length) for i in range(1, args.generators_number + 1)] +\
    [smp.normal_closure([list(range(1, args.generators_number + 1))], args.generators_number, args.part_max_length)]

g = smp.join(gs)
g = smp.shuffle(g)
g = smp.reduce(commutator, g)

g = smp.join(*repeat(g, args.max_number_multipliers))
g = smp.subset(g)
g = smp.reduce(multiply, g)

g = map(normalize, g)
g = filter(lambda x: len(x) > 0 and len(x) < args.max_length, g)

dataset = list(smp.take_unique(args.size, g))

print('average length =', int(sum(map(len, dataset)) / len(dataset)))
real_max_length = max(map(len, dataset))
print('max length =', real_max_length)

with open(f"datasets/symcom_n={args.generators_number}_l={real_max_length}.pkl", "wb") as file:
    pickle.dump(dataset, file)
