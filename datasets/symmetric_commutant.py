import pickle
from itertools import islice
from argparse import ArgumentParser
from freegorup.sampling import symmetric_commutant


parser = ArgumentParser(description='Generate dataset of elements from symmetric commutant')
parser.add_argument('generators_number', type=int, help='number of generators (x, y, z, ...)')
parser.add_argument('max_length', type=int, help='maximal length of word from one normal closure (generated to be included in result)')
parser.add_argument('size', type=int, help='desired number of words in the dataset')
args = parser.parse_args()

symcom = symmetric_commutant(args.generators_number, args.max_length)
dataset = list(islice(symcom, args.size))

print('average length =', int(sum(map(len, dataset)) / len(dataset)))
real_max_length = max(map(len, dataset))
print('max length =', real_max_length)

with open(f"datasets/symcom_n={args.generators_number}_l={real_max_length}.pkl", "wb") as file:
    pickle.dump(dataset, file)
