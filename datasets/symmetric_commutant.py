import pickle
from itertools import islice
from argparse import ArgumentParser
from group_tool.reduced_words import symmetric_commutant


parser = ArgumentParser(description='Generate dataset of elements from symmetric commutant')
parser.add_argument('generators_number', type=int, help='number of generators (x, y, z, ...)')
parser.add_argument('max_length', type=int, help='maximal length of word from one normal closure (generated to be included in result)')
parser.add_argument('size', type=int, help='desired number of words in the dataset')
args = parser.parse_args()

symcom = symmetric_commutant(args.generators_number, args.max_length)
dataset = list(islice(symcom, args.size))

with open(f"datasets/symcom_n={args.generators_number}_l={args.max_length}.pkl", "wb") as file:
    pickle.dump(dataset, file)
