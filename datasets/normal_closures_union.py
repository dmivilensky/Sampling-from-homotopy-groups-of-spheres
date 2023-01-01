import pickle
from itertools import islice
from argparse import ArgumentParser
from group_tool.reduced_words import normal_closure
from group_tool.utils import random_union


parser = ArgumentParser(description='Generate dataset of elements from union of normal closures')
parser.add_argument('generators_number', type=int, help='number of generators (x, y, z, ...)')
parser.add_argument('max_length', type=int, help='maximal length of generated word')
parser.add_argument('size', type=int, help='desired number of words in the dataset')
args = parser.parse_args()

closures =\
    [normal_closure([[i]], args.generators_number, args.max_length) for i in range(1, args.generators_number + 1)] +\
    [normal_closure([list(range(1, args.generators_number + 1))], args.generators_number, args.max_length)]

clsunion = random_union(closures)
dataset = list(islice(clsunion, args.size))

with open(f"datasets/clsunion_n={args.generators_number}_l={args.max_length}.pkl", "wb") as file:
    pickle.dump(dataset, file)
