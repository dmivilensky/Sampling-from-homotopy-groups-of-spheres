from free_group import print_word
from trivial_sampler import TrivialSampler


print('sampler')
sampler = TrivialSampler(baseline="joint", max_length=6)
for i in range(10):
    print_word(next(sampler))
