from pprint import pprint
from itertools import islice
from free_group import normal_closure, free_group_bounded, is_from_normal_closure, print_words

max_length = 6
m = 20

x = [1]
y = [2]
xy = [1, 2]

# ncl_x = list(islice(normal_closure([x], max_length=max_length), m // 2))
# free_group_minus_ncl_x = list(islice(filter(
#     lambda word: not is_from_normal_closure(x, word),
#     free_group_bounded(max_length=max_length)), m // 2))
# ncl_x_ncl_y_union = list(islice(filter(
#     lambda word: is_from_normal_closure(x, word) and is_from_normal_closure(y, word),
#     free_group_bounded(max_length=max_length)), m // 2))

# print_words(ncl_x_ncl_y_union)

ncl_x_ncl_y_ncl_xy_intersection = list(islice(filter(
    lambda word: is_from_normal_closure(x, word) and is_from_normal_closure(y, word) and is_from_normal_closure(xy, word),
    free_group_bounded(max_length=max_length)), m))

print_words(ncl_x_ncl_y_ncl_xy_intersection)