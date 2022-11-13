import numpy as np
from numpy.linalg import inv
from itertools import chain
from matplotlib import pyplot as plt
import group_tool.reduced_words as fg

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def all_words(number_of_generators, word_length):
    X_gens = np.arange(1,number_of_generators + 1)
    X_even = np.empty(2 * number_of_generators, dtype = np.int32)
    X_even[0::2] = X_gens
    X_even[1::2] = -X_gens
    X_odd = -X_even

    if word_length > 1:
        steps = np.pad(np.asarray(np.meshgrid(*np.repeat([np.arange(1, 2 * number_of_generators)], word_length - 1, axis = 0))).T.reshape(-1, word_length - 1),((0,0),(1,0)))
    else:
        steps = np.zeros(1, dtype = np.int32)
    first_letter = np.arange(2 * number_of_generators)

    X = np.empty((steps.shape[0] * 2 * number_of_generators, word_length), dtype = np.int32)
    idx = 0
    for f in first_letter:
        for s in steps:
            ind = np.mod(f + np.cumsum(s), 2 * number_of_generators)
            ind_even = ind[0::2]
            ind_odd = ind[1::2]
            X[idx, 0::2] = X_even[ind_even]
            X[idx, 1::2] = X_odd[ind_odd]
            idx = idx + 1
    return X

def get_all_words_and_edges(words):
    max_length = len(words[0])
    if max_length == 1:
        return [(words, np.zeros(4, dtype = np.int32))]
    else:
        prev_words, edges_idx = np.unique(words[:,np.arange(max_length - 1)], return_inverse = True, axis = 0)
        return get_all_words_and_edges(prev_words) + [(words, edges_idx)]

x = np.asarray([[1,2],[0,1]])
y = np.asarray([[1,0],[2,1]])

def generators_map(n):
    match n:
        case 1: return x
        case 2: return y
        case -1: return inv(x)
        case -2: return inv(y)
        case 0: return np.identity(2)

def get_matrix(words):
    return [np.linalg.multi_dot(list(map(generators_map, w))+[np.identity(2)]) for w in words]

def mobius(z, A):
    return (z * A[0,0] + A[0,1]) / (z * A[1,0] + A[1,1])

def get_geodesic(z1, z2, t):
    middle_point = (z2 + z1) / 2
    circle_center = (abs(z2) ** 2 - abs(z1) ** 2) / (2 * (z2.real - z1.real))
    radius = abs(z1 - circle_center)
    return circle_center + radius * np.exp(1j * ( t * np.angle(z2 - circle_center) + (1 - t) * np.angle(z1 - circle_center) ))

def words_to_vecs(words):
    matrices = get_matrix(words)
    return np.asarray(list(map(lambda m: mobius(1j, m), matrices)))

def word_to_str(word):
    return ''.join(list(map(lambda c : {1 : 'x', 2 : 'y', -1 : 'x\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}', -2 : 'y\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}'}[c], word)))

def color_function(color_code):
    return {
        0 : '#1f77b4',
        1 : '#e8320b',
        10 : '#c00ae8',
        100 : '#4fcb08',
        101 : '#d1e80a',
        110 : '#0ad8e8',
        111 : '#000000'
    }[color_code]

def coloring_list(words):
    x_closure = [int(fg.is_from_singleton_normal_closure([[1]],w)) for w in words]
    y_closure = [int(fg.is_from_singleton_normal_closure([[2]],w)) * 10 for w in words]
    xy_closure = [int(fg.is_from_singleton_normal_closure([[1,2]],w)) * 100 for w in words]
    colors = np.sum(np.asarray([x_closure, y_closure, xy_closure], dtype = np.int32), axis = 0)
    return [color_function(c) for c in colors]

max_length = 6
words = all_words(2, max_length)
tail = get_all_words_and_edges(words)
vecs = [words_to_vecs(t[0]) for t in tail]

vecs_for_plotting = np.hstack(vecs)
words_for_coloring = list(chain.from_iterable([t[0].tolist() for t in tail]))
words_for_annotating = [word_to_str(w) for w in words_for_coloring]

fig,ax = plt.subplots()
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax.set_yticks([1,2])

arc_param = np.linspace(0,1,20)
for idx, t in enumerate(tail):
    if idx > 0:
        for edge_start, edge_end in enumerate(t[1]):
            pt1 = vecs[idx - 1][edge_end]
            pt2 = vecs[idx][edge_start]
            data = get_geodesic(pt1, pt2, arc_param)
            plt.plot(data.real, data.imag, color = '#1f77b4')

for i in range(4):
    init_data = get_geodesic(1j, vecs[0][i], arc_param)
    plt.plot(init_data.real, init_data.imag, color = '#1f77b4')

plt.xlim([-10,10])
plt.ylim([0,2])
plt.grid(True)
sc = plt.scatter(vecs_for_plotting.real,vecs_for_plotting.imag, zorder=3, c = np.asarray(coloring_list(words_for_coloring)));
plt.scatter(0,1, zorder = 3, color = '#1f77b4')

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    annot.set_text(words_for_annotating[ind['ind'][0]])

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)
plt.show()
