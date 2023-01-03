import pickle
import numpy as np
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix
from sampling.intersection.matrix_sampler import one_hot, MatrixSampler

generators_number = 3
max_len = 66

def matrix_embedding(x):
    oh = one_hot(generators_number, x)
    matrix = MatrixSampler.distance_from_normal_closure(generators_number, [], oh, distance=False)
    return matrix.flatten()

with open(f"datasets/symcom_n={generators_number}_l={max_len}.pkl", "rb") as file:
    symcom = pickle.load(file)

with open(f"datasets/clsunion_n={generators_number}_l={max_len}.pkl", "rb") as file:
    clsunion = pickle.load(file)

print('to matrices...')
symcom = list(map(matrix_embedding, symcom))
clsunion = list(map(matrix_embedding, clsunion))

X = symcom + clsunion
y = [1] * len(symcom) + [0] * len(clsunion)

print('training k-SVM...')
svclassifier = SVC(kernel="poly", degree=8)
svclassifier.fit(X, y)

print(confusion_matrix(y, svclassifier.predict(X)))

print('training CatBoost...')
cat = CatBoostClassifier(iterations=100)
cat.fit(X, y, verbose=False)

print(confusion_matrix(y, cat.predict(X)))
