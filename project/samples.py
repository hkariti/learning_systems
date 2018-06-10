import numpy
import scipy.io
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

bmat = scipy.io.loadmat("BreastCancerData.mat")

X = numpy.transpose(bmat['X'])
y = bmat['y'][...,0]

_pca = PCA(n_components=2)
X_2d = _pca.fit_transform(X)

_test_indices = [3,11,17,21,22,28,29,41,44,46,68,69,70,77,84,95,116,124,127,130,133,139,144,145,147,162,166,168,173,183,184,185,188,195,198,208,212,213,215,221,222,226,227,230,234,243,244,247,252,256,257,259,261,263,266,271,275,276,285,292,293,297,298,307,309,310,313,318,321,326,337,345,346,356,358,362,363,364,374,377,378,379,381,389,414,416,423,438,443,446,456,462,465,467,469,477,478,480,486,489,492,493,505,514,516,518,519,520,523,524,525,538,549,553,568]
test_set_idx = numpy.zeros(len(y), dtype=numpy.bool)
test_set_idx[_test_indices] = 1

training_set_idx = numpy.ones(len(y), dtype=numpy.bool) ^ test_set_idx

def graph_tagged_samples_2d():
    scatter_x = X_2d[...,0]
    scatter_y = X_2d[...,1]
    good_indices = y == 0
    bad_indices = numpy.invert(good_indices)
    f = plt.figure()
    plt.scatter(scatter_x[good_indices],scatter_y[good_indices], c='b', s=3)
    plt.scatter(scatter_x[bad_indices], scatter_y[bad_indices], c='r', s=3)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("Tagged samples in 2d")
    plt.legend(["Benign", "Malignant"])
    return f
