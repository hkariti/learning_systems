import numpy as np
import scipy.io
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import NeuralNet as neural

bmat = scipy.io.loadmat("BreastCancerData.mat")

X = np.transpose(bmat['X'])
y = bmat['y'][...,0]

_pca = PCA(n_components=2)
X_2d = _pca.fit_transform(X)

_test_indices = [3,11,17,21,22,28,29,41,44,46,68,69,70,77,84,95,116,124,127,130,133,139,144,145,147,162,166,168,173,183,184,185,188,195,198,208,212,213,215,221,222,226,227,230,234,243,244,247,252,256,257,259,261,263,266,271,275,276,285,292,293,297,298,307,309,310,313,318,321,326,337,345,346,356,358,362,363,364,374,377,378,379,381,389,414,416,423,438,443,446,456,462,465,467,469,477,478,480,486,489,492,493,505,514,516,518,519,520,523,524,525,538,549,553,568]
_validation_indices = [485,386,151,50,109,209,431,137,291,388,149,268,502,179,319,152,395,86,290,380,119,434,460,468,302,106,509,387,314,449,554,269,199,331,87,164,501,294,306,408,288,94,303,260,148,10,78,354,27,539,217,481,507,158,105,452,262,255,348,432,163,216,15,537,89,457,483,556,458,506,12,223,34,562,487,277,274,531,527,413,176,205,296,479,476,430,282,495,156,66]
test_set_idx = np.zeros(len(y), dtype=np.bool)
test_set_idx[_test_indices] = 1

validation_set_idx = np.zeros(len(y), dtype=np.bool)
validation_set_idx[_validation_indices] = 1

training_set_idx = np.ones(len(y), dtype=np.bool) ^ test_set_idx ^ validation_set_idx

total_training_set_idx = training_set_idx | validation_set_idx

train_set = X[training_set_idx].astype('float64')
train_tags = y[training_set_idx].reshape(-1, 1).astype('float64')
validation_set = X[validation_set_idx]
validation_tags = y[validation_set_idx]
test_set = X[test_set_idx]
test_tags = y[test_set_idx]


def normalize1(a, axis=0, order=2):
    avg = np.average(a, axis)
    a -= np.expand_dims(avg)
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1

    return a / np.expand_dims(l2, axis)


def normalize2(a, axis=0):
    a -= np.expand_dims(np.min(a, axis), axis)
    a = a / np.expand_dims(np.max(a, axis), axis)
    return a * 2 - 1


train_set = normalize2(train_set)
net = neural.NeuralNet(100, 'relu')
error_rate = net.train_model(train_set, train_tags, learning_rate=0.1)
print(error_rate)





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
