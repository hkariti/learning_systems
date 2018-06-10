import numpy
import matplotlib.pyplot as plt

import samples
import kmeans

K = 2

def graph_kmeans(classes):
    k = len(classes)
    legend = []
    f = plt.figure()
    for i in range(k):
        selected = samples.X_2d[classes[i]]
        x = selected[...,0]
        y = selected[...,1]
        legend.append("Class {}".format(i+1))
        plt.scatter(x, y, s=3)
    plt.legend(legend)
    plt.title("K-mean with K={} in 2d".format(k))
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    return f

def run_kmeans_on_all():
    print("Running kmeans on all samples with K={0}".format(K))
    classes,_,_,_ = kmeans.kmeans(samples.X, K)
    print("Class 1: ", numpy.nonzero(classes[0])[0])
    print("Class 2: ", numpy.nonzero(classes[1])[0])

def run_kmeans_2d():
    print("Running kmeans in all samples in 2d with K={0}".format(K))
    classes,_,_,_ = kmeans.kmeans(samples.X_2d, K)

    plt_tagged = samples.graph_tagged_samples_2d()
    plt_kmeans = graph_kmeans(classes)

    new_k = 5
    print("Running kmeans in all samples in 2d with K={0}".format(new_k))
    classes,_,_,_ = kmeans.kmeans(samples.X_2d, new_k)
    plt_kmeans_new_k = graph_kmeans(classes)

    return plt_tagged, plt_kmeans, plt_kmeans_new_k

run_kmeans_on_all()
run_kmeans_2d()
