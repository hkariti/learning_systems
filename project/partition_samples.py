#!/usr/bin/python
import numpy
import scipy.io
import random

bmat = scipy.io.loadmat("BreastCancerData.mat")

test_set = []
training_set = []
samples_count = numpy.size(bmat['X'], 1)
for i in range(samples_count):
    if random.random() > 0.8:
        test_set.append(i)
    else:
        training_set.append(i)

print(test_set)
print(training_set)
