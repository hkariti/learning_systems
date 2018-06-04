#!/usr/bin/python
import numpy

def mle_independent_normal(samples):
    # Simple mean for each coordinate
    mean = numpy.mean(samples, axis=0)
    # Caculate ||samples - mean|| for each coordinate
    mean_mat = numpy.tile(mean, (len(samples), 1))
    distance_vec = samples - mean_mat
    variance = numpy.linalg.norm(distance_vec, axis=0)

    return mean, variance
