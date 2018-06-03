#!/usr/bin/python
import numpy
import random

sqdist = lambda x1, x2: numpy.linalg.norm(x1 - x2)

def kmeans(samples, k, threshold=0.1):
    ## Init empty classes
    centers = []
    classes = numpy.zeros((k, len(samples)), dtype=numpy.bool)

    ## Randomize initial centers locations over x and y's range
    # Get range
    x = samples[...,0]
    y = samples[...,1]
    min_x, max_x = (min(x), max(x))
    min_y, max_y = (min(y), max(y))
    # Get random center location
    for i in range(k):
        center_x = min_x + max_x*random.random()
        center_y = min_y + max_y * random.random()
        centers.append(numpy.array((center_x, center_y)))

    ## Run the alrogirthm until we improve by less than given threshold
    # Set initial error values to make sure the first iteration is run
    error = 100*threshold
    prev_error = 2*error
    iterations = 0
    while (numpy.abs(prev_error - error) > threshold and error > threshold):
        # Associate samples to closest center
        for sample_index, sample in enumerate(samples):
            closest_class = numpy.argmin([sqdist(sample, c) for c in centers])
            classes[..., sample_index] = 0
            classes[closest_class, sample_index] = 1
        # Recalculate center of each class and the error
        prev_error = error
        error = 0
        for class_id in range(len(classes)):
            samples_in_class = samples[classes[class_id]]
            if numpy.any(samples_in_class):
                new_center = numpy.mean(samples_in_class, axis=0)
                centers[class_id] = new_center
            error_from_center = sum([ sqdist(s, centers[class_id]) for s in samples_in_class])
            error += error_from_center
        iterations += 1
    return classes, centers, error, iterations
