#!/usr/bin/python
import numpy
import random

sqdist = lambda x1, x2: numpy.linalg.norm(x1 - x2)

def kmeans(samples, k, threshold=0.1):
    ## Init empty classes
    centers = []
    classes = numpy.zeros((k, len(samples)), dtype=numpy.bool)

    ## Randomize initial centers locations over x and y's range
    range_max = numpy.max(samples, axis=0)
    range_min = numpy.min(samples, axis=0)
    # Get random center location
    for i in range(k):
        random_coords = numpy.array([ random.random() for i in range(len(range_max)) ])
        random_coords = range_max*random_coords + range_min
        centers.append(random_coords)

    ## Run the alrogirthm until we improve by less than given threshold
    # Set high initial error values to make sure the first iteration is run
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
