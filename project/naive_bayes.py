#!/usr/bin/python
import numpy
from scipy.stats import norm


class NaiveBayesModel:
    def __init__(self, classes):
        """
        Init the Naive Bayes Model with given set of classes. Classes should be a list of classes
        """
        self.classes = classes
        # For holding approximated P(y)
        self._class_probabilities = dict()
        # For P(x|y) we assume the following:
        #   - All components of a sample are indenpendent (i.e. naive bayes)
        #   - All components are normally distributed, so their distribution is defined by mean and variance
        # These two arrays will hold mean and variance for each component per class
        self._sample_mean_per_class = dict()
        self._sample_variance_per_class = dict()

    def train_model(self, training_set, training_tags):
        """ For each class, approximate P(y) and parameters for P(x|y) given the training set """
        for c in self.classes:
            mean_vec, variance_vec = self._mle_independent_normal_by_class(training_set, training_tags, c)
            class_prob = self._apriori_probability(training_tags, c)
            self._sample_mean_per_class[c] = mean_vec
            self._sample_variance_per_class[c] = variance_vec
            self._class_probabilities[c] = class_prob

    def classify(self, sample):
        """ Return the class (y) for the given sample (x), by finding the class with the highest probability P(y|x) """
        probabilities = { c: self._class_probability_given_sample(sample, c) for c in self.classes }
        
        return max(probabilities, key=(lambda k: probabilities[k]))

    def test_model(self, test_set, test_tags):
        """
        Classify the test set, compare against its tags and return the error
        Error is a simple sum of indicators: class != tag adds 1 to the error
        """
        classifications = numpy.array([ self.classify(s) for s in test_set ])
        error_vec = classifications != test_tags

        return error_vec.sum()

    def _mle_independent_normal_by_class(self, training_set, training_tags, classification):
        """ Approximate parameters of a normal distribution for independent components of the given sample set """
        selected_samples = training_set[training_tags == classification]
        # Simple mean for each coordinate
        mean_vec = numpy.mean(selected_samples, axis=0)
        # Caculate ||selected_samples - mean||^2 for each coordinate
        mean_mat = numpy.tile(mean_vec, (len(selected_samples), 1))
        distance_vec = selected_samples - mean_mat
        variance_vec = numpy.linalg.norm(distance_vec, axis=0)

        return mean_vec, variance_vec

    def _apriori_probability(self, training_tags, classification):
        """ Approximate the apriori probability of a given class """
        # Return a simple ratio of appearances to total samples of the given class
        total_count = len(training_tags)
        class_count = numpy.count_nonzero(training_tags == classification)
        class_prob = class_count / total_count

        return class_prob

    def _class_probability_given_sample(self, sample, classification):
        """
         Calculate P(x|y) for this sample, based on approximated P(x|y) dist from training.
         This probability is defined as the product of prob for all components (naive bayes)
        """
        mean_vec = self._sample_mean_per_class[classification]
        variance_vec = self._sample_variance_per_class[classification]
        prob_vec = norm.pdf(sample, loc=mean_vec, scale=variance_vec)
        prob_x_given_y = prob_vec.prod()

        # Get P(y), we have this from training too.
        prob_y = self._class_probabilities[classification]

        # Use Bayes rule to calculate P(y|x)
        # We actually calculate P(y|x)P(x) because P(x) doesn't matter for comparing classes because it's constant
        return prob_x_given_y*prob_y


