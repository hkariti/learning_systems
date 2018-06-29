#!/usr/bin/python
import numpy as np
import sys

np.seterr(all='raise')

# logistic_func = lambda z: 1 / (1 + np.exp(-z))

def logistic_func(z):
    try:
        if np.isscalar(z):
            if z > 100:
                return 1
            if z < -100:
                return 0
            return 1 / (1 + np.exp(-z))
        else:
            log_reg = np.zeros(np.size(z))
            for i in range(len(z)):
                if z[i] > 600:
                    log_reg[i] = 1
                elif z[i] < -600:
                    log_reg[i] = 0
                else:
                    log_reg[i] =  1 / (1 + np.exp(-z[i]))
            return log_reg
    except:
        print(z)
        raise
        
def shuffle_set(training_set, training_tags):
    unified = np.zeros([np.size(training_set,0), np.size(training_set,1)+1])
    unified[:,0] = training_tags
    unified[:,1:] = training_set
    np.random.shuffle(unified)
    training_set = unified[:,1:]
    training_tags = unified[:,0]
    return training_set, training_tags

class LogisticRegressionModel:
    def __init__(self, num_of_features, X):
        """
        Init the Logistic Regression Model with number of features
        """
        self._num_of_features = num_of_features
        self._weights = (np.random.rand(num_of_features + 1) * 2 - 1) / num_of_features
        self._samples_downsize_factor = 0
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)
        
    def normalize(self, X):
        for i in range(np.size(X,1)):
            X[:,i] = (X[:,i] - self._mean[i]) / self._std[i]
        return X

    def calc_error(self, samples, tags):
        error = 0
        inner = np.inner(samples, self._weights)
        for out, tag in zip(inner, tags):
            if tag == 1 and out > 0:
                continue
            if tag == 0 and out < 0:
                continue
            else:
                error += 1
        return error / len(tags)
    
    def add_bias_to_array(self, t_set):
        bias = np.ones((np.size(t_set, 0), 1))
        t_set = np.append(t_set, bias, axis=1)
        return t_set

    def train_model(self, training_set, training_tags, learning_rate, batch=False, threshold=0.0001):
        if self._samples_downsize_factor == 0:
            self._samples_downsize_factor = (np.max(training_set) + np.min(training_set))
        training_set = self.normalize(training_set)
        training_set = self.add_bias_to_array(training_set)
        self.iteration = 0
        error = self.calc_error(training_set, training_tags)
        errors = [error]
        
        while True:
            prev_error = errors[self.iteration]
            training_set, training_tags = shuffle_set(training_set, training_tags)

            if batch:
                inner = np.inner(training_set, self._weights)
                output = logistic_func(inner)
                training_tags = training_tags.reshape(len(training_tags))
                weights_deltas = np.inner(training_set.transpose(), (training_tags - output))
                self._weights += weights_deltas * learning_rate
                
            else: # online
                for sample_index, (sample, label) in enumerate(zip(training_set, training_tags)):
                    output = logistic_func(np.inner(sample, self._weights))
                    weights_deltas = np.multiply((label - output), sample)
                    self._weights += weights_deltas * learning_rate
            error = self.calc_error(training_set, training_tags)
            errors.append(error)
            
            if np.abs(prev_error - error) <= threshold:
                break
            self.iteration += 1
            if self.iteration > 5000: # just if it wont converge
                break

        return errors
    
    def classify(self, sample):
        """ Return the class (y) for the given sample (x)"""
        return 1 if np.inner(sample, self._weights) > 0 else 0

    def test_model(self, test_set, test_tags):
        """
        Classify the test set, compare against its tags and return the error
        """
        test_set = self.normalize(test_set)
        test_set = self.add_bias_to_array(test_set)
        classifications = np.array([ self.classify(s) for s in test_set ])
        error = self.calc_error(test_set, test_tags)
        return classifications, error
