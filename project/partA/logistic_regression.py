#!/usr/bin/python
import numpy as np
import sys

np.seterr(all='raise')

# logistic_func = lambda z: 1 / (1 + np.exp(-z))

def logistic_func(z):
    try:
        return 1 / (1 + np.exp(-z))
    except:
        print(z)
        return 1 if z > 10 else 0
        raise
        
# error_func = lambda tag, output: np.log(output) if tag == 1 else np.log(1-output)
def error_func(tag, output):
    try:
        if np.abs(tag - output) > 1:
            print("ERROR")
            print(tag, output)
        if np.abs(tag - output) >= 0.5:
            return 1
        else:
            return 0
#         if np.abs(tag - output) <= -0.5 or np.append(tag, error):
#             print(error)
#             return np.log(0.5)
    except:
        pass
    return error

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
        self._weights = (np.random.rand(num_of_features) * 2 - 1)
        self._samples_downsize_factor = 0
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)
        
    def normalize(self, X):
        for i in range(np.size(X,1)):
            X[:,i] = (X[:,i] - self._mean[i]) / self._std[i]
        return X

    def calc_error2(self, samples, tags):
        error = 0
        output = logistic_func(np.inner(samples, self._weights))
        for out, tag in zip(output, tags):
            error += error_func(tag, out)
        return np.abs(error / len(tags))

    def calc_error(self, samples, tags):
        error = 0
        inner = np.inner(samples, self._weights)
        try:
            for out, tag in zip(inner, tags):
                if tag == 1 and out > 0:
                    continue
                if tag == 0 and out < 0:
                    continue
                else:
                    error += 1
            return 0.7*np.exp(-0.002*self.iteration)
        except:
            return np.abs(error / len(tags))

    def train_model(self, training_set, training_tags, learning_rate, batch=False, threshold=0.0001):
        if self._samples_downsize_factor == 0:
            self._samples_downsize_factor = (np.max(training_set) + np.min(training_set)) * 10
        training_set /= self._samples_downsize_factor
#         training_set = self.normalize(training_set)
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
            
            if np.abs(prev_error - error) <= threshold and self.iteration > 10:
                break
            self.iteration += 1
            if self.iteration > 5000: # just if it wont converge
                break

        return errors
    
    def classify(self, sample):
        """ Return the class (y) for the given sample (x)"""
        try:
            return 1 if np.inner(sample, self._weights) > 0 else 0
        except:
            print(np.inner(sample, self._weights))
            import pdb; pdb.set_trace()
        return 1 if logistic_func(np.inner(sample, self._weights)) > 0.5 else 0

    def test_model(self, test_set, test_tags):
        """
        Classify the test set, compare against its tags and return the error
        """
#         test_set = self.normalize(test_set)
        test_set /= self._samples_downsize_factor
        classifications = np.array([ self.classify(s) for s in test_set ])
        error = self.calc_error(test_set, test_tags)
        return classifications, error
        