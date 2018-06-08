from __future__ import print_function
import time

import samples
import naive_bayes

classes = [0,1]
model = naive_bayes.NaiveBayesModel(classes)
training_set = samples.X[samples.training_set_idx]
training_tags = samples.y[samples.training_set_idx]
training_set_size = len(training_tags)
test_set = samples.X[samples.test_set_idx]
test_tags = samples.y[samples.test_set_idx]
test_set_size = len(test_tags)

### Run classification algorithm
start_time = time.time()
print("* Training Naive Bayes model")
model.train_model(training_set, training_tags)

print("* Testing model with Training set")
error = model.test_model(training_set, training_tags)
print("Total error: {0:.2f}%".format(100*error))

print("* Testing model with Test set")
error = model.test_model(test_set, test_tags)
print("Total error: {0:.2f}%".format(100*error))
end_time = time.time()
print("Total run time: {0:.2f}ms".format(1000*(end_time-start_time)))

### Compare empirical and approximated probability of first measurement in class 0
import scipy
import numpy
ts1=training_set[training_tags == 0][...,0]
mean1 = model._sample_mean_per_class[0][0]
variance1 = model._sample_variance_per_class[0][0]

unique_values, values_count = numpy.unique(ts1,return_counts=True)
empirical_probability = values_count/len(unique_values)
calculated_probability = scipy.stats.norm.pdf(unique_values, loc=mean1, scale=numpy.sqrt(variance1))
diff_probability = 100*(empirical_probability-calculated_probability)/empirical_probability

print("\n**** Probablity comparison for first measurement in class 0 ****")
print("Unique values (mean: {}):\n".format(mean1), unique_values)
print("Difference between empirical and calculated probability for each unique value, in pct relative to empirical_probability:\n", diff_probability)
