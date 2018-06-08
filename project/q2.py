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

print("* Training Naive Bayes model...", end="")
start_time = time.time()
model.train_model(training_set, training_tags)
end_time = time.time()
print("Done. Took {0:.2f}ms".format(1000*(end_time-start_time)))

print("* Testing model with Training set....", end="")
start_time = time.time()
error = model.test_model(training_set, training_tags)
end_time = time.time()
print("Done. Took {0:.2f}ms".format(1000*(end_time-start_time)))
error_pct = 100*error/training_set_size
print("Total error: {0} out of {1} samples. Error pct: {2:.2f}".format(error, training_set_size, error_pct))

print("* Testing model with Test set...", end="")
start_time = time.time()
error = model.test_model(test_set, test_tags)
end_time = time.time()
print("Done. Took {0:.2f}ms".format(1000*(end_time-start_time)))
error_pct = 100*error/test_set_size
print("Total error: {0} out of {1} samples. Error pct: {2:.2f}".format(error, test_set_size, error_pct))
