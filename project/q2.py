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
