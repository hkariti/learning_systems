import samples
import logistic_regression as lr
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time

def lr_graph(lr_model, sets, tags):
    X_2d = samples._pca.fit_transform(sets)
    y, _ = lr_model.test_model(sets, tags)
    scatter_x = X_2d[...,0]
    scatter_y = X_2d[...,1]
    good_indices = y == 0
    bad_indices = numpy.invert(good_indices)
    plt.figure()
    plt.scatter(scatter_x[good_indices],scatter_y[good_indices], c='b', s=3)
    plt.scatter(scatter_x[bad_indices], scatter_y[bad_indices], c='r', s=3)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("Tagged samples in 2d")
    plt.legend(["Benign", "Malignant"])
    plt.show()

def normalize(X):
    for i in range(np.size(X,1)):
        X[:,i] = (X[:,i] - np.mean(X[:,i])) / np.std(X[:,i])
    return X

def add_bias_to_array(X):
    bias = np.ones((np.size(X, 0), 1))
    X = np.append(X, bias, axis=1)
    return X

training_set = samples.X[samples.training_set_idx]
training_tags = samples.y[samples.training_set_idx]
training_set_size = len(training_tags)
test_set = samples.X[samples.test_set_idx]
test_tags = samples.y[samples.test_set_idx]
test_set_size = len(test_tags)

cross_validation = scipy.io.loadmat("cross_validation_sets.mat")
cross_validation_sets_num = 10

num_of_features = np.size(training_set[1,])
start_time = time.time()

learning_rates = [0.05, 0.1, 0.15]
best_errors = np.zeros([len(learning_rates), cross_validation_sets_num])
num_of_iterations = np.zeros([len(learning_rates), cross_validation_sets_num])
for j, learning_rate in enumerate(learning_rates):
    min_error = np.inf
    print("Evaluating Learning rate: %f" % (learning_rate))
    for i in range(cross_validation_sets_num):
        print("Iteration %d" % (i))
        train_samples = cross_validation['train_samples_sets'][i]
        train_tags = cross_validation['train_tags_sets'][i]
        test_samples = cross_validation['test_samples_sets'][i]
        test_tags = cross_validation['test_tags_sets'][i]
        model = lr.LogisticRegressionModel(num_of_features, samples.X)
        train_errors = model.train_model(train_samples, train_tags, batch=False, learning_rate=learning_rate, threshold=0.0001)
        _, best_errors[j, i] = model.test_model(test_samples, test_tags)
        num_of_iterations[j, i] = len(train_errors)

for j, learning_rate in enumerate(learning_rates):
    print("Learning Rate %f: \n\tError-Rate mean:%f%%,  std:%f, \n\tIterations mean:%d,  std:%f" 
          % (learning_rate, 100*np.mean(best_errors[j]), 100 * np.std(best_errors[j]), np.mean(num_of_iterations[j]), np.std(num_of_iterations[j])))
plt.figure()
plt.errorbar(learning_rates, np.mean(best_errors,axis=1), xerr=0, yerr=np.std(best_errors,axis=1), fmt='X')
plt.title("Error vs Learning Rate")
plt.show() 


# Train Model with train set and test it with sequential & batch algos
chosen_learning_rate=0.1

model = lr.LogisticRegressionModel(num_of_features, samples.X)
seq_train_errors = model.train_model(training_set, training_tags, batch=False, learning_rate=chosen_learning_rate, threshold=0.00001)
print(seq_train_errors)
_, seq_test_errors = model.test_model(test_set, test_tags)
model = lr.LogisticRegressionModel(num_of_features, samples.X)
batch_train_errors = model.train_model(training_set, training_tags, batch=True, learning_rate=chosen_learning_rate, threshold=0.0001)
_, batch_test_errors = model.test_model(test_set, test_tags)

# model.test_model(test_set, test_tags)
# lr_graph(model, training_set, training_tags)
plt.figure()
plt.plot(seq_train_errors)
plt.plot(seq_test_errors)
plt.xlabel("Algo Iterations")
plt.ylabel("Error")
plt.title("Sequential Algo Errors")
plt.legend(["Training Set", "Test Set"])
plt.show()
plt.figure()
plt.plot(batch_train_errors)
plt.plot(batch_test_errors)
plt.xlabel("Algo Iterations")
plt.ylabel("Error")
plt.title("Batch Algo Errors")
plt.legend(["Training Set", "Test Set"])
plt.show()

print("Sequential Algo Errors: \n\tTrain:\t\tError-Rate:%f%% \tIterations:%d" % (100*np.min(seq_train_errors), len(seq_train_errors)))
print("\tTest:\t\tError-Rate:%f%%" % (100*np.min(seq_test_errors)))
print("Batch Algo Errors: \n\tTrain:\t\tError-Rate: %f%% \tIterations: %d" 
        % (100*np.min(batch_train_errors), len(batch_train_errors)))
end_time = time.time() - start_time
print("End Time: %f" % (end_time))