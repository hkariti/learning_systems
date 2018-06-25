import numpy as np

class NeuralNet:
    def ReLU(x):
        return x * (x>0)

    def tanh(x):
        return np.tanh(x/2)

    def ReLU_d(x):
        return 1*(x > 0)

    def tanh_d(x):
        return 0.5*(1-NeuralNet.tanh(x))*(1+NeuralNet.tanh(x))

    def log(x):
        log_result = -np.logaddexp(0, -x)
        result = np.exp(log_result)
        return result

    def log_d(x):
        return NeuralNet.log(x)*(1-NeuralNet.log(x))

    FUNCS = dict(relu=ReLU, log=log, tanh=tanh)
    DERIVATIVES = dict(relu=ReLU_d, log=log_d, tanh=tanh_d)

    def __init__(self, middle_layer_size, activation_func, input_layer_size=30):
        """
        Initialize a NN with 1 hidden layer of the given size and activation function.

        The input and output layers have a default size of 30 features and 1 binary neuron,
        respectively.
        The activation function for the hidden layer can be one of these strings:
            relu, log, tanh
        The activation function for the output layer is hardcoded to log (logistic regression)
        The weights are intialized to zeroes. Customize obj.weights afterwards.
            Note: The first column in each weight matrix represents the bias of the relevant neuron
        """

        self.input_layer_size = input_layer_size
        self.middle_layer_size = middle_layer_size
        self._input_layer_size_with_bias = input_layer_size + 1
        self._middle_layer_size_with_bias = middle_layer_size + 1
        self.output_layer_size = 1
        self._activation_funcs = [self.FUNCS[activation_func], NeuralNet.log]
        self._derivative_funcs = [self.DERIVATIVES[activation_func], NeuralNet.log_d]

        # Init wights and biases.
        # In each weight matrix, each row is a neuron in the receiving layer.
        self.weights = []
        # Init weights for input->hidden. we add an extra column for the bias of each neuron in this layer
        self.weights.append(np.random.random((middle_layer_size, self._input_layer_size_with_bias)))
        # Init weights for hidden->output. we add an extra column for the bias of each neuron in this layer
        self.weights.append(np.random.random((self.output_layer_size, self._middle_layer_size_with_bias)))

    def _feed_forward(self, samples):
        layer_count = len(self.weights)
        if samples.ndim == 1:
            samples = np.array([samples])
        # Init data structures for BP
        self._outputs = []
        self._inputs = []
        # Feed data into the net
        outs = []
        for sample in samples:
            outputs = []
            inputs = []
            out = sample
            for layer in range(layer_count):
                weights = self.weights[layer]
                activation_func = self._activation_funcs[layer]
                prev_out = out
                # Add the constant neuron, representing bias, to the previous layer's output
                prev_out = np.insert(prev_out, 0, 1, axis=0)
                outputs.append(prev_out)
                v = weights.dot(prev_out)
                inputs.append(v)
                out = activation_func(v)
            self._outputs.append(outputs)
            self._inputs.append(inputs)
            outs.append(out)

        return np.array(outs)

    def classify(self, samples):
        """
        Classify samples. samples should be a vector of size input_layer_size or a matrix of samples, with each sample being a row with input_layer_size columns.
        """
        return np.round(self._feed_forward(samples))

    def _error(self, output, output_tags):
        # Use square error
        return np.sum((output_tags-output[...,0])**2)

    def _calculate_gradient_serial(self, output, output_tags):
        # Calculate the gradient for each layer's weight matrix, using the serial version of the BP algorithm.
        #   This could probably be done in a more generic way and much more efficiently using
        #   matrices properly, but matrices confuse me so for loops it is.

        # Init values
        d_weights_1 = np.zeros((self.output_layer_size, self._middle_layer_size_with_bias))
        d_weights_0 = np.zeros((self.middle_layer_size, self._input_layer_size_with_bias))
        # Select a sample at random and calculate its error's gradient
        sample_idx = np.random.randint(len(output))
        # Calculate delta for output layer
        deltas_out = np.zeros(self.output_layer_size)
        # Calculate gradient for each input, includng extra one for bias (outer for is redundant atm because we're hardcoded to 1 output neuron)
        for output_neuron in range(self.output_layer_size):
            deltas_out[output_neuron] = (output_tags[sample_idx] - output[sample_idx][output_neuron])*self._derivative_funcs[1](self._inputs[sample_idx][1][output_neuron])
            for hidden_neuron in range(self._middle_layer_size_with_bias):
                addition_from_current_sample = deltas_out[output_neuron] * self._outputs[sample_idx][1][hidden_neuron]
                d_weights_1[output_neuron, hidden_neuron] = -addition_from_current_sample

        # Calculate delta for each neuron in the hidden layer and gradient for each of their inputs (including extra one for bias)
        for neuron in range(self.middle_layer_size):
            delta = self._derivative_funcs[0](self._inputs[sample_idx][0][neuron])*np.sum(self.weights[1][:,neuron]*deltas_out)
            for input_neuron in range(self._input_layer_size_with_bias):
                d_weights_0[neuron, input_neuron] = - (delta * self._outputs[sample_idx][0][input_neuron])

        return [d_weights_0, d_weights_1]

    def train_model(self, training_set, training_tags, learning_rate=0.1, threshold=1e-4, max_iterations=1000000):
        """
        Train the neural net via the given training set at the given learning_rate

        Learning is complete after the total error or error improvement is beneath the given threshold.
        Will raise RuntimeError if learning didn't complete after max_iterations iterations.
        """
        output = self._feed_forward(training_set)
        current_error = self._error(output, training_tags)
        prev_error = 0
        learning_progress = [current_error]
        iterations = 0
        while (current_error > threshold and np.abs(current_error - prev_error) > threshold):
            iterations = iterations + 1
            if (iterations > max_iterations):
                raise RuntimeError("Didn't converge to threshold {} after {} iterations. Current error is {}".format(threshold, max_iterations, current_error))
            [gradient_0, gradient_1] = self._calculate_gradient_serial(output, training_tags)
            self.weights[0] = self.weights[0] - learning_rate * gradient_0
            self.weights[1] = self.weights[1] - learning_rate * gradient_1
            output = self._feed_forward(training_set)
            prev_error = current_error
            current_error = self._error(output, training_tags)
            learning_progress.append(current_error)

        return learning_progress

    def test_model(self, test_set, test_tags):
        """
        Test the model with the given test_set against the given tags and return the error.

        Error is calculated using square error.
        """
        output = self._feed_forward(test_set)
        return self._error(output, test_tags)
