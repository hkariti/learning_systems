import numpy as np

class NeuralNet:
    def ReLU(x):
        return x * (x>0)
    
    def tanh(x):
        return np.tanh(x/2)

    def log(x):
        return 1/(1+np.exp(-x))

    def ReLU_d(x):
        return 1*(x > 0)

    def tanh_d(x):
        return 0.5*(1-tanh(x))*(1+tanh(x))

    def log_d(x):
        return log(x)*(1-log(x))

    def logstic_regression(v_vector):
        return numpy.exp(x)/numpy.sum(numpy.exp(x))

    FUNCS = dict(ReLU=ReLU, log=log, tanh=tanh)
    DERIVATIVES = dict(ReLU=ReLU_d, log=log_d, tanh=tanh_d)

    def __init__(self, middle_layer_size, activation_func, input_layer_size=30):
        """
        Initialize a NN with 1 hidden layer of the given size and activation function.

        The input and output layers have a default size of 30 features and 1 binary neuron,
        respectively.
        The activation function for the hidden layer can be one of these strings:
            ReLU, log, tanh
        The activation function for the output layer is hardcoded to logistic regression
        The weights are intialized to zeroes. Customize obj.weights and obj.biases afterwards.
        """

        self.input_layer_size = input_layer_size
        self.middle_layer_size = middle_layer_size
        self.output_layer_size = 1
        self.activation_funcs = [self.FUNCS[activation_func], NeuralNet.log]
        self.derivative_func = self.DERIVATIVES[activation_func]

        # Init wights and biases.
        # In each weight matrix, each row is a neuron in the receiving layer.
        # Biases is a list of vectors for each neuron layer
        self.weights = []
        self.biases = []
        # Init weights for input->hidden
        self.weights.append(np.zeros((middle_layer_size, input_layer_size)))
        self.biases.append(np.zeros(middle_layer_size))
        # Init weights for hidden->output
        self.weights.append(np.zeros((self.output_layer_size, middle_layer_size)))
        self.biases.append(np.zeros(self.output_layer_size))

        # Init data structures for BP
        self._outputs = []
        self._inputs = []
        self._deltas = []

    def _feed_forward(self, samples):
        layer_count = len(self.weights)
        out = samples
        for layer in range(layer_count):
            weights = self.weights[layer]
            biases = self.biases[layer]
            activation_func = self.activation_funcs[layer]
            prev_out = out
            self._outputs.append(prev_out)
            v = weights.dot(prev_out) + biases
            self._inputs.append(v)
            out = activation_func(v)

        return out[0]

    def classify(self, samples):
        """
        Classify samples. samples should be a vector of size input_layer_size or a matrix of samples, with each sample being a column with input_layer_size rows.
        """
        return np.round(self._feed_forward(samples))

    def _cross_entropy(self, output, output_tags):
        return -np.sum(output_tags*np.log(output))

    def _calculate_gradient_batch(self, output, output_tags):
        deltas_out = output_tags - output
        deltas_out = deltas_out.reshape((len(deltas_out) ,1))
        deltas_hidden = self.derivative_func(self._inputs[0]).dot(deltas_out.dot(self.weights[1]))
        self._detlas = [deltas_hidden, deltas_out]

        d_weights_0 = -np.sum(deltas_hidden * self._outputs[0])
        d_weights_1 = -np.sum(deltas_out * self._outputs[1])

        return [d_weights_0, d_weights_1]

#    def train_model(self, training_set, training_tags, learning_rate=0.1, figure=None):
#
#    def test_model(self, test_set, test_tags):
