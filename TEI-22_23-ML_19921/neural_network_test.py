import numpy as np
#Used to Import Data
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

#Hidden Layers
class Layer_Dense:

    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def foward(self, inputs):
        #The inputs can be the X iputs or the outputs of the previous layer of neurons
        self.output = np.dot(inputs, self.weights) + self.biases


#Activation Function Rectified Linear Class
class Activation_ReLU:

    def foward(self, inputs):
        self.output = np.maximum(0, inputs)


#SoftMax Varaible Class -> Prevents from getting overflow values by changing them but not changing 
#their meaning
class Activation_Softmax:

    def foward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities



#Common Loss Class
class Loss:

    def calculate(self, output, y):
        sample_losses = self.foward(input, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):

    def foward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

X, y = spiral_data(samples=100, classes=3)

# First Number is the number of Features and the Second is the Number of Neurons
dense1 = Layer_Dense(2, 3)

activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.foward(X)
activation1.foward(dense1.output)

dense2.foward(activation1.output)
activation2.foward(dense2.output)

print(activation2.output[:5])

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

print(loss)
