import numpy as np

# Defination of Activation function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# To find slope of x, needed for error calculation
def sigmoid_derivative(x):
    return x * (1 - x)

# Input layer for Neural Network
training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

# Ground Truth for the given inputs
# Example input [0,0,1] gives output 0
# .T represent transpose of matrix
training_outputs = np.array([[0,1,1,0]]).T

# np.random.seed(0) makes the random numbers predictable
# Here we make it unpredictable
np.random.seed(1)

# This equation makes weight to be in between -1 and 1. Please try in pen paper if you don't get it
weights = 2 * np.random.random((3,1)) - 1

print ("Initial random weights in NN: ")
print (weights)

# 1000 iteration to try if it converges. You can change this and try
for iteration in range(1000):
    #input is given to first layer
    input_layer = training_inputs
    
    # Output layer is dot of input and weights and passed through activation function (Sigmoid in this case)
    output_layer = sigmoid(np.dot(input_layer,weights))
    
    #The difference in expected output and actual output
    error = training_outputs - output_layer
    
    #Adjustment for weights to backpropagate
    adjustment = error * sigmoid_derivative(output_layer)
    
    # New weights after adjustments
    weights += np.dot(input_layer.T, adjustment)
    
print ("weights after training: ")
print (weights)
    
print ("outputs after training: ")
print (output_layer)
