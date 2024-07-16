# All coding and research done by Anna Selig
# Main Code Completed by 7/16/2024

import numpy as np

# define input and output (x and y)
training_set = 'AND' # set to any of the below to select training dataset
if training_set == 'AND': 
    # AND dataset 
    x = np.array([[0,1,0,1],[0,0,1,1]])
    y = np.array([[0,0,0,1]])
if training_set == 'OR': 
    # OR dataset
    x = np.array([[0,1,0,1],[0,0,1,1]])
    y = np.array([[0,1,1,1]])
if training_set == 'NAND': 
    # NAND dataset
    x = np.array([[0,1,0,1],[0,0,1,1]])
    y = np.array([[1,1,1,0]])
if training_set == 'NOR': 
    # NOR dataset
    x = np.array([[0,1,0,1],[0,0,1,1]])
    y = np.array([[1,0,0,0]])
if training_set == 'XOR': 
    # XOR dataset
    x = np.array([[0,1,0,1],[0,0,1,1]])
    y = np.array([[0,1,1,0]])
if training_set == 'XNOR': 
    # XNOR dataset
    x = np.array([[0,1,0,1],[0,0,1,1]])
    y = np.array([[1,0,0,1]])

m = x.shape[1] # number of columns (4)
learningrate = .37 # learning rate works fine for all of the values

iterations = 6000 #decreased from 10000 to 6000, still consistently learns all

# for reproduceability (same results, same weights every time), can use np.random.seed(0)

def initialize_params(features, hidden_size, output_size): # initializing weights
    w1 = np.random.randn(hidden_size, features)
    w2 = np.random.randn(output_size, hidden_size)

    print("w1: ", w1)
    print("\nw2: ", w2)

    return [w1, w2]

def sigmoid(Z): # first activation function 
    return (1/(1 + np.exp(-Z)))

def relu(x): # second activation function
    x_array = np.array(x)
    return np.maximum(0, x_array)

def forward_prop(params, x): # forward propagation
    w1, w2 = params
    Z1 = np.dot(w1, x)
    
    A1 = sigmoid(Z1)
    A1 = relu(A1)

    Z2 = np.dot(w2, A1)
    
    A2 = sigmoid(Z2)
    A2 = relu(A2)

    return [Z1, A1, Z2, A2]

def binary_cross_entropy_loss(output, expected): # calculating loss function
    m = expected.shape[1]
    loss = -1/m * np.sum(expected * np.log(output) + (1-expected) * np.log(1 - output))
    return loss

def backward_prop(m, params, forwardparams, y): # backward propagation function
    w1, w2 = params
    Z1, A1, Z2, A2 = forwardparams

    dZ2 = A2-y
    dW2 = np.dot(dZ2,A1.T) / m
    
    dZ1 = np.dot(w2.T,dZ2) * A1 * (1-A1)
    dW1 = np.dot(dZ1,x.T) / m
    dW1 = np.reshape(dW1,w1.shape)

    dW2 = np.reshape(dW2,w2.shape)    
    return [dZ2, dW2, dZ1, dW1]

def predict(finalweights, test): # forward propagation on inputs to receive the outputs ("predicts" using the final weights)
    _, _, _, A2 = forward_prop(finalweights, test)
    output = np.squeeze(A2) # removes "single-element dimensions"
    prediction = 1 if output >= 0.5 else 0
    print(f"For input {[i[0] for i in test]}, output is {prediction}")

params = initialize_params(2, 4, 1) # input shape of 2, hidden layer with 4 neurons, output layer with 1 (binary classification)
w1, w2 = params # extracting weights from the array of weights

for i in range(iterations): # training for the set number of iterations
    forwardparams = forward_prop(params,x)
    Z1, A1, Z2, A2 = forwardparams
    loss = binary_cross_entropy_loss(A2, y)
    dZ2, dW2, dZ1, dW1 = backward_prop(m, params, forwardparams, y)
    w2 -= learningrate * dW2
    w1 -= learningrate * dW1
    params = (w1, w2)

    if i%500==0: # can be changed (the 500) if you want to see the updated loss more or less often (say, every 1000 iterations vs every 100)
        print(f"Iteration: {i}, Loss: {loss}")
        #print("\nw1 updated: ", w1) # can be un-commented if you want to see either of the set of weights printed  
        #print("\nw2 updated: ", w2) # every 500 iterations

print("\nFinal weights after training:")
print(params[0])
print(params[1])

# Testing the predictions
print(f"\nTesting predictions for {training_set}:")
predict(params, np.array([[0], [0]]))
predict(params, np.array([[1], [0]]))
predict(params, np.array([[0], [1]]))
predict(params, np.array([[1], [1]]))

# expected outputs for each logic gate
'''
AND        OR          NAND        NOR         XOR         XNOR
0, 0 -> 0 | 0, 0 -> 0 | 0, 0 -> 1 | 0, 0 -> 1 | 0, 0 -> 0 | 0, 0 -> 1
1, 0 -> 0 | 1, 0 -> 1 | 1, 0 -> 1 | 1, 0 -> 0 | 1, 0 -> 1 | 1, 0 -> 0
0, 1 -> 0 | 0, 1 -> 1 | 0, 1 -> 1 | 0, 1 -> 0 | 0, 1 -> 1 | 0, 1 -> 0
1, 1 -> 1 | 1, 1 -> 1 | 1, 1 -> 0 | 1, 1 -> 0 | 1, 1 -> 0 | 1, 1 -> 1
'''

# examples of successful final weights (there's variation in the pattern of final weights that do work, but more than one works)
'''
AND 
    w1: [[ 2.59843188  0.17453925]
        [ 3.00352835 -7.04837167]
        [-2.92797684  0.08256068]
        [-5.99387477  2.06400045]]
    w2: [[  5.90022051 -11.20499078  -5.28646416  -7.58140952]]

OR 
    w1: [[ 3.20914282  3.25757711]
        [-2.99244764 -2.99603061]
        [ 2.65496964  1.6542933 ]
        [-3.2615063  -3.52798133]]
    w2: [[ 5.39059148 -7.95314822  2.10360264 -9.80285155]]

NAND 
    w1: [[-4.48539908  1.40441567]
        [-5.40532359  1.94145166]
        [-0.6496005   3.4009474 ]
        [ 2.29148542 -6.07865106]]
    w2: [[ 6.52966539  8.01987751 -6.36501548  9.15969094]]

NOR 
    w1: [[-2.75430944 -2.97833468]
        [ 3.24376191  3.3722118 ]
        [ 1.71939531  0.33313436]
        [-3.73589762 -3.67136826]]
    w2: [[ 6.96879536 -6.55275175 -0.80221899 10.62710435]]

XOR 
    w1: [[ 7.71800389 -3.59937036]
        [ 5.40629864  5.42690623]
        [-3.86583577 -4.02681265]
        [-3.64311386  7.81035048]]
    w2: [[-9.79647475 14.81594173 -5.39662909 -9.80688263]]

XNOR 
    w1: [[-8.02025372  3.95063663]
        [-4.10721323 -4.92053778]
        [ 4.20947226  4.75645473]
        [-3.29650273  7.42763748]]
    w2: [[-9.94598048 13.47561805 -5.83802546 10.95161919]]
'''
