import numpy as np

# define input and output (x and y)
training_set = 'XNOR' # set to any of the below to select training dataset
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
learningrate = .37 # learning rate cannot be optimal for every single set of randomly initialized weights 
                   # unless you use adaptive learning rate (think: adam)
                   # .37

iterations = 10000

# np.random.seed(1) # can be un-commented to get the same set of weights and therefore same outcome every single time.
# working seed for AND is 1, OR is 1, NAND is 3, NOR is 1, XOR is 4, XNOR is 14
    # note that these aren't the only seeds that work. or and nor are pretty flexible and "easy" to learn
    # so a bunch of different seeds work. Even "harder" to learn functions like XOR and XNOR will have multiple seeds that work
 
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

params = initialize_params(2, 2, 1) # input shape of 2, hidden layer with 2 neurons, output layer with 1 (binary classification)
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

# examples of successful final weights
'''
AND 
    w1: [[ 9.03681566 -4.43003342]
        [ 0.3929085  -1.59067341]]
    w2: [[  9.99783045 -24.37007623]]

OR 
    w1: [[ 3.60299717  3.8094532 ]
        [-4.35421705 -4.24539024]]
    w2: [[  6.77110175 -16.75731215]]

NAND 
    w1: [[ 9.33405565 -4.58114562]
        [ 0.39993752 -1.59139508]]
    w2: [[-10.07245212  24.47689651]]

NOR 
    w1: [[ 3.95055149  4.00879527]
        [-4.21894159 -4.19884405]]
    w2: [[-6.70376844 16.5785729 ]]

XOR 
    w1: [[0.85785772 0.85781279]
        [7.55004596 7.52900852]]
    w2: [[-31.02168063  24.3113322 ]]

XNOR 
    w1: [[0.85744396 0.85746351]
        [7.51518311 7.52416033]]
    w2: [[ 30.92178227 -24.23126075]]
'''