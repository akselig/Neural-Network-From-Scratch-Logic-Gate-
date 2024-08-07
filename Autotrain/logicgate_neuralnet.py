# All coding and research done by Anna Selig
# Main Code Completed by 8/5/2024

import numpy as np

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
    _, A1, _, A2 = forwardparams

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
    test_flat = test.flatten()
    indices = [i for i in range(x.shape[1]) if np.all(x[:, i] == test_flat)]
    if indices:
        expected = y[0, indices[0]]
    print(f"For input {[i[0] for i in test]}, output is {prediction}, expected: {expected}, {prediction==expected}")
    return (prediction == expected)

def main(training_type, training_set, params, learningrate=.37, iterations=10000): 
    # define input and output (x and y)
    global x, y
    losses = np.array([])

    if training_type == '2':
        if training_set == 'AND': 
            # AND dataset 
            x = np.array([[0,1,0,1],
                        [0,0,1,1]])

            y = np.array([[0,0,0,1]])
        if training_set == 'OR': 
            # OR dataset
            x = np.array([[0,1,0,1],
                        [0,0,1,1]])

            y = np.array([[0,1,1,1]])
        if training_set == 'NAND': 
            # NAND dataset
            x = np.array([[0,1,0,1],
                        [0,0,1,1]])

            y = np.array([[1,1,1,0]])
        if training_set == 'NOR': 
            # NOR dataset
            x = np.array([[0,1,0,1],
                        [0,0,1,1]])

            y = np.array([[1,0,0,0]])
        if training_set == 'XOR': 
            # XOR dataset
            x = np.array([[0,1,0,1],
                        [0,0,1,1]])

            y = np.array([[0,1,1,0]])
        if training_set == 'XNOR': 
            # XNOR dataset
            x = np.array([[0,1,0,1],
                        [0,0,1,1]])

            y = np.array([[1,0,0,1]])
    if training_type == '3': 
        if training_set == 'AND': 
            x = np.array([[0,0,0,0,1,1,1,1], 
                        [0,0,1,1,0,0,1,1], 
                        [0,1,0,1,0,1,0,1]])
                        
            y = np.array([[0,0,0,0,0,0,0,1]])
        if training_set == 'OR': 
            x = np.array([[0,0,0,0,1,1,1,1], 
                        [0,0,1,1,0,0,1,1], 
                        [0,1,0,1,0,1,0,1]])

            y = np.array([[0,1,1,1,1,1,1,1]])
        if training_set == 'NAND': 
            x = np.array([[0,0,0,0,1,1,1,1], 
                        [0,0,1,1,0,0,1,1], 
                        [0,1,0,1,0,1,0,1]])

            y = np.array([[1,1,1,1,1,1,1,0]])
        if training_set == 'NOR': 
            x = np.array([[0,0,0,0,1,1,1,1], 
                        [0,0,1,1,0,0,1,1], 
                        [0,1,0,1,0,1,0,1]])

            y = np.array([[1,0,0,0,0,0,0,0]])
        if training_set == 'XOR': # exclusive or, so just if ONE, not two or three are 1
            x = np.array([[0,0,0,0,1,1,1,1], 
                        [0,0,1,1,0,0,1,1], 
                        [0,1,0,1,0,1,0,1]])

            y = np.array([[0,1,1,0,1,0,0,0]])
        if training_set == 'XNOR': 
            x = np.array([[0,0,0,0,1,1,1,1], 
                        [0,0,1,1,0,0,1,1], 
                        [0,1,0,1,0,1,0,1]])

            y = np.array([[1,0,0,1,0,1,1,0]])
    if training_type == '4': 
        if training_set == 'AND': 
            x = np.array([[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
                        [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
                        [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                        [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]])

            y = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])
        if training_set == 'OR': 
            x = np.array([[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
                        [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
                        [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                        [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]])

            y = np.array([[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
        if training_set == 'NAND': 
            x = np.array([[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
                        [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
                        [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                        [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]])

            y = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]])
        if training_set == 'NOR': 
            x = np.array([[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
                        [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
                        [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                        [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]])

            y = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
        if training_set == 'XOR': 
            x = np.array([[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
                        [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
                        [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                        [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]])

            y = np.array([[0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0]])
        if training_set == 'XNOR': 
            x = np.array([[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
                        [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
                        [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                        [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]])

            y = np.array([[1,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1]])
    if training_type == '5': 
        if training_set == 'AND':
            x = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                        [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
                        [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
                        [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                        [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]])

            y = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])
        
        if training_set == 'OR':
            x = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                        [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
                        [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
                        [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                        [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]])

            y = np.array([[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
        
        if training_set == 'NAND':
            x = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                        [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
                        [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
                        [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                        [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]])

            y = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]])
        
        if training_set == 'NOR':
            x = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                        [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
                        [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
                        [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                        [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]])

            y = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
        
        if training_set == 'XOR':
            x = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                        [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
                        [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
                        [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                        [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]])

            y = np.array([[0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
        
        if training_set == 'XNOR':
            x = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                        [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
                        [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
                        [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                        [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]])

            y = np.array([[1,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
        
    m = x.shape[1] # number of columns (4)
    features = x.shape[0] #number of rows (number of inputs/features)

    w1, w2 = params # extracting weights from the array of weights
    starting_weights = np.append(w1, w2.reshape(-1,1), axis=1)


    for i in range(iterations): # training for the set number of iterations
        forwardparams = forward_prop(params,x)
        Z1, A1, Z2, A2 = forwardparams
        loss = binary_cross_entropy_loss(A2, y)
        losses = np.append(losses, loss)
        dZ2, dW2, dZ1, dW1 = backward_prop(m, params, forwardparams, y)
        w2 -= learningrate * dW2
        w1 -= learningrate * dW1
        params = (w1, w2)

        if i%500==0: # can be changed (the 500) if you want to see the updated loss more or less often (say, every 1000 iterations vs every 100)
            print(f"Iteration: {i}, Loss: {loss}")
            
    print("\nFinal weights after training:")
    print(params[0])  
    print(params[1])
    ending_weights = np.append(params[0], params[1].reshape(-1,1), axis=1)

    # Testing the predictions
    if training_type == '2':
        print("\nTwo-Input Logic Gates: ")
        print(f"Testing predictions for {training_set}:")
        val = all([
        predict(params, np.array([[0], [0]])),
        predict(params, np.array([[1], [0]])),
        predict(params, np.array([[0], [1]])),
        predict(params, np.array([[1], [1]]))])
    if training_type == '3': 
        print("\nThree-Input Logic Gates: ")
        print(f"Testing predictions for {training_set}:")
        val = all([
        predict(params, np.array([[0], [0], [0]])),
        predict(params, np.array([[0], [0], [1]])),
        predict(params, np.array([[0], [1], [0]])),
        predict(params, np.array([[0], [1], [1]])),
        predict(params, np.array([[1], [0], [0]])),
        predict(params, np.array([[1], [0], [1]])),
        predict(params, np.array([[1], [1], [0]])),
        predict(params, np.array([[1], [1], [1]]))])
    if training_type == '4':   
        print('\nFour-Input Logic Gates: ')
        print(f'Testing predictions for {training_set}: ')
        val = all([
        predict(params, np.array([[0],[0],[0],[0]])),
        predict(params, np.array([[0],[0],[0],[1]])),
        predict(params, np.array([[0],[0],[1],[0]])),
        predict(params, np.array([[0],[0],[1],[1]])),
        predict(params, np.array([[0],[1],[0],[0]])),
        predict(params, np.array([[0],[1],[0],[1]])),
        predict(params, np.array([[0],[1],[1],[0]])),
        predict(params, np.array([[0],[1],[1],[1]])),
        predict(params, np.array([[1],[0],[0],[0]])),
        predict(params, np.array([[1],[0],[0],[1]])),
        predict(params, np.array([[1],[0],[1],[0]])),
        predict(params, np.array([[1],[0],[1],[1]])),
        predict(params, np.array([[1],[1],[0],[0]])),
        predict(params, np.array([[1],[1],[0],[1]])),
        predict(params, np.array([[1],[1],[1],[0]])),
        predict(params, np.array([[1],[1],[1],[1]]))])
    if training_type == '5': 
        print('\nFive-Input Logic Gates: ')
        print(f'Testing predictions for {training_set}: ')
        val = all([
        predict(params, np.array([[0],[0],[0],[0],[0]])),
        predict(params, np.array([[0],[0],[0],[0],[1]])),
        predict(params, np.array([[0],[0],[0],[1],[0]])),
        predict(params, np.array([[0],[0],[0],[1],[1]])),
        predict(params, np.array([[0],[0],[1],[0],[0]])),
        predict(params, np.array([[0],[0],[1],[0],[1]])),
        predict(params, np.array([[0],[0],[1],[1],[0]])),
        predict(params, np.array([[0],[0],[1],[1],[1]])),
        predict(params, np.array([[0],[1],[0],[0],[0]])),
        predict(params, np.array([[0],[1],[0],[0],[1]])),
        predict(params, np.array([[0],[1],[0],[1],[0]])),
        predict(params, np.array([[0],[1],[0],[1],[1]])),
        predict(params, np.array([[0],[1],[1],[0],[0]])),
        predict(params, np.array([[0],[1],[1],[0],[1]])),
        predict(params, np.array([[0],[1],[1],[1],[0]])),
        predict(params, np.array([[0],[1],[1],[1],[1]])),
        predict(params, np.array([[1],[0],[0],[0],[0]])),
        predict(params, np.array([[1],[0],[0],[0],[1]])),
        predict(params, np.array([[1],[0],[0],[1],[0]])),
        predict(params, np.array([[1],[0],[0],[1],[1]])),
        predict(params, np.array([[1],[0],[1],[0],[0]])),
        predict(params, np.array([[1],[0],[1],[0],[1]])),
        predict(params, np.array([[1],[0],[1],[1],[0]])),
        predict(params, np.array([[1],[0],[1],[1],[1]])),
        predict(params, np.array([[1],[1],[0],[0],[0]])),
        predict(params, np.array([[1],[1],[0],[0],[1]])),
        predict(params, np.array([[1],[1],[0],[1],[0]])),
        predict(params, np.array([[1],[1],[0],[1],[1]])),
        predict(params, np.array([[1],[1],[1],[0],[0]])),
        predict(params, np.array([[1],[1],[1],[0],[1]])),
        predict(params, np.array([[1],[1],[1],[1],[0]])),
        predict(params, np.array([[1],[1],[1],[1],[1]]))])
    
    print(f'\nAll Predictions: {val}')

    return val, np.array([[training_set], [training_type], [starting_weights], [ending_weights], [losses]], dtype=object)
