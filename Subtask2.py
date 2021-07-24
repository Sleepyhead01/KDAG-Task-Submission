''' 
Dataset 1
train accuracy: 88.5 %
test accuracy: 91.0 %
Dataset 2
train accuracy: 91.125 %
test accuracy: 92.0 %
'''

#Please enter the path to the datasets in the following variables
trainDataset1Path = 'ds1_train.csv'
testDataset1Path = 'ds1_test.csv'
trainDataset2Path = 'ds2_train.csv'
testDataset2Path = 'ds2_test.csv'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Preprocessing data
train1 = pd.read_csv(trainDataset1Path)
Y_train1 = train1['y']
X_train1 = train1.drop('y', axis = 1)

test1 = pd.read_csv(testDataset1Path)
Y_test1 = test1['y']
X_test1 = test1.drop('y', axis = 1)

Y_train1 = np.reshape(Y_train1.values,(Y_train1.shape[0],1))
Y_train1 = np.transpose(Y_train1)
X_train1 = X_train1.T

Y_test1 = np.reshape(Y_test1.values,(Y_test1.shape[0],1))
Y_test1 = np.transpose(Y_test1)
X_test1 = X_test1.T

train2 = pd.read_csv(trainDataset2Path)
Y_train2 = train2['y']
X_train2 = train2.drop('y', axis = 1)

test2 = pd.read_csv(testDataset2Path)
Y_test2 = test2['y']
X_test2 = test2.drop('y', axis = 1)

Y_train2 = np.reshape(Y_train2.values,(Y_train2.shape[0],1))
Y_train2 = np.transpose(Y_train2)
X_train2 = X_train2.T

Y_test2 = np.reshape(Y_test2.values,(Y_test2.shape[0],1))
Y_test2 = np.transpose(Y_test2)
X_test2 = X_test2.T

def sigmoid(z):
    s = 1/(1 + np.exp(-z))

    return s

#initializing weights and biases
def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0
    
    return w, b

#Forward and Back propagation
def propagate(w, b, X, Y):

    m = X.shape[1]
    
    #Forward Prop
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost
    
    # Back Prop
    dw = (1 / m) * np.dot(X, (A - Y).T)     #gradient of the loss with respect to w
    db = (1 / m) * np.sum(A - Y)        #gradient of the loss with respect to b

    cost = np.squeeze(cost)
    
    #storing the gradients as a dictionary
    grads = {"dw": dw, "db": db}
    
    return grads, cost

#optimizing w and b by using gradient descent
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
   
    costs = []
    
    # number of iterations of the optimization loop
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        # W and b Update
        w = w - learning_rate * dw 
        b = b - learning_rate * db
        
        # costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
    
    params = {"w": w, "b": b}
    
    grads = {"dw": dw, "db": db}
    
    return params, grads, costs

#prediction
def predict(w, b, X):
  
    m = X.shape[1]
    Y_prediction = np.zeros((1, m)) #initializing prediction vector
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    
    return Y_prediction

#defining the Linear Regression Model
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
   
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train dataset
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

#Fitting the model
num_iterations1 = 40000
d1 = model(X_train1, Y_train1, X_test1, Y_test1, num_iterations1 , learning_rate = 0.0024, print_cost = False)

num_iterations2 = 50000
d2 = model(X_train2, Y_train2, X_test2, Y_test2, num_iterations2 , learning_rate = 0.0035, print_cost = False)

#Plots
iterate = np.arange(0, num_iterations1, 100)
cost_plot = d1['costs']

plt.plot(iterate, cost_plot)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d1["learning_rate"]))
plt.show()


iterate = np.arange(0, num_iterations2, 100)
cost_plot = d2['costs']

plt.plot(iterate, cost_plot)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d2["learning_rate"]))
plt.show()