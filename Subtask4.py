'''
Dataset 1
Train Accuracy: 86.625 %
Test Accuracy: 83.0 %
-------------------------
Dataset 2
Train Accuracy: 91.375 %
Test Accuracy: 91.0 %
'''
#Please enter the path to the datasets in the following variables
trainData_1_Path = 'ds1_train.csv'
testData_1_Path = 'ds1_test.csv'
trainData_2_Path = 'ds2_train.csv'
testData_2_Path = 'ds2_test.csv'

import numpy as np
import pandas as pd

class GDA:
    def __init__(self, trainPath, testPath):
        self.trainPath = trainPath
        self.testPath = testPath

#preprocessing training data
    def preProcess(self):
        df = pd.read_csv(self.trainPath)
        df_Y = df['y']
        df_X = df.drop('y', axis=1)

        X = df_X.values
        Y = df_Y.values

        return X, Y

#preprocessing test data
    def preProcessTest(self):
        df = pd.read_csv(self.testPath)
        df_Y = df['y']
        df_X = df.drop('y', axis=1)

        X = df_X.values
        Y = df_Y.values

        return X, Y

#computing mu0, mu1
    def mu(self):
        X, Y = self.preProcess()
        num_features=X.shape[1]

        mu0n=np.sum(X[Y==0],axis=0) # mu0 Numerator
        mu0d=np.sum([Y==0]) #mu0 Denominator
        mu0=(mu0n/mu0d).reshape(num_features,1) 

        mu1n=np.sum(X[Y==1],axis=0) # mu1 Numerator
        mu1d=np.sum([Y==1])  #mu1 Denominator
        mu1=(mu1n/mu1d).reshape(num_features,1)
      
        return mu0, mu1

#computing sigma
    def Sigma(self):
        X, Y = self.preProcess()
        mu0, mu1 = self.mu()
        num_features=X.shape[1]

        m=X.shape[0]
        Sigma=np.zeros((num_features,num_features))
        for i in range(m):
            xi = X[i,:].reshape(num_features,1)
            muyi = mu1 if Y[i]==1 else mu0 
            Sigma += (xi-muyi).dot((xi-muyi).T)
        Sigma = 1/m * Sigma

        return Sigma

#computing phi             
    def phi(self):
        X, Y = self.preProcess()
        mu1d=np.sum([Y==1])
        m=X.shape[0]
        
        return mu1d/m

#computing theta0, theta1
    def theta(self):
        Sigma = self.Sigma()
        mu0, mu1 = self.mu()
        phi = self.phi()

        S=np.linalg.inv(Sigma)
        theta12=S.dot(mu1-mu0).flatten()
        w1=mu0.T.dot(S.dot(mu0))
        w2=mu1.T.dot(S.dot(mu1))
        theta0=1/2*(w1-w2)[0,0]-np.log((1-phi)/phi)

        return theta12, theta0

    def sigmoid(self, z):
        s = 1/(1 + np.exp(-z))
        return s

#prediction and accuracy
    def trainAccuracy(self):
        X, Y = self.preProcess()
        theta, theta0 = self.theta()
        m=X.shape[0]
        Y_prediction = np.zeros((1, m))
        
        theta = np.reshape(theta, (1,2))
        theta = theta.T
        X = X.T

        yHat = self.sigmoid(np.dot(theta.T, X) + theta0)

        
        for i in range(yHat.shape[1]):
            Y_prediction[0, i] = 1 if yHat[0, i] > 0.5 else 0
        
        X, Y = self.preProcess()
        print("Train Accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction - Y)) * 100))
        return '---------------------------------------'

    def testAccuracy(self):
        X, Y = self.preProcessTest()
        theta, theta0 = self.theta()
        m=X.shape[0]
        Y_prediction = np.zeros((1, m))

        theta = np.reshape(theta, (1,2))
        theta = theta.T
        X = X.T

        yHat = self.sigmoid(np.dot(theta.T, X) + theta0)
        
        for i in range(yHat.shape[1]):
            Y_prediction[0, i] = 1 if yHat[0, i] > 0.5 else 0


        print("Test Accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction - Y)) * 100))
        return '---------------------------------------'

#printing parameters
    def print(self):
        Sigma = self.Sigma()
        mu0, mu1 = self.mu()
        phi = self.phi()
        theta, theta0 = self.theta()
        print('mu0 = ', mu0)
        print('mu1 = ', mu1)
        print('Sigma = ', Sigma)
        print('phi = ', phi)
        print("theta = ", theta)
        print("theta0 = ", theta0)
        print(self.trainAccuracy())
        print(self.testAccuracy())
        return '.'

#Fitting data
Dataset1 = GDA(trainData_1_Path, testData_1_Path)
Dataset2 = GDA(trainData_2_Path, testData_2_Path)

#printing the requried parameters
print('\t Dataset-1')
Dataset1.print()

print('\t Dataset-2')
Dataset2.print()




