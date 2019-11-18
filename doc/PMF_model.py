import pandas as pd
import numpy as np
from numpy.random import RandomState
import copy
from sklearn.metrics import mean_squared_error
from math import sqrt

class PMF():
    # initialize paprameters
    def __init__(self, m, n, lambda_u=0.3, lambda_v=0.3, latent_size=10,
                 lr=0.01, num_iter=2000, stopping_deriv=None, seed=None):
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.random_state = RandomState(seed)
        self.latent_size=latent_size
        self.lr = lr
        self.iterations = num_iter
        self.stopping_deriv = stopping_deriv
        self.R = np.zeros([n,m])
        self.I = None
        self.U = None
        self.V = None
        

    def loss(self):
        # loss function
        loss = 0.5*(np.sum(self.I*(self.R-np.dot(self.U.T, self.V))**2) + self.lambda_u*np.sum(np.square(self.U)) + self.lambda_v*np.sum(np.square(self.V)))
        return loss
    
    def predict(self, data):
        index = np.array([[int(element[0]-1), int(element[1]-1)] for element in data], dtype=int)
        u_features = self.U.take(index.take(0, axis=1), axis=1)
        v_features = self.V.take(index.take(1, axis=1), axis=1)
        preds = np.sum(u_features*v_features, 0)
        return preds

    def fit(self, train_data, validation_data = None):
        for element in train_data:
            self.R[int(element[0]-1), int(element[1]-1)] = float(element[2])
        
        self.I = copy.deepcopy(self.R)
        self.I[self.I != 0] = 1

        self.U = 0.1*self.random_state.rand(self.latent_size, np.size(self.R, 0))
        self.V = 0.1*self.random_state.rand(self.latent_size, np.size(self.R, 1))

        
        last_validation_rmse = None
        stop_u = False
        stop_v = False

        for it in range(self.iterations):
            # derivate of U
            grads_u = np.dot(self.I*(self.R-np.dot(self.U.T, self.V)), -self.V.T).T + self.lambda_u*self.U

            # derivate of V
            grads_v = np.dot((self.I*(self.R-np.dot(self.U.T, self.V))).T, -self.U.T).T + self.lambda_v*self.V

            # update the parameters
            if self.stopping_deriv is None:
                self.U = self.U - self.lr * grads_u
                self.V = self.V - self.lr * grads_v
            else:
                if np.all(abs(grads_u) > self.stopping_deriv):
                    self.U = self.U - self.lr * grads_u
                else:
                    stop_u = True

                if np.all(abs(grads_v) > self.stopping_deriv):
                    self.V = self.V - self.lr * grads_v
                else:
                    stop_v = True

                if stop_u and stop_v:
                    print('early stopping:{: d}'.format(it))
                    break

            # training loss
            train_loss = self.loss()
            
            if validation_data is None:
                if (it%100 == 0):
                    print('traning iteration:{: d}, loss:{: f}'.format(it, train_loss))
            else:
                validation_preds = self.predict(validation_data)
                validation_rmse = sqrt(mean_squared_error(validation_data[:,2], validation_preds))
                if (it%100 == 0):
                    print('traning iteration:{: d}, loss:{: f}, validation_rmse:{: f}'.format(it, train_loss, validation_rmse))

                if last_validation_rmse and (last_validation_rmse - validation_rmse) <= 0:
                    print('convergence at iterations:{: d}'.format(it))
                    break
                else:
                    last_validation_rmse = validation_rmse
        train_preds = self.predict(train_data)
        train_rmse = sqrt(mean_squared_error(train_data[:,2], train_preds))
        
        if validation_data is None:
            return self.U, self.V, train_rmse
        else:
            return self.U, self.V, train_rmse, last_validation_rmse