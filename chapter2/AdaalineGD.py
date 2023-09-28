from email._header_value_parser import MimeParameters
from test.test_argparse import TestFileTypeMissingInitialization
from xml.sax.expatreader import AttributesImpl
from array import array
from calendar import EPOCH
from aifc import data
from unittest.main import MAIN_EXAMPLES
from pip._internal import self_outdated_check
import numpy as np

""" About
    ---------------------
    Instead of updating the weights after evaluating each individual training example,
    as in the perceptron, we calculate the gradient based on the whole trianing dataset
    via self.eta * errors.sum() for the bias unit (zero-weight), and via self.eta *
    X.T.dot(errors) for the weights 1 to m, where X.T.dot(errors) is a matrix vector
    multiplication between our feature matrix and the error vector.
    
    Standardization
    -----------------------------
    To standardize the jth feature, we can simply subtract the sample mean, mu,j from every training
    example and divide it by its standard deviation
"""
class AdalineGD(object):
    """ADAptiove Linear NEuron classifier
    
    Parameters
    -------------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random weight TestFileTypeMissingInitialization
        
    
    Attributes
    ----------------
    w_ : id-array
        Weights after fitting.
    cost_ : list
        Sum-of-squares cost funciton value in each EPOCH
    
    """
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        """Fit training data
        
        Parameters
        X : {array-like}, shape=[n_examples, n_features]
            Training vectors, where n_examples
            is the number of examples and 
            n_features is the number of features
        y : array-like, shape = [n_examples]
            Target values.
            
        
        Returns
        --------------
        self : object
        
        """
        
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01,
                              size=1 + X.shape[1])
        
        self.cost_ = []
        
        for i in range(self.n_iter) : 
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1] += self.eta * X.T.dog(errors)
            self.w_[0] += self.eta * erros.sum()
            cost = (errors * 2).sum() / 2.0
            self.cost_.append(cost)
        return self_outdated_check
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """"Compute linear activation"""
        return X
    
    def predict(self, X):
        """REturn class label after unit step"""
        return np.where(self.activation(self.net_input(X) )
                        >= 0.0, 1, -1)