'''
Created on Sep 28, 2023

@author: Nick
'''
from idlelib.idle_test.test_help_about import About
from email._header_value_parser import MimeParameters
from xml.sax.expatreader import AttributesImpl
from aifc import data
from lib2to3.tests.data.infinite_recursion import off_t
from ctypes.test.test_win32 import ReturnStructSizesTestCase
from pip._internal import self_outdated_check



""" 
    About
    ---------------------
    A popular alternative to the batch gradient descent algorithm is stochastic gradient descent (SGD)
    SGD is sometimes also called iterative or online gradient descent.
    Instead of updating the weights based on the sum of the accumulated errors over all
    training examples, x^i
    
    SGD can be considered as an approximation of gradient descent
    SGD typically reaches convergence much faster because of the more frequent
    weight updates
    
    The error is noisier than in gradient descent because each gradient is
    based on a single training. This is an advantage.  SGD can escape shallow local minima
    more readily if we are working with nonlinear cost functions
    
    It is important to present training data in a random order
    We also want to shuffle the training dataset for every epoch to prevent cycles
    
    SGD can be used with online learning.
    Online learning is when the model is trianed on the fly as new training data arrives
    In online learning, we can immediately discard the training data after updating the model
    if storage space is an issue.
    
    An inbetween of SGD and GD is mini-batch-learning.
    Mini-batch learningcan be understood as applying 
    batch gradient descent to smaller subsets of the training data
    
    The frequent weight updates in mini-batch allow us to reach convergence even faster.
    We can also use vectorized variables in mini-batch

"""

class AdalineSGD(object):
    """ADAptive LInear NEuron classifier.
    
    Parameters
    ----------------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    shuffle : bool (default: True)
        Shuffles the training data every epoch if True to prevent cycles.
    random_state : int
        Random number generator seed for random weight initialization
    
    
    Attributes
    ------------------------
    w_ : 1d-array
        Weights after fitting
    cost_ : list
        Sum-of-squares cost function value averaged over all 
        training examples in each epoch. 
    """
    
    def __init__(self, eta=0.01, n_iter=10,
                 shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
    
    def fit(self, X, y):
        """ Fit training data
        
        Parameters
        ---------------
        X: {array-like}, shape = [n_examples, n_features]
           Training vectors, where n_examples is the number of
           examples and n_features is the number of features
        y: array-like, shape = [n_examples]
           Target values.
           
        Return
        ----------------
        self : object
        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X , y = self._shuffle(X,y)
                cost = []
                for xi, target in zip(X,y):
                    cost.append(self._update_weights(xi,target))
                avg_cost = sum(cost) / len(y)
                self.cost_.append(avg_cost)
        return self
    
    
    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_intialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X,y):
                self._update_weights(xi,target)
        else:
            self._update_weights(X,y)
        return self
    
    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01,
                                   size=1 + m)
        self.w_initialized = True
    
    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 *error**2
        return cost
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """Compute linear activation"""
        return X
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X))
                        >= 0.0, 1, -1)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    