'''
Created on Oct 01, 2023
@author: Nick
'''



""" About
    -----------------------
    Linear model for binary classification
    
    Odds:
        odds can be expressed as p/(1-p) where p stands for the probability of the 
        positive evbent
        
    logit function:
        logarithm of the odds(log-odds)
        logit(p) = log([p/(1-p)]
           -log refers to the natural logarithm
           -logit function takes input values in the range 0 to 1 and 
            transforms them to values over the entire real-number range
            
        we can use this to express a linear relationship between 
        feature values and the log-odds
    
    logistic sigmoid function:
        - sometimes abbreviated to sigmoid function due to its characteristic "s" shape
         phi(z) = [(1)/(1+e^-z)
             z = transpose(vector_w) * (vector_x)
             - z is the net input, the linear combination of weights, and the inputs 
             (that is, the features asspcoated with the training examples)
        
        The activation function simply becomes the sigmoid function in logistic regression
        -output is the probability of something beloning to a class
        
    resources
    ------------------------
        natural regression for multiple classes
        ---------------------------------------
        https://sebastianraschka.com/pdf/lecture-notes/stat479ss19/L05_gradient_descent_slides.pdf
        http://rasbt.github.io/mlxtend/user_guide/classifier/SoftmaxRegression/
        
"""
from email._header_value_parser import MimeParameters
from xmlrpc.server import ExampleService
class LogisticRegressionGD(object):
    """ Logistic Regression Classifier using gradient descent/
    
    Parameters
    ----------------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter: int 
        Passes over the training dataset
    random_state : int
        Random number generator seed for random weight initialization
    
    Attributes
    ---------------------------
    w_ : 1d-array
        Weights after fitting
    cost_ : list
        Logistic cost function value in each epoch
    """
    
    def __init__(self, eta=0.5,n_iter=100,random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    def fit(self, X, y):
        """ Fit training data.
        
        Parameters
        ---------------------
        X: {array_like}, shape = [n_examples, n_features]]
            Training vectors, where n_examples is the number of examples
            and n_features is the number of features
        y: array-like, shape = [n_examples]
            Target values.
        
        Returns
        -------------------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01,
                              size = 1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * erros.sum()
            
            # note that we compute the logistic 'cost' now
            # instead of the sum opf squared errors cost
            
            cost = (-y.dot(np.log(1-output)) - 
                    ((1-y).dot(np.log(1-output))))
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def activation(self,z):
        """Compute logistic sigmoid activation"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1.0)
        #equivalent to:
        #return np.where(self.activation(self.net_input(X))
        #                    >= 0.5, 1.0
        
        
        # page 71 for math on the gradient descent learning 
        # algorithm for logistic regression