'''
Created on Sep 28, 2023

@author: Nick
'''

from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
'''About
        We defined k_features paramter to specify the desiored number of features we want to return
        
        we use accuracy_score from scikit-learn to evaluate the performance of a model
        (an estimator for classification) on the feature subsets
        
        Inside the while loop of the fit method, the feature subsets
        cerated by the itertools.combination function are evaluated and reduced
        until the feature subset has the desired dimensionality. In each 
        iteration, the accuracy score of the best subset is collected in a list,
        self_scores_, based on the internally created test dataset,
        X_test.
        
        We will use those scores later to evaluate the reuslts. The column
        indices of the final feature subset are assigned to self.indices_, 
        which we can use via the transform method to return a new data array with the selcted
        feature columns
        
        Note that, instead of acaluclating the criterion explicityly inside the fit 
        method, we simply removed the feature that is not contained in the best 
        performing feature subset
        
        SBS collects the scores of the best feature subset 
        at each stage
        
        TIDBITS:
            test dataset can also be valled validation dataset

'''


class SBS():
    def __init__ (self, estimator, k_features,
                    scoring=accuracy_score,
                    test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
        
    def fit(selfself, X, y):
        X_train, X_test, y_train, y_test = \
            train_test_split(X,y, test_size = self.test_size,
                             random_state = self.random_state)
            
        dim = X_train.shape[1]
        self.indices = tuple(range(dim))
        self.subsets = [self.indices_]
        score = self._calc_score(X_train, y_train,    
                                X_test, y_test, self.indices_)
        self.scores_ = [score]
        
        while dim > self.k_features:
            scores = []
            subsets = []
            
            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
                
            best = np.argmax(scores)
            self.indices = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        return self
    
    def transform(self, X):
        return X[:, self.indices_]
    
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:,indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
    
    
    
    