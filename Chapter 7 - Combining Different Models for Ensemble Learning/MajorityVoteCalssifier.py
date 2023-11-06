from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator
from _ast import In
from email._header_value_parser import MimeParameters
from ctypes.test.test_win32 import ReturnStructSizesTestCase
from builtins import object



class MajorityVoteClassifier(BaseEstimator,ClassifierMixin):
    """ A majority vote ensemble classifier
    
    Parameters
    -------------------------------
    classifiers: array-like, shape = [n_classifiers]
        Different classifiers for the ensemble
    vote: str, {'classlabel', 'probability'}
        Default: 'classlabel'
        If 'classlabel' the prediction is based ON
        the argmax of class labels. Else If
        'probability', the argmax of the sum of 
        probabilities is used to predict the class label
        (recommended for calibrated classifiers)
        
    weights: array-like, shape = [n_classifiers]
        Optional, default: None
        If a list of 'int' or 'float' values are
        provided, the classifiers are weighted by
        importance; Uses uniform weights if 'weights=None'.
    
    """
    
    def __init___(self, classifiers,
                  vote='classlable', weights=None):
        self.classifiers=classifiers
        self.named_classifiers= {key: value for 
                                 key, value in
                                 _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights
    
    def fit(self, X, y):
        
        """ Fit classifiers
        
        Parameters
        ------------------------
        X : {array-like, sparce matrix},
        shape=[n_examples, n_features]
        Matrix of training examples.
        
        y : array-like, shape = [n_examples]
            Vector of target class labels
            
        
        Returns
        ---------------------
        self : object
        
        """
        
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError("vote must be 'probability'"
                             "or 'classlabel'; got (vote=%r)"
                             % self.vote)
        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError("Number of classifiers and weights"
                             "must be equal; got %d weights,"
                             "%d classifiers"
                             % (len(self.weights),
                                len(self.classifiers)))
        
        # Use LabelEncoder to ensure class labels start
        # with 0, which is important for np.argmax
        # call in self.predict
        self.lablenc_ = LabelEncoder()
        self.labelnc_.fit(y)
        self.classes_ = self.labelnc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X,
                                        self.labelnc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self
    
    """
    More Notes
    ----------------------
    We will add the predict method to predict the class label via a majority vote
    based on the class labels if we initialize a new MajorityVoteClassifier object
    with vote='classlabel'. Alternatively, we wil be able to initialize the ensemble
    classifier with vote='probability to predict the class label based on the class
    membership probabilities. Furthermore, we will also add a predict_proba method
    to return the averaged probabiliteis, which is useful when computing the receiver
    operating characterisitic area under the curve (ROC AUC):
    """
    
    def predict(self, X):
        """ Predict class labels for X.
        
        Parameters
        -----------------------
        X : {array-like, sparse matrix},
            Shape = [n_examples, n_features]
            Matrix of training examples
        
        Returns
        ------------------------
        maj_vote : array-like, shape = [n_examples]
            Predicted class labels.
            
        """
        
        if self.vote== 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else: #'classlabel' vote
            
            #Collect results from clf.predict calls
            predictions = np.asarray([clf.predict(X)
                                      for clf in self.classifiers_]).T
                                
                                      
            maj_vote = np.apply_along_axis(lambda x: np.argmax(
                                        np.bincount(x,
                                                    weights=self.weights)),
                                                    axis=1,
                                                    arr=predicitons)
            maj_vote = self.lablenc_.inverse_transform(maj_vote)
            return maj_vote
    
    def predict_proba(self, X):
        """ Predict class probabilities for X.
        
        Parameters
        -----------------------
        X : {array-like, sparse matrix},
            shape = [n_examples, n_features]
            Training vectors, where
            n_examples is the number of examples and
            n_features is the number of features.
        
        Returns
        -----------------------------
        avg_proba : array-like,
                    shape=[n_examples, n_classes]
                    Weighted average probability for
                    each class per example.
        
        """
        
        probs = np.asarray([clf.predict_proba(X)
                            for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0,
                               weights=self.weights)
        return avg_proba
    
    
    def get_params(self, deep=True) :
        """ Get classifier parameters names for GridSearch"""
        """
            This is our own modified version of get_params.
            
            This is so we can use _name_estimators function to access the parameters
            of individual classifiers in the ensemble; this may look a little bit 
            complicated at first, but it will make perfect sense when we use 
            gird search for hyperparameter tuning in later sections
        """
        if not deep:
            return super(MajorityVoteClassifier,
                         self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items(): 
                for key, value in step.get_params(
                          deep=True). items():
                    out['%s__%s'% (name,key)] = value
            return out
        
        
        
        