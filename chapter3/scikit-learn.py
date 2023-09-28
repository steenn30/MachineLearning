'''
Created on Sep 28, 2023

@author: Nick
'''


""" About
    ---------------------------------------

    The five main steps that are involved in training a 
    supervised machine learning algorithm can be summarized
    as follows:
        
        1. Selecting features and collecting labeled training exmamples
        2. Choosing a performance metric]
        3. Choosing a classifier and optimization algorithm
        4. Evaluating the performance of the model
        5. Tuning the algorithm
        
    Stratification means that the train_test_split method returns training and test 
    subsets that have the same proportions of class labels as the input datasets
        
        
    This Module
    ---------------------------------------
    -  x : feature matrix - pedal length and petal width
    -  y  : vector array -  class labels of species

"""
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

iris = datasets.load_iris
X = iris.data[:, [2,3]]
y = iris.target
print('Class labels:' , np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state = 1, stratify=y)

''' standardize features
'''

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

'''
    algorithms in scikit-learn already support multicalss classification by default via the one-vs.-rest (OvR) method

    random_state ensures reproducibility of the initial shuffling of the training dataset after each epoch
'''


ppn = Perceptron(eta0=0.1, random_state = 1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Misclassified examples: %d' % (y_test != y_pred).sum())




'''
    
    classification accuracy of a model is calculated simply as:
        1-error

    For performance metrics use
        from sklearn.metric import accuracy_score among others
        
        
    Each classifier in scikit-learn has a score method, which computes
    a classifier's prediction accuracy by combining the predict call
    with accuracy_score
    
    
    
    Overfitting is when the model captures the patterns in the training data well 
    butfails to generalize well to unseen data
'''








