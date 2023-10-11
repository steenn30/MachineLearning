'''
Created on Oct 10, 2023

@author: Nick
'''

'''
About
------------------
    Retrieve a dataset for breast cancer
    
    
'''
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

'''

    We will start by readiung in the dataset directly from the UCI website using pandas:

'''
df = pd.read_csv('https://archive.ics.uci.edu/m1/'
                 'machine-learning-databases'
                 '/breast-cancer-wisconsin/wdbc.data',
                 header = None)


'''
    Next, we will assign the 30 featuyres to a NumPy array, x. Using 
    a LabelEncoder object, we will transform the class labels from
    their original string representation ('M' and 'B') into integers

'''

X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
le.classes_


'''
    After encoding the class labels (diagnossi in an array, y, the malignant
    tumors are now represented as class 1, and the benign tumors are
    represented as class 0, respectivley. WE can double-check this mapping by
    calling the transform method of the fitted LabelEncoder on two dummy
    class labels:
'''

le.transform(['M', 'B'])
    
    
'''
    Before we construct our first model pipeline in the following subsection, let's
    divide the dataset into a separate training dataset (80 percent of the data) and
    a separate test dataset (20 percent of the data)
'''

X_train, X_test, y_train, y_test = \
    train_test_split(X, y,
                     test_size = 0.20,
                     stratify=y,
                     random_state=1)