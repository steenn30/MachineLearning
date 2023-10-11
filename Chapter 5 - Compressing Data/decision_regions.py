from sklearn_linear_model import LogisticRegression
from sklearn.decomposition import PCA
from functools import lru_cache
# Initializing the PCA transformer and
# logistic regression estimator
pca = PCA(n_components = 2)
lr = LogisticRegression(multi_class = 'ovr',
                        random_state = 1,
                        solver = 'lbfgs')
# dimensionality reduction:
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
# fitting the logistic regression model on the reduced dataset
lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc = 'lower left')
plt.tight_layout()
plt.show()


'''
    When we compare PCA projections via scikit-learn with our own PCA
    implementation, it can happen that the resulting plots are mirror
    images of each other. Note that this is not due to an error in
    either of those two implementations;
    the reason for this difference is that, depending on the eigensolver,
    eigenvectors can have either negative or positive signs

'''


'''
     To flip the decision regions
     
     
>>> plot_decision_regions(X_test_pca, y_test, classifier=lr)
>>> plt.xlabel('PC1')
>>> plt.ylabel('PC2')
>>> plt.legend(loc = 'lower left')
>>> plt.tight_layout()
>>> plt.show()


    If we are interested in the explained variance ratios of the different principal componenets, 
    we can simply initialize the PCA class with the n_compoennets parameter
    set to None, so all principal components are kept and the explained variance ratio can
    then be accessed via the explained_variance_ratio_ attribute:
    
    >>> pca = PCA(n_components = None)
    >>> X_train_pca = pca.fit_transform(X_train_std)
    >>> pca.explained_variance_ratio_
    
    We set n_components = None when we initialized the PCA class so that it will return all principal
    components in a sorted order, instead of performing a dimensionality reduction.
'''