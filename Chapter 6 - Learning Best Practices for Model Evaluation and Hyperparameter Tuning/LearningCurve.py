import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

'''
About
    We passed max_iter=10000 as an additional arguement when isntantiating
    the LogisticRegression object (which uses 1,000 iterations as a default) 
    to avoid convergence issues for smaller dataset sizes or extreme regularization 
    parameter values
    
    With the "train_sizes" parameter in the learning_curve function, we can control
    the absolute or relative number of training examples that are used to generate the 
    learning curves
        - We set train_sizes = np.linspace(0.1, 1.0, 10) to use 10 evenly spaced, relative
        intervals for the training dataset sizes
        
        - By default the learning_curve function uses stratified k-fold cross-validation 
          to calculate the cross_calidation accuracy of a classifer
              - We set k=10 via the cv parameter for 10-fold stratified cross-validation
              
        - Then, we calculated the average accuracies from the retunred cross-validated
          training and test scores for the different sizes of the training dataset
              - plotted with Matplotlib's plot function
        - Furthermore, we added the standard deviate of the average accuracy to the plot
          using the fill_between function to indicate the variance of the estimate
    
    - When the gap between validation and training accuracy widens, this is an indicator
      of an increasing degree of overfitting
    

'''

pipe_lr = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty='12',
                                           random_state=1,
                                           solver='lbfgs',
                                           max_iter=10000))

train_sizes, train_scores, test_scores =\
                    learning_curve(estimator=pipe_lr,
                                   X=X_train,
                                   y=y_train,
                                   train_sizes=np.linspace(
                                                0.1, 1.0, 10),
                                   cv=10,
                                   n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis =1)

pit.plot(train_sizes, train_mean,
        color='blue', marker='o',
        markersize=5, label='Training accuracy')

plt.fill_between(train_sizes,
                 treain_mean + train_std,
                 train_mean - train_std,
                 alpha = 0.15, color='blue')


plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validation accuracy')


plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha = 0.15, color='green')



plt.grid()
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.03])
plt.show()