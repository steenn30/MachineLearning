from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt

'''
About
        The validation curve function uses straified k-fold cross-validaiton by default to estimate
        the performance of the calssifier
        
        Inside the validation_curve function, we specified the parameter that we wanted to evaluate
            - In this case, it is C, the inverse regularaization of the LogisticRegression classifer,
              which we wrote as 'logisticregression__C' to access the LogisticRegression object inside 
              the scikit-learn pipeline for a specified value range that we set via the param_range
              parameter.
        
        We plot the average training and cross-validation accuracies and the corresponding standard deviations
        
        We can see that the model slightly underfits the data when we increase the regularizaion strength of 
        regularization, so the model tends to slightly overfit the data
        
        In this case, the sweet spot appears to be between 0.01 and 0.1 of the c value
'''
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(
                                estimator=pipe_lr,
                                X=X_train,
                                y=y_train,
                                param_name='logisticregression__C',
                                param_range=param_range,
                                cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis = 1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis = 1)

plt.plot(param_range, train_mean,
         color='blue', marker='o',
         markersize=5, label='Training accuracy')

plt.fill_between(paramrange, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')

plt.plot(param_range, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validation accuracy')


plt.fill_between(param_range,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpah=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8,1.0])
plt.show()