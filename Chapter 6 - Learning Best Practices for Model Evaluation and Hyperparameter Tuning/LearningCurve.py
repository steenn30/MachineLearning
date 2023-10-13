import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


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