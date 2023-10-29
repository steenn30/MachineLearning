from sklearn.metrics import roc_curve, aoc
from scipy import interp
from test.ann_module2 import CV

'''
About
    We use StraifiedKFold class from scikit-learn and calculated the ROC performance of the LogisticRegression
    classifier in our pipe_lr pipeline using the roc_curve function from the sklearn.metrics module separately 
    for each iteration.
    
    Furthermore, we interpolate the average ROC curve from the three folds via the interp function that we imported
    from SciPy and calculated the area the area under the curve via the auc function.
    
    The resulting ROC curve indicates that there is a certain degree of variance between the different folds,
    and the average ROC AUC(0.76) falls between a perfect score(1.0) and random guessing (0.5)
    
    If we are just interested in the ROC AUC score, we could also directly import the roc_auc_score function
    from the sklearn.metrics submodule, which can be used similarlyt to the other scoring functions (for example,
    precision_score) that were introduced in the previous sections.
    
    Reporting the performance of a classifier as the ROC AUC can yield further insights into a classifier's performance 
    with respect to imbalanced samples. However, while the accuracy score can be interpreted as a single cut-off point
    on an ROC curve, A.P. Bradley showed that the ROC AUC and accuracy metrics mosly agree with each other:
        -     The use of the area under the ROC curve in the evaluation of machine learning algorithms, A. P. Bradley, 
                    Pattern Recognition, 30(7): 1145-1159, 1997.
        
'''

pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_compnents = 2),
                        LogisticRegression(penalty='12',
                                           random_state=1,
                                           solver='lbgfs',
                                           C=100.0))

X_train2 = X_train[:, [4, 14]]

cv = list(StratifiedKFold(n_splits =3,
                          random_state=1).split(X_train,
                                                y_train))

fig = plt.figure(figsize(7,5))

mean_tpr = 0.0
mean_fpr = np.linspace(0,1,100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(
        X_train2[train],
        y_train[train]).predict_proba(X_train2[test])
    fpr, tpr, thresholds = roc_curve(y_train[test],
                                     probas[:,1],
                                     pos_label=1)
    
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr,
             tpr,
             label='ROC fold %d (area-%0,2f)' 
             % (i+1, roc_auc))
    
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fprm, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label= 'Mean ROC (area=%0.2f)' % mean_auc, lw=2)
    plt.plot([0,0,1],
             [0,1,1],
             linestyle=':',
             color='black',
             label='Perfect performance')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc="lower right")
    plt.show()

    