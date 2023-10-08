


from matplot importlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02 ):
    
    #setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    
    #plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 0].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arrange(x1_min, x1_max, resolution)
                           np.arrange(x2_min, x2_max, resolution))
    Z=classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    #plot examples by class
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1, 0]
                    y=X[y== c1, 1]),
                    alpha = 0.6,
                    color = cmap(idx),
                    edgecolor = 'black',
                    marker=markers[idx],
                    label = c1)
                    