import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_blobs, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

# config
step_size = 0.2
color = plt.get_cmap('RdBu')
clf_name = "RBF SVM"
clf = SVC(gamma=2, C=1, probability=True)
clf2 = KNeighborsClassifier(5)
clf3 = MLPClassifier(hidden_layer_sizes=(100,50))
esb_name = "Bagging"
esb = BaggingClassifier(base_estimator=clf, random_state=0)
#esb = RandomForestClassifier(n_estimators=20, random_state=0)
#esb = AdaBoostClassifier(n_estimators=50)
#esb = GradientBoostingClassifier(n_estimators=100)
#esb = VotingClassifier(estimators=[('SVM', clf), ('KNN',clf2), ('MLP', clf3)], voting='soft', weights=[2,1,2])

# data Set Generation
data_sets = [make_classification(n_features=2, n_classes=3, n_informative=2, n_redundant=0, random_state=0,
                                 n_clusters_per_class=1),
             make_moons(noise=0.2, random_state=0),
             make_circles(noise=0.2, factor=0.5, random_state=0),
             make_blobs(n_features=2, centers=3, cluster_std=1.0, random_state=0)]

# plot
figure = plt.figure(figsize=(8, 8))

plt_idx = 0

# about each data set...
for ds_idx, ds in enumerate(data_sets):
    X, y = ds

    # separate data(train/test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    # first subplot : input data - training set
    plt_idx += 1
    ax = plt.subplot(len(data_sets), 4, plt_idx)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=color, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    if ds_idx == 0:
        ax.set_title("Input data (training)")

    # second subplot : input data - testing set
    plt_idx += 1
    ax = plt.subplot(len(data_sets), 4, plt_idx)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=color, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    if ds_idx == 0:
        ax.set_title("Input data (testing)")

    # training classifier
    clf.fit(X_train, y_train)

    # prediction with trained classifier
    score = clf.score(X_test, y_test)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # third subplot : classifier model result with training data set
    plt_idx += 1
    ax = plt.subplot(len(data_sets), 4, plt_idx)
    ax.contourf(xx, yy, Z, cmap=color, alpha=0.4)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=color, edgecolors='k')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=color, edgecolors='k', alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    if ds_idx == 0:
        ax.set_title(clf_name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')

    # training ensemble classifier
    esb.fit(X_train, y_train)

    # prediction with trained ensemble classifier
    score = esb.score(X_test, y_test)
    Z = esb.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # fourth subplot : ensemble classifier model result with test data set
    plt_idx += 1
    ax = plt.subplot(len(data_sets), 4, plt_idx)
    ax.contourf(xx, yy, Z, cmap=color, alpha=0.4)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=color, edgecolors='k')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=color, edgecolors='k', alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    if ds_idx == 0:
        ax.set_title(esb_name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')

plt.tight_layout()
plt.show()
