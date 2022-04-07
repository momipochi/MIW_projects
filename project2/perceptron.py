from mimetypes import init
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from plotka import plot_decision_regions
from reglog import LogisticRegressionGD

class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


class Classifier:
    def __init__(self, ppn1: Perceptron, ppn2: Perceptron) -> None:
        self.ppn1 = ppn1
        self.ppn2 = ppn2
    
    def predict(self, x):
        return np.dot(
            self.ppn1.predict(x) == 1, 0, np.where(self.ppn2.predict(x) == 1, 2, 1)
        )
    

def main():

    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y
    )


    X_train_01_subset = X_train.copy()
    y_train_01_subset = y_train.copy()
    y_train_03_subset = y_train.copy()
    # X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]

    y_train_01_subset[(y_train == 1) | (y_train == 2)] = -1
    y_train_01_subset[(y_train_01_subset == 0)] = 1

    y_train_03_subset[(y_train == 1) | (y_train == 0)] = -1
    y_train_03_subset[(y_train_03_subset == 2)] = 1
    # w perceptronie wyjście jest albo 1 albo -1
    # y_train_01_subset[(y_train_01_subset == 0)] = -1
    # y_train_01_subset[(y_train_01_subset == 1)] = 0
    # y_train_01_subset[(y_train_01_subset == 2)] = 1

    ppn = Perceptron(eta=0.1, n_iter=50)
    ppn.fit(X_train_01_subset, y_train_01_subset)
    ppn2 = Perceptron(eta=0.1, n_iter=50)
    ppn2.fit(X_train_01_subset, y_train_03_subset)
    
    ppnclassifier = Classifier(ppn,ppn2)

    # plot_decision_regions(X=X_train_01_subset, y=y_train_01_subset, classifier=ppn)
    # plot_decision_regions(X=X_train_01_subset, y=y_train_03_subset, classifier=ppn2)
    # plot_decision_regions(X=X_train_01_subset, y=y_train_01_subset, classifier=ppnclassifier)
    # plot_decision_regions(X=X_train_01_subset, y=y_train_03_subset, classifier=ppnclassifier)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()
