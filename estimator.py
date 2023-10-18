from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored
import sklearn.linear_model as lm
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class AlgorithmsWithIrisDataset:
    def RunEstimator(dataset, estimator):
        # load data and targets
        X = dataset.data
        y = dataset.target
        # determine testpartition and trainingpartition
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
        # fites estimator
        estimator.fit(X_train, y_train)
        # prints results
        print(colored("prediction for 6.3, 2.7, 5.5, 1.5 (which should be 2)", "blue"))
        print(colored(estimator.predict([[6.3, 2.7, 5.5, 1.5]]), "green"))
        print(colored("score for Trainingdata:", "blue"))
        print(colored(estimator.score(X_train, y_train), "green"))
        print(colored("score for Testdata:", "blue"))
        print(colored(estimator.score(X_test, y_test), "green"))

    def NearestNeighborClassification(dataset):
        print(
            "_________________________________________________________________________"
        )
        print("Nearest Neighbor Classification")
        AlgorithmsWithIrisDataset.RunEstimator(
            dataset, neighbors.KNeighborsClassifier()
        )

    def DecisionTreeClassifierNonRegulations(dataset):
        print(
            "_________________________________________________________________________"
        )
        print("Deciscion Tree Classifier without Regulations")
        AlgorithmsWithIrisDataset.RunEstimator(dataset, DecisionTreeClassifier())

    def DecisionTreeClassifierWithRegulations(dataset):
        print(
            "_________________________________________________________________________"
        )
        print("Deciscion Tree Classifier with Regulations")
        AlgorithmsWithIrisDataset.RunEstimator(
            dataset,
            DecisionTreeClassifier(
                max_depth=10, min_samples_leaf=10, min_samples_split=2, max_leaf_nodes=8
            ),
        )

    def LogisticRegression(dataset):
        print(
            "_________________________________________________________________________"
        )
        print("logistic Regression")
        AlgorithmsWithIrisDataset.RunEstimator(dataset, lm.LogisticRegression())

    def SupportVectorMachineLinear(dataset):
        print(
            "_________________________________________________________________________"
        )
        print("Support Vector Machine (Linear)")
        AlgorithmsWithIrisDataset.RunEstimator(
            dataset, SVC(kernel="linear", C=1, gamma=1)
        )

    def SupportVectorMachineNonLinear(dataset):
        print(
            "_________________________________________________________________________"
        )
        print("Support Vector Machine (Not Linear)")
        AlgorithmsWithIrisDataset.RunEstimator(dataset, SVC(kernel="rbf", C=1, gamma=1))

    def OrdinaryLeastSquare(dataset):
        print(
            "_________________________________________________________________________"
        )
        print("Ordinary Least Square")
        AlgorithmsWithIrisDataset.RunEstimator(dataset, lm.LinearRegression())

    def StochasticGradientDescentRegressor(dataset):
        print(
            "_________________________________________________________________________"
        )
        print("Stochastic Gradient Descent Regressor")
        AlgorithmsWithIrisDataset.RunEstimator(dataset, lm.SGDRegressor())


class AlgorithmsWithCustomDataset:
    def RunEstimator(dataset, estimator):
        # load data and targets
        X = dataset.data[:, 5:6]
        y = dataset.target
        # determine testpartition and trainingpartition
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
        # fites estimator
        estimator.fit(X_train, y_train)
        # plots results
        y_pred = estimator.predict(X)
        jointplot = sns.jointplot(x=X[:, 0], y=y, kind="scatter")
        sns.regplot(
            x=X[:, 0], y=y_pred, ax=jointplot.ax_joint, scatter=False, color="r"
        )
        jointplot.ax_joint.set_xlabel("rooms per dwelling")
        jointplot.ax_joint.set_ylabel("median value of owner occupied homes (x1000)")
        plt.show()
        # print results to console
        print(colored("score for Trainingdata:", "blue"))
        print(colored(estimator.score(X_train, y_train), "green"))
        print(colored("score for Testdata:", "blue"))
        print(colored(estimator.score(X_test, y_test), "green"))

    def OrdinaryLeastSquare(dataset):
        print(
            "_________________________________________________________________________"
        )
        print("Ordinary Least Square")
        AlgorithmsWithCustomDataset.RunEstimator(dataset, lm.LinearRegression())

    def StochasticGradientDescentRegressor(dataset):
        print(
            "_________________________________________________________________________"
        )
        print("Ordinary Least Square")
        AlgorithmsWithCustomDataset.RunEstimator(dataset, lm.SGDRegressor(tol=None))
