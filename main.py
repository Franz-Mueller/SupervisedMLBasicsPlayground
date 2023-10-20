import tkinter as tk
from sklearn.datasets import load_iris
from estimator import *
from loaddataset import *


# create tkinter Window
root = tk.Tk()
root.config(width=800, height=450)
root.title("Supervised Learning Playground")

# Load Datasets upfront
iris_dataset_scikitlearn = load_iris()
bostonhousing_dataset_custom = LoadDataset("bostonhousing.csv")

# buttons
button_1 = tk.Button(
    root,
    text="Nearest Neighbor Classification",
    command=lambda: AlgorithmsWithIrisDataset.NearestNeighborClassification(
        iris_dataset_scikitlearn
    ),
).grid(row=1, column=1)
button_2 = tk.Button(
    root,
    text="Decision Tree Classifier without regulations",
    command=lambda: AlgorithmsWithIrisDataset.DecisionTreeClassifierNonRegulations(
        iris_dataset_scikitlearn
    ),
).grid(row=3, column=1)
button_3 = tk.Button(
    root,
    text="Decision Tree Classifier with regulations",
    command=lambda: AlgorithmsWithIrisDataset.DecisionTreeClassifierWithRegulations(
        iris_dataset_scikitlearn
    ),
).grid(row=5, column=1)
button_4 = tk.Button(
    root,
    text="Logistic Regression",
    command=lambda: AlgorithmsWithIrisDataset.LogisticRegression(
        iris_dataset_scikitlearn
    ),
).grid(row=7, column=1)
button_5 = tk.Button(
    root,
    text="Support Vector Machine (Linear)",
    command=lambda: AlgorithmsWithIrisDataset.SupportVectorMachineLinear(
        iris_dataset_scikitlearn
    ),
).grid(row=9, column=1)
button_6 = tk.Button(
    root,
    text="Support Vector Machine (NotLinear)",
    command=lambda: AlgorithmsWithIrisDataset.SupportVectorMachineNonLinear(
        iris_dataset_scikitlearn
    ),
).grid(row=11, column=1)
button_7 = tk.Button(
    root,
    text="Ordinary Least Square",
    command=lambda: AlgorithmsWithIrisDataset.OrdinaryLeastSquare(
        iris_dataset_scikitlearn
    ),
).grid(row=13, column=1)
button_8 = tk.Button(
    root,
    text="Stochastic Gradient Descent Regressor",
    command=lambda: AlgorithmsWithIrisDataset.StochasticGradientDescentRegressor(
        iris_dataset_scikitlearn
    ),
).grid(row=15, column=1)
button_9 = tk.Button(
    root,
    text="Ordinary Least Square (Boston Housing)",
    command=lambda: AlgorithmsWithCustomDataset.OrdinaryLeastSquare(
        bostonhousing_dataset_custom
    ),
).grid(row=17, column=1)
button_10 = tk.Button(
    root,
    text="Stochastic Gradient Descent Regressor (Boston Housing)",
    command=lambda: AlgorithmsWithCustomDataset.StochasticGradientDescentRegressor(
        bostonhousing_dataset_custom
    ),
).grid(row=19, column=1)
button_11 = tk.Button(
    root,
    text="Random Forest Classifier Non Regulations",
    command=lambda: AlgorithmsWithIrisDataset.RandomForestClassifierNonRegulations(
        iris_dataset_scikitlearn
    ),
).grid(row=21, column=1)
button_10 = tk.Button(
    root,
    text="Boosted Decision Tree Using Ada",
    command=lambda: AlgorithmsWithIrisDataset.BoostedDecisionTreeUsingAda(
        iris_dataset_scikitlearn
    ),
).grid(row=32, column=1)
# run tkinter Window
root.mainloop()
