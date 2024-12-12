import numpy as np
from decision_tree import DecisionTreeRegressor

class RandomForestRegressor:
    def __init__(self, num_trees = 5, max_depth = 10, min_samples_split = 2, min_samples_leaf = 1, num_features = None):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.num_features = num_features
        self.min_samples_leaf = min_samples_leaf
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for i in range(self.num_trees):
            tree = DecisionTreeRegressor(max_depth = self.max_depth, min_samples_split = self.min_samples_split,
                                         min_samples_leaf = self.min_samples_leaf, num_features = self.num_features)
            X_sample, y_sample = self.bootstrap(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
    def bootstrap(self, X, y):
        num_sampls = X.shape[0]
        idxs = np.random.choice(num_sampls, num_sampls, replace = True)
        return X[idxs], y[idxs]


    def predict(self, X):
        predictions = np.array([tree.predict (X) for tree in self.trees])
        tree_preds = np.mean(predictions, axis=0)
        return tree_preds


from random import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def train():
    data = datasets.load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate custom RandomForest
    custom_clf = RandomForestRegressor(num_trees=10, max_depth=10)
    custom_clf.fit(X_train, y_train)
    custom_pred = custom_clf.predict(X_test)
    custom_mse_value = mse(y_test, custom_pred)
    print("Custom RandomForest MSE:", custom_mse_value)

    # Train and evaluate sklearn RandomForest
    sklearn_clf = SklearnRandomForestRegressor(n_estimators=10, max_depth=10, random_state=42)
    sklearn_clf.fit(X_train, y_train)
    sklearn_pred = sklearn_clf.predict(X_test)
    sklearn_mse_value = mean_squared_error(y_test, sklearn_pred)
    print("Sklearn RandomForest MSE:", sklearn_mse_value)

def main():
    train()

if __name__ == "__main__":
    main()