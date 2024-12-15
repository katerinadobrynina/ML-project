import numpy as np
from decision_tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd

class RandomForestRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, num_trees = 5, max_depth = 10, min_samples_split = 2, min_samples_leaf = 1, num_features = None):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.num_features = num_features
        self.min_samples_leaf = min_samples_leaf
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = np.array(X)
        if isinstance(y, pd.Series):
            y = y.values
        else:
            y = np.array(y)
        for i in range(self.num_trees):
            tree = DecisionTreeRegressor(max_depth = self.max_depth, min_samples_split = self.min_samples_split,
                                         min_samples_leaf = self.min_samples_leaf, num_features = self.num_features)
            X_sample, y_sample = self.bootstrap(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
    @staticmethod
    def bootstrap(X, y):
        num_sampls = X.shape[0]
        idxs = np.random.choice(num_sampls, num_sampls, replace = True)
        return X[idxs], y[idxs]


    def predict(self, X):
        if isinstance (X, pd.DataFrame):
            X = X.values
        else:
            X = np.array (X)
        predictions = np.array([tree.predict (X) for tree in self.trees])
        tree_preds = np.mean(predictions, axis=0)
        return tree_preds