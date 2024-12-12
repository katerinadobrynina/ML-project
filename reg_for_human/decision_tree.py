import numpy as np
import pandas as pd
from collections import Counter


class Node:
    def __init__(self, feature_index = None, threshold = None, left = None, right = None, var_red = None, var_imp = None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red
        self.var_imp = var_imp



class DecisionTreeRegressor:
    def __init__(self, max_depth = 10, min_samples_split = 2, min_samples_leaf = 1, num_features = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.num_features = num_features
        self.root = None


    def fit(self, X, y):
        self.num_features = X.shape[1] if not self.num_features else min(X.shape[1], self.num_features)
        self.root = self.build_tree(X, y)


    def build_tree(self, X, y, depth = 0) -> Node:
        num_samples, num_features = X.shape
        if(depth >= self.max_depth or len(np.unique(y)) == 1 or num_samples < self.min_samples_split):
            leaf_val = np.mean(y)
            return Node(var_red=leaf_val)
        feat_ids = np.random.choice(num_features, self.num_features, replace = False)
        best_feat, best_threshold = self.best_split(X,y, feat_ids)
        if best_feat is None or best_threshold is None:
            leaf_val = np.mean(y)
            return Node(var_red=leaf_val)
        left_ids, right_ids = self.split(X[:, best_feat], best_threshold)
        left_subtree = self.build_tree(X[left_ids, :], y[left_ids], depth + 1)
        right_subtree = self.build_tree(X[right_ids, :], y[right_ids], depth + 1)
        return Node(best_feat, best_threshold, left_subtree, right_subtree)



    def best_split(self, X, y, feat_ids) -> tuple:
        best_gain = -1
        split_idx, split_thes = None, None
        for feat_id in feat_ids:
            X_col = X[:, feat_id]
            thresholds = np.unique(X_col)
            for thr in thresholds:
                left_ids, right_ids = self.split(X_col, thr)
                if len(left_ids) == 0 or len(right_ids) == 0:
                    continue
                gain = self.variance_reduction(y,y[left_ids],y[right_ids])
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_id
                    split_thes = thr
        return split_idx, split_thes

    def variance_reduction(self, y, left_y, right_y) -> float:
        parent_var = np.var(y)
        n = len (y)
        n_l = len(left_y)
        n_r = len(right_y)
        return parent_var - (n_l / n) * np.var(left_y) - (n_r / n) * np.var(right_y)


    def split(self, X_col, threshold) -> tuple:
        left_idx = np.argwhere(X_col <= threshold).flatten()
        right_idx = np.argwhere(X_col > threshold).flatten()
        return left_idx, right_idx

    def predict(self, X) -> np.array:
        return np.array([self.traverse_tree(x, self.root) for x in X])

    def traverse_tree(self, x, node) -> float:
        if node.var_red is not None:
            return node.var_red
        else:
            if x[node.feature_index] <= node.threshold:
                return self.traverse_tree(x, node.left)
            else:
                return self.traverse_tree(x, node.right)



"""
This is basic test section, which tests on the sklearn preprocessed dataset and can be removed later.
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from reg_for_human.decision_tree import DecisionTreeRegressor


def train():
    data = datasets.load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeRegressor()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = mse(y_test, pred)
    print (acc)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


def main():
    train()

if __name__ == "__main__":
    main()


