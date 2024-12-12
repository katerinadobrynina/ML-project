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


    def build_tree(self, X, y, depth = 0):
        num_samples, num_features = X.shape
        if(depth >= self.max_depth or len(np.unique(y)) == 1 or num_samples < self.min_samples_split):
            leaf_val = self.most_comm_lab(y)
            return Node(var_red=leaf_val)
        feat_ids = np.random.choice(num_features, self.num_features, replace = False)
        best_feat, best_threshold = self.best_split(X,y, feat_ids)

        best_gain = 0.0
        best_criteria = None
        best_sets = None

        feature_indices = list(range(num_features))


    def best_split(self, X, y, feat_ids):
        best_gain = -1
        slit_idx, split_thes = None, None
        for feat_id, in feat_ids:
            X_col = X[:, feat_id]
            thresholds = np.unique(X_col)

            for thr in thresholds:
                gain = self.inf_gain(y,X_col,thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_id
                    split_thes = thr
        return split_idx, split_thes


    def inf_gain(self,y, X_col, threshold) -> np.float64:
        parent_entropy = self.entropy(y)
        ledf_idx, right_idx = self.split(X_col, threshold)
        if len(ledf_idx) == 0 or len(right_idx) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(ledf_idx), len(right_idx)
        e_l, e_r = self.entropy(y[ledf_idx]), self.entropy(y[right_idx])
        return parent_entropy - (n_l/n)*e_l - (n_r/n)*e_r


    def split(self, X_col, threshold) -> tuple:
        left_idx = np.argwhere(X_col <= threshold).flatten()
        right_idx = np.argwhere(X_col > threshold).flatten()
        return left_idx, right_idx


    def entropy(self, y) -> np.float64:
        return -np.sum([p * np.log2(p) for p in np.bincount(y)/len(y) if p > 0])


    def most_comm_lab(self, y):
        counter = Counter(y)
        return  counter.most_common(1)[0][0]