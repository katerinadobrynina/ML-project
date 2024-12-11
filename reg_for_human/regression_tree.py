import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def _mse(self, y):
        return np.mean((y-np.mean(y)) ** 2)
    def _best_split(self, X ,y):
        n_samples, n_features = X.shape
        if n_samples < 2 * self.min_samples_leaf:
            return None, None, None

        best_score = np.inf
        best_feature = None
        best_threshold = None
        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:, feature]):
                left_idx = np.where(X[:, feature] <= threshold)[0]
                right_idx = np.where(X[:, feature] > threshold)[0]
                if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
                    continue
