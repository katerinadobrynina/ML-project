import cupy as cp
from joblib import Parallel, delayed

class DecisionTreeRegressor:
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, num_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.num_features = num_features
        self.root = None

    def fit(self, X, y):
        self.num_features = X.shape[1] if not self.num_features else min(X.shape[1], self.num_features)
        self.root = self.build_tree(X, y)

    def build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if depth >= self.max_depth or len(cp.unique(y)) == 1 or num_samples < self.min_samples_split:
            leaf_val = cp.mean(y)
            return Node(var_red=leaf_val)

        feat_ids = cp.random.choice(num_features, self.num_features, replace=False)
        best_feat, best_threshold = self.best_split(X, y, feat_ids)

        if best_feat is None or best_threshold is None:
            leaf_val = cp.mean(y)
            return Node(var_red=leaf_val)

        left_ids, right_ids = self.split(X[:, best_feat], best_threshold)
        left_subtree = self.build_tree(X[left_ids, :], y[left_ids], depth + 1)
        right_subtree = self.build_tree(X[right_ids, :], y[right_ids], depth + 1)
        return Node(best_feat, best_threshold, left_subtree, right_subtree)

    def best_split(self, X, y, feat_ids):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_id in feat_ids:
            X_col = X[:, feat_id]
            thresholds = cp.unique(X_col)

            for thresh in thresholds:
                left_ids, right_ids = self.split(X_col, thresh)
                if len(left_ids) == 0 or len(right_ids) == 0:
                    continue

                gain = self.variance_reduction(y, y[left_ids], y[right_ids])
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_id
                    split_thresh = thresh

        return split_idx, split_thresh

    @staticmethod
    def variance_reduction(y, left_y, right_y):
        parent_var = cp.var(y)
        n = len(y)
        n_l = len(left_y)
        n_r = len(right_y)
        return parent_var - (n_l / n) * cp.var(left_y) - (n_r / n) * cp.var(right_y)

    @staticmethod
    def split(X_col, threshold):
        left_idx = cp.argwhere(X_col <= threshold).flatten()
        right_idx = cp.argwhere(X_col > threshold).flatten()
        return left_idx, right_idx

    def predict(self, X):
        return cp.array([self.traverse_tree(x, self.root) for x in X])

    def traverse_tree(self, x, node):
        if node.var_red is not None:
            return node.var_red
        else:
            if x[node.feature_index] <= node.threshold:
                return self.traverse_tree(x, node.left)
            else:
                return self.traverse_tree(x, node.right)


class RandomForestRegressor:
    def __init__(self, num_trees=5, max_depth=10, min_samples_split=2, min_samples_leaf=1, num_features=None):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.num_features = num_features
        self.min_samples_leaf = min_samples_leaf
        self.trees = []

    def fit(self, X, y):
        X, y = cp.asarray(X), cp.asarray(y)
        self.trees = Parallel(n_jobs=-1)(
            delayed(self.build_tree)(X, y) for _ in range(self.num_trees)
        )

    def build_tree(self, X, y):
        tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            num_features=self.num_features
        )
        X_sample, y_sample = self.bootstrap(X, y)
        tree.fit(X_sample, y_sample)
        return tree

    @staticmethod
    def bootstrap(X, y):
        num_samples = X.shape[0]
        idxs = cp.random.choice(num_samples, num_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        X = cp.asarray(X)
        predictions = cp.array([tree.predict(X) for tree in self.trees])
        return cp.mean(predictions, axis=0)
