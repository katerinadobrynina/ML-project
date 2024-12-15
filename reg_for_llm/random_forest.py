import numpy as np
import pandas as pd

class RegressionTree:
    """
    A single regression tree built from scratch for numeric prediction.

    This tree splits the data recursively to minimize mean squared error (MSE).
    """

    def __init__ (self,
                  max_depth: int = None,
                  min_samples_split: int = 2,
                  min_samples_leaf: int = 1,
                  max_features: int = None):
        """
        Parameters
        ----------
        max_depth : int, optional
            Maximum depth of the tree. If None, the tree grows until all leaves are pure or
            min_samples_split is reached.
        min_samples_split : int, optional
            Minimum number of samples required to split an internal node.
        min_samples_leaf : int, optional
            Minimum number of samples required to be at a leaf node.
        max_features : int, optional
            Number of features to consider when looking for the best split.
            If None, all features are used.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root = None  # Will hold the tree structure after fitting

    class Node:
        """
        Represents a node in the regression tree.
        """

        def __init__ (self, feature_idx=None, threshold=None, left=None, right=None, value=None):
            """
            If 'feature_idx' is not None, this node splits on a feature. Otherwise, it's a leaf node.
            """
            self.feature_idx = feature_idx
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit (self, X: np.ndarray, y: np.ndarray):
        """
        Build the regression tree by recursively splitting the training data.
        """
        # If max_features is not set, default to using all features
        if self.max_features is None or self.max_features > X.shape[1]:
            self.max_features = X.shape[1]

        # Build the tree recursively
        self.root = self._build_tree (X, y, current_depth=0)

    def predict (self, X: np.ndarray) -> np.ndarray:
        """
        Predict values for each example in X.
        """
        predictions = [self._predict_sample (sample, self.root) for sample in X]
        return np.array (predictions)

    def _build_tree (self, X: np.ndarray, y: np.ndarray, current_depth: int):
        """
        Recursively build the regression tree.
        """
        num_samples, num_features = X.shape
        # If stopping conditions are met, return a leaf node
        if (num_samples < self.min_samples_split or
                (self.max_depth is not None and current_depth >= self.max_depth) or
                np.unique (y).size == 1):
            leaf_value = np.mean (y)
            return self.Node (value=leaf_value)

        # Randomly select the features to consider at this node
        feature_indices = np.random.choice (num_features, self.max_features, replace=False)

        best_feature, best_threshold, best_mse = None, None, float ('inf')
        best_left_idx, best_right_idx = None, None

        # Compute the current node's MSE to compare improvement
        current_mse = self._variance (y)  # overall variance is used as baseline

        for feature_idx in feature_indices:
            thresholds = np.unique (X[:, feature_idx])

            for threshold in thresholds:
                left_idx = X[:, feature_idx] <= threshold
                right_idx = X[:, feature_idx] > threshold

                if np.sum (left_idx) < self.min_samples_leaf or np.sum (right_idx) < self.min_samples_leaf:
                    continue

                left_mse = self._variance (y[left_idx]) * len (y[left_idx])
                right_mse = self._variance (y[right_idx]) * len (y[right_idx])

                total_mse = (left_mse + right_mse) / num_samples

                # We want the split that yields the lowest MSE
                if total_mse < best_mse:
                    best_mse = total_mse
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_left_idx = left_idx
                    best_right_idx = right_idx

        # If no split improves, return a leaf
        if best_feature is None:
            leaf_value = np.mean (y)
            return self.Node (value=leaf_value)

        # Recursively build left and right subtrees
        left_child = self._build_tree (X[best_left_idx, :], y[best_left_idx], current_depth + 1)
        right_child = self._build_tree (X[best_right_idx, :], y[best_right_idx], current_depth + 1)

        # Create an internal node
        return self.Node (feature_idx=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def _predict_sample (self, sample: np.ndarray, node: Node) -> float:
        """
        Traverse the tree to predict the value for a single sample.
        """
        # Leaf node
        if node.value is not None:
            return node.value

        # Internal node
        if sample[node.feature_idx] <= node.threshold:
            return self._predict_sample (sample, node.left)
        else:
            return self._predict_sample (sample, node.right)

    @staticmethod
    def _variance (y: np.ndarray) -> float:
        """
        Returns the variance of y.
        """
        return np.var (y) if len (y) > 0 else 0.0


class RandomForestRegressor:
    """
    A random forest regressor built from scratch using multiple RegressionTree estimators.

    The final prediction is the average of predictions from all the individual trees.
    """

    def __init__ (self,
                  n_trees: int = 10,
                  max_depth: int = None,
                  min_samples_split: int = 2,
                  min_samples_leaf: int = 1,
                  max_features: int = None,
                  bootstrap: bool = True,
                  random_state: int = None):
        """
        Parameters
        ----------
        n_trees : int
            Number of trees in the forest.
        max_depth : int, optional
            Maximum depth of each tree.
        min_samples_split : int
            Minimum samples to split an internal node.
        min_samples_leaf : int
            Minimum samples allowed in a leaf node.
        max_features : int, optional
            Number of features to consider at each split.
            If None, all features are considered.
        bootstrap : bool
            Whether to use bootstrap samples (sampling with replacement).
        random_state : int, optional
            Seed for reproducibility.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        # For reproducibility
        if random_state is not None:
            np.random.seed (random_state)

        self.trees = []  # will hold individual RegressionTree objects

    def fit (self, X, y):
        """
        Train the random forest on the given data and labels.
        """
        self.trees = []  # reset if re-fitting
        n_samples, _ = X.shape

        for _ in range (self.n_trees):
            if self.bootstrap:
                # Generate bootstrap sample
                indices = np.random.choice (n_samples, size=n_samples, replace=True)
                X_sample = X[indices]
                y_sample = y[indices]
            else:
                # Use the entire dataset without sampling
                X_sample = X
                y_sample = y

            tree = RegressionTree (
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features
            )
            tree.fit (X_sample, y_sample)
            self.trees.append (tree)

    def predict (self, X) -> np.ndarray:
        """
        Predict values for each example in X by averaging predictions from all trees.
        """
        # Collect predictions from each tree
        all_preds = np.array ([tree.predict (X) for tree in self.trees])
        # Average predictions
        return np.mean (all_preds, axis=0)


# -------------------
# Demonstration usage:
# -------------------
if __name__ == "__main__":
    # Example usage with dummy data
    np.random.seed (42)
    X = np.random.rand (100, 5)  # 100 samples, 5 features
    y = X[:, 0] * 5 + np.random.randn (100) * 0.5  # Some synthetic target

    # Create a Random Forest Regressor
    rf = RandomForestRegressor (
        n_trees=5,  # number of trees
        max_depth=5,  # max depth of each tree
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=3,  # random subset of 3 features at each split
        bootstrap=True,
        random_state=123
    )

    # Fit the model
    rf.fit (X, y)

    # Make predictions
    preds = rf.predict (X)

    # Evaluate performance (example: mean squared error)
    mse = np.mean ((preds - y) ** 2)
    print (f"Training MSE: {mse:.4f}")

    # You can also do cross-validation or train-test splits for a more robust evaluation.
