"""Classification tree."""
import numpy as np

from .base import BaseDecisionTree
from .node import Node


class DecisionTreeClassifier(BaseDecisionTree):
    """Tree variant for classification.

    Parameters:
        max_depth (int or None): See ``BaseDecisionTree``.
        min_samples_split (int): See ``BaseDecisionTree``.
        criterion (str): Supported values: 'gini', 'entropy'.

    Each hook below documents what it receives and must return; see
    ``BaseDecisionTree`` for the full contract.
    """

    _supported_criteria = ("gini", "entropy")

    def __init__(self, max_depth=None, min_samples_split=2, criterion="gini"):
        super().__init__(max_depth, min_samples_split, criterion)

    def _build_tree(self, X, y, depth):
        """Recursively grow the tree.

        Args:
            X (ndarray, shape (n_samples, n_features)): Features at this node.
            y (ndarray, shape (n_samples,)): Targets at this node.
            depth (int): Current depth (root is 0).

        Returns:
            Node: Leaf (when a stopping rule triggers) or internal node with
            two subtrees built from the best split.
        """
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return Node(value=self._leaf_value(y))

        split = self._best_split(X, y)
        if split is None:
            return Node(value=self._leaf_value(y))

        feature_index, threshold = split
        left = X[:, feature_index] <= threshold
        return Node(
            feature_index=feature_index,
            threshold=threshold,
            left=self._build_tree(X[left], y[left], depth + 1),
            right=self._build_tree(X[~left], y[~left], depth + 1),
        )

    def _impurity(self, y):
        """Impurity of ``y`` per ``self.criterion``: Gini impurity 1 - sum(p_k^2) or entropy -sum(p_k * log2(p_k)), where p_k is the fraction of samples of class k.

        Args:
            y (ndarray, shape (n_samples,)): Targets of a node.

        Returns:
            float: Impurity (0 if pure).
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()

        if self.criterion == "gini":
            return 1.0 - np.sum(probabilities ** 2)
        if self.criterion == "entropy":
            return -np.sum(probabilities * np.log2(probabilities))
        raise ValueError(f"Invalid criterion for classification: {self.criterion}")

    def _best_split(self, X, y):
        """Search all features/thresholds for the largest impurity decrease.

        Args:
            X (ndarray, shape (n_samples, n_features)): Features at the node.
            y (ndarray, shape (n_samples,)): Targets at the node.

        Returns:
            tuple (int, float) or None: ``(feature_index, threshold)``, or
            None if no split reduces impurity.
        """
        parent_impurity = self._impurity(y)
        best_gain = 0.0
        best_split = None
        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            values = np.unique(X[:, feature_index])
            for lower, upper in zip(values[:-1], values[1:]):
                threshold = lower / 2.0 + upper / 2.0
                left = X[:, feature_index] <= threshold
                right = ~left
                n_left = np.sum(left)
                n_right = np.sum(right)
                if n_left == 0 or n_right == 0:
                    continue

                weighted_impurity = (
                    n_left * self._impurity(y[left])
                    + n_right * self._impurity(y[right])
                ) / n_samples
                gain = parent_impurity - weighted_impurity
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_index, threshold)

        return best_split

    def _leaf_value(self, y):
        """Return the majority (most frequent) class of y.

        Args:
            y (ndarray, shape (n_samples,)): Targets that reached the leaf.

        Returns:
            The prediction for the leaf.
        """
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]

    def _score(self, y_true, y_pred):
        """Metric for ``eval``: accuracy = fraction of correct predictions.

        Args:
            y_true (ndarray, shape (n_samples,)): True targets.
            y_pred (ndarray, shape (n_samples,)): Predicted targets.

        Returns:
            float: Score (higher is better).
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(np.mean(y_true == y_pred))
