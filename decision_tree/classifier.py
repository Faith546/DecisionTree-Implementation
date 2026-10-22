"""Classification tree skeleton."""
from .base import BaseDecisionTree


class DecisionTreeClassifier(BaseDecisionTree):
    """Criteria to support: 'gini', 'entropy'."""

    def __init__(self, max_depth=None, min_samples_split=2, criterion="gini"):
        super().__init__(max_depth, min_samples_split, criterion)

    def _build_tree(self, X, y, depth):
        raise NotImplementedError("TODO")

    def _impurity(self, y):
        raise NotImplementedError("TODO: gini / entropy")

    def _best_split(self, X, y):
        raise NotImplementedError("TODO")

    def _leaf_value(self, y):
        raise NotImplementedError("TODO: majority class")

    def _score(self, y_true, y_pred):
        raise NotImplementedError("TODO: accuracy")
