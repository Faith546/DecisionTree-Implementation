"""Base class for decision trees with a scikit-learn like API."""
from abc import ABC, abstractmethod


class BaseDecisionTree(ABC):
    """Skeleton shared by every tree variant (classification, regression, ...).

    Hyperparameters are set in ``__init__``; ``fit``, ``predict`` and ``eval``
    follow the scikit-learn conventions. Subclasses implement the abstract
    hooks; the TODOs are left to the students.
    """

    def __init__(self, max_depth=None, min_samples_split=2, criterion=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None

    def get_params(self):
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "criterion": self.criterion,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key not in self.get_params():
                raise ValueError(f"Invalid parameter: {key}")
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        """Build the tree from training data. Returns ``self``."""
        # TODO: validate hyperparameters and inputs, then build the tree
        #       recursively with ``_build_tree`` and store it in ``self.root``.
        raise NotImplementedError("TODO: implement fit")

    def predict(self, X):
        """Predict a target for each row of ``X``."""
        # TODO: traverse the tree for every sample (see ``_predict_one``).
        raise NotImplementedError("TODO: implement predict")

    def eval(self, X, y):
        """Score the model on ``(X, y)`` (accuracy, R^2, ...)."""
        # TODO: call ``predict`` and compute the metric via ``_score``.
        raise NotImplementedError("TODO: implement eval")

    @abstractmethod
    def _build_tree(self, X, y, depth):
        """Recursively grow the tree and return a ``Node``."""

    @abstractmethod
    def _impurity(self, y):
        """Impurity of a set of targets according to ``self.criterion``."""

    @abstractmethod
    def _best_split(self, X, y):
        """Return the best (feature_index, threshold) or None."""

    @abstractmethod
    def _leaf_value(self, y):
        """Prediction stored in a leaf (majority class, mean, ...)."""

    @abstractmethod
    def _score(self, y_true, y_pred):
        """Metric used by ``eval``."""
