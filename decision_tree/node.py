"""Tree node structure."""
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Node:
    """A node of the tree. Students may extend or replace this.

    Internal nodes store a split rule (``feature_index`` and ``threshold``)
    and two children; samples with ``X[:, feature_index] <= threshold`` go to
    ``left`` and the rest go to ``right``. Leaves store only ``value``.

    Attributes:
        feature_index: Column of ``X`` used for the split (None in leaves).
        threshold: Split value for that feature (None in leaves).
        left: Subtree for samples satisfying ``x <= threshold``.
        right: Subtree for samples satisfying ``x > threshold``.
        value: Prediction stored in a leaf (class label, mean, ...).
    """

    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    value: Any = None

    @property
    def is_leaf(self) -> bool:
        """Return True if the node has no children (it is a leaf)."""
        return self.left is None and self.right is None
