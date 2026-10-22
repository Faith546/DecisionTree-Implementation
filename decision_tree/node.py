"""Tree node structure."""
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Node:
    """A node of the tree. Students may extend or replace this."""

    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    value: Any = None  # prediction stored in leaves

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None
