# DecisionTree-Implementation

Skeleton for a course activity: implement a decision tree from scratch with a
scikit-learn like API (`fit`, `predict`, `eval`).

- `decision_tree/base.py`: `BaseDecisionTree` with hyperparameters
  (`max_depth`, `min_samples_split`, `criterion`) and abstract hooks.
- `decision_tree/classifier.py`, `regressor.py`: subclasses to complete.
- `decision_tree/node.py`: tree node.

Search for `TODO` / `NotImplementedError` to find what to implement.
Run `pytest` to check the skeleton; extend the tests as you implement.
