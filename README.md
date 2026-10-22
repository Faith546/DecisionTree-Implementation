# DecisionTree-Implementation

Skeleton for a course activity: implement a decision tree from scratch with a
scikit-learn like API (`fit`, `predict`, `eval`).

- `decision_tree/base.py`: `BaseDecisionTree` with hyperparameters
  (`max_depth`, `min_samples_split`, `criterion`) and abstract hooks.
- `decision_tree/classifier.py`, `regressor.py`: subclasses to complete.
- `decision_tree/node.py`: tree node.

Search for `TODO` / `NotImplementedError` to find what to implement.
Run `pytest` to check the skeleton; extend the tests as you implement.

## Suggested Implementation Roadmap

Build the tree incrementally; finish and test each phase before moving on.

- [ ] **Phase 1: Minimal classification tree**
  - [ ] Implement one impurity criterion (e.g. Gini).
  - [ ] Implement a basic split search (try every candidate split, keep the best).
  - [ ] Grow the tree recursively, stop on pure nodes, and store the majority class in leaves.
  - [ ] Implement `predict` by traversing the tree.
- [ ] **Phase 2: Numeric and categorical features**
  - [ ] Numeric: binary splits on a threshold (`x <= t`), using midpoints between sorted unique values.
  - [ ] Categorical: one-vs-rest splits (`x == c`) or subset/grouped splits.
  - [ ] Detect feature types (or let the user declare them) and route each to the right split search.
  - [ ] Decide how to handle categories unseen during `fit` at inference time (e.g. send to the majority child or return the node's prediction).
- [ ] **Phase 3: Second classification criterion**
  - [ ] Support both Gini and Entropy via the `criterion` hyperparameter.
  - [ ] Validate the criterion name and compare the resulting trees.
- [ ] **Phase 4: Hyperparameters**
  - [ ] `max_depth`, `min_samples_split`, `min_samples_leaf`.
  - [ ] `max_features` (random feature subset per split) and, optionally, `min_impurity_decrease`.
  - [ ] Validate parameter values and test edge cases (e.g. `max_depth=0`).
- [ ] **Phase 5: Validation, evaluation and tests**
  - [ ] Implement `eval` with metrics such as accuracy.
  - [ ] Add a train/validation split or cross-validation helper.
  - [ ] Write unit tests: impurity values, split search, numeric/categorical data, unseen categories, hyperparameter limits.
- [ ] **Phase 6: Regression tree**
  - [ ] Use variance / MSE reduction as the split criterion (optionally MAE).
  - [ ] Store the mean (or median) of the targets in leaves.
  - [ ] Reuse the numeric and categorical split logic; evaluate with MSE / R².
- [ ] **Phase 7: Optional advanced enhancements**
  - [ ] Pruning (pre- or post-pruning, e.g. cost-complexity).
  - [ ] Feature importance.
  - [ ] Missing-value strategy (e.g. imputation or sending to the larger child).
  - [ ] Serialization (save/load the tree, e.g. to JSON).
