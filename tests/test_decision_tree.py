import csv
import unittest
from pathlib import Path

import numpy as np

from decision_tree import BaseDecisionTree, DecisionTreeClassifier, DecisionTreeRegressor


class TestDecisionTrees(unittest.TestCase):
    def test_base_class_is_abstract(self):
        with self.assertRaises(TypeError):
            BaseDecisionTree()

    def test_parameters_and_validation(self):
        for cls in (DecisionTreeClassifier, DecisionTreeRegressor):
            with self.subTest(cls=cls):
                model = cls(max_depth=3, min_samples_split=4)
                self.assertEqual(model.get_params()['max_depth'], 3)
                self.assertIs(model.set_params(max_depth=2), model)
                with self.assertRaises(ValueError):
                    model.set_params(unknown=1)
                with self.assertRaises(ValueError):
                    cls(min_samples_split=1).fit([[0]], [0])
                with self.assertRaises(ValueError):
                    cls(criterion='invalid').fit([[0]], [0])
                with self.assertRaises(ValueError):
                    cls().fit([[0], [1]], [0])
                with self.assertRaises(ValueError):
                    cls().predict([[0]])

    def test_classifier_impurity_and_leaf(self):
        labels = np.array(['a', 'a', 'b', 'b'])
        self.assertAlmostEqual(DecisionTreeClassifier(criterion='gini')._impurity(labels), 0.5)
        self.assertAlmostEqual(DecisionTreeClassifier(criterion='entropy')._impurity(labels), 1.0)
        self.assertEqual(DecisionTreeClassifier()._leaf_value(['a', 'b', 'a']), 'a')

    def test_regressor_impurity_and_leaf(self):
        self.assertAlmostEqual(DecisionTreeRegressor(criterion='mse')._impurity([1, 2, 3]), 2 / 3)
        self.assertAlmostEqual(DecisionTreeRegressor(criterion='mae')._impurity([1, 2, 9]), 8 / 3)
        self.assertEqual(DecisionTreeRegressor()._leaf_value([1, 2, 3]), 2)

    def test_classifier_fit_predict_and_stopping(self):
        X = [[1], [2], [3], [4]]
        y = ['a', 'a', 'b', 'b']
        model = DecisionTreeClassifier().fit(X, y)
        self.assertEqual(model.root.threshold, 2.5)
        self.assertEqual(model.predict([[1], [2.5], [4]]).tolist(), ['a', 'a', 'b'])
        self.assertEqual(model.eval(X, y), 1.0)
        self.assertTrue(DecisionTreeClassifier(max_depth=0).fit(X, y).root.is_leaf)
        self.assertTrue(DecisionTreeClassifier().fit([[1], [1]], ['a', 'b']).root.is_leaf)

    def test_regressor_fit_predict_and_score(self):
        X = [[1], [2], [3], [4]]
        model = DecisionTreeRegressor(max_depth=1).fit(X, [1, 3, 9, 11])
        np.testing.assert_allclose(model.predict(X), [2, 2, 10, 10])
        self.assertAlmostEqual(model._score([1, 2, 3], [2, 2, 2]), 0.0)
        self.assertEqual(model._score([2, 2], [2, 2]), 1.0)
        self.assertEqual(model._score([2, 2], [1, 2]), 0.0)

    def test_vegetation_dataset(self):
        path = Path(__file__).resolve().parents[1] / 'dataset.csv'
        with path.open(newline='', encoding='utf-8') as file:
            rows = list(csv.DictReader(file))
        features = ('STREAM', 'SLOPE', 'ELEVATION')
        categories = {feature: sorted({row[feature] for row in rows}) for feature in features}
        X = [[int(row[feature] == category)
              for feature in features for category in categories[feature]] for row in rows]
        y = [row['VEGETATION'] for row in rows]
        model = DecisionTreeClassifier(criterion='entropy').fit(X, y)
        self.assertEqual(model.predict(X).tolist(), y)
        self.assertEqual(model.eval(X, y), 1.0)


if __name__ == '__main__':
    unittest.main()
