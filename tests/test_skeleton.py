import pytest

from decision_tree import BaseDecisionTree, DecisionTreeClassifier, DecisionTreeRegressor


def test_base_is_abstract():
    with pytest.raises(TypeError):
        BaseDecisionTree()


@pytest.mark.parametrize("cls", [DecisionTreeClassifier, DecisionTreeRegressor])
def test_hyperparameters(cls):
    model = cls(max_depth=3, min_samples_split=4)
    assert model.get_params()["max_depth"] == 3
    model.set_params(criterion="entropy")
    assert model.criterion == "entropy"
    with pytest.raises(ValueError):
        model.set_params(foo=1)


@pytest.mark.parametrize("cls", [DecisionTreeClassifier, DecisionTreeRegressor])
def test_students_must_implement(cls):
    model = cls()
    for call in (lambda: model.fit([[0]], [0]), lambda: model.predict([[0]]),
                 lambda: model.eval([[0]], [0])):
        with pytest.raises(NotImplementedError):
            call()
