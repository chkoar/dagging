import pytest
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.utils.estimator_checks import check_estimator

from dagging import DaggingClassifier, DaggingRegressor


def test_check_dagging_classifier():
    model = DaggingClassifier(random_state=0, n_estimators=2)
    check_estimator(model)


def test_check_dagging_regressor():
    model = DaggingRegressor(random_state=0, n_estimators=2)
    check_estimator(model)


def test_dagging_classifier_hard_voting():
    X, y = load_iris(True)
    model = DaggingClassifier(random_state=0, n_estimators=3, voting='hard')
    model.fit(X, y)
    model.score(X, y)


def test_dagging_classifier_soft_voting():
    X, y = load_iris(True)
    model = DaggingClassifier(random_state=0, n_estimators=3, voting='soft')
    model.fit(X, y)
    model.score(X, y)


def test_fail_dagging_classifier_wrong_voting():
    X, y = load_iris(True)
    with pytest.raises(ValueError):
        model = DaggingClassifier(random_state=0, n_estimators=2, voting='lol')
        model.fit(X, y)


def test_fail_dagging_classifier_hard_voting():
    X, y = load_iris(True)
    with pytest.raises(AttributeError):
        model = DaggingClassifier(base_estimator=SVC(),
                                  random_state=0,
                                  n_estimators=3,
                                  voting='hard')
        model.fit(X, y)
        model.predict_proba(X)
