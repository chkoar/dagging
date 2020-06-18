import pytest

from sklearn.datasets import load_iris, load_boston
from sklearn.naive_bayes import GaussianNB
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
    X, y = load_iris(return_X_y=True)
    model = DaggingClassifier(
        SVC(gamma="scale", random_state=0), random_state=0, n_estimators=3
    )
    model.fit(X, y)
    model.score(X, y)
    assert model.voting_ == "hard"


def test_dagging_classifier_soft_voting():
    X, y = load_iris(return_X_y=True)
    model = DaggingClassifier(GaussianNB(), random_state=0, n_estimators=3)
    model.fit(X, y)
    model.score(X, y)
    assert model.voting_ == "soft"


def test_dagging_classifier_too_many_estimators():
    X, y = load_iris(return_X_y=True)
    model = DaggingClassifier(GaussianNB(), random_state=0, n_estimators=300)
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_dagging_regressor_too_many_estimators():
    X, y = load_boston(return_X_y=True)
    model = DaggingRegressor(random_state=0, n_estimators=3000)
    with pytest.raises(ValueError):
        model.fit(X, y)
