from sklearn.utils.estimator_checks import check_estimator

from dagging import DaggingClassifier, DaggingRegressor


def test_check_dagging_classifier():
    model = DaggingClassifier(random_state=0, n_estimators=2)
    check_estimator(model)


def test_check_dagging_regressor():
    model = DaggingRegressor(random_state=0, n_estimators=2)
    check_estimator(model)
