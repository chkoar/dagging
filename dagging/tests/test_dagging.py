from sklearn.utils.estimator_checks import check_estimator

from dagging import Dagging


def test_check_estimator():
    model = Dagging(random_state=0)
    check_estimator(model)
