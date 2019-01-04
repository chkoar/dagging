import numpy as np

from sklearn.base import ClassifierMixin, RegressorMixin, is_classifier
from sklearn.ensemble.base import BaseEnsemble
from sklearn.model_selection import check_cv
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, check_random_state


class BaseDagging(BaseEnsemble):
    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 voting='soft',
                 random_state=None):
        super(BaseDagging, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators)
        self.voting = voting
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit the estimators.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.
        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y)

        if self.voting not in ('soft', 'hard'):
            raise ValueError("Voting must be 'soft' or 'hard'; got (voting=%r)"
                             % self.voting)

        self._validate_estimator()
        if is_classifier(self.base_estimator_):
            check_classification_targets(y)
            if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1: # noqa:
                raise NotImplementedError('Multilabel and multi-output'
                                          ' classification is not supported.')
            self.le_ = LabelEncoder().fit(y)
            self.classes_ = self.le_.classes_
            transformed_y = self.le_.transform(y)
        else:
            transformed_y = y

        self.estimators_ = []

        rs = check_random_state(self.random_state)
        splitter = check_cv(cv=self.n_estimators,
                            y=transformed_y,
                            classifier=is_classifier(self.base_estimator_))

        for _, index in splitter.split(X, transformed_y):
            estimator = self._make_estimator(append=False,
                                             random_state=rs)
            estimator.fit(X[index], transformed_y[index])
            self.estimators_.append(estimator)

        return self


class DaggingClassifier(BaseDagging, ClassifierMixin):
    """A Dagging classifier.
    This meta classifier creates a number of disjoint, stratified folds out of
    the data and feeds each chunk of data to a copy of the supplied base
    classifier. Predictions are made via hard or soft voting.
    Useful for base classifiers that are quadratic or worse in time behavior,
    regarding number of instances in the training data.

    Parameters
    ----------
    base_estimator : object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.
    n_estimators : int, optional (default=3)
        The number of base estimators in the ensemble.
    voting : str, {'hard', 'soft'} (default='soft')
        If 'hard', uses predicted class labels for majority rule voting.
        Else if 'soft', predicts the class label based on the argmax of
        the sums of the predicted probabilities, which is recommended for
        an ensemble of well-calibrated classifiers.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.
    estimators_ : list of estimators
        The collection of fitted base estimators.
    References
    ----------
    .. [1] Ting, K. M., Witten, I. H.: Stacking Bagged and Dagged Models.
           In: Fourteenth international Conference on Machine Learning,
           San Francisco, CA, 367-375, 1997
    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 voting='soft',
                 random_state=None):
        super(DaggingClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            voting=voting,
            random_state=random_state)

    def predict(self, X):
        """ Predict class labels for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.
        """

        check_is_fitted(self, 'estimators_')
        if self.voting == 'soft':
            maj = np.argmax(self.predict_proba(X), axis=1)

        else:  # 'hard' voting
            predictions = self._predict(X)
            maj = np.apply_along_axis(
                lambda x: np.argmax(
                    np.bincount(x)),
                axis=1, arr=predictions)

        maj = self.le_.inverse_transform(maj)

        return maj

    def _predict(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.estimators_]).T

    def _collect_probas(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict_proba(X) for clf in self.estimators_])

    def _predict_proba(self, X):
        """Predict class probabilities for X in 'soft' voting """
        if self.voting == 'hard':
            raise AttributeError("predict_proba is not available when"
                                 " voting=%r" % self.voting)
        check_is_fitted(self, 'estimators_')
        avg = np.average(self._collect_probas(X), axis=0)
        return avg

    @property
    def predict_proba(self):
        """Compute probabilities of possible outcomes for samples in X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        ----------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """
        return self._predict_proba

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(DaggingClassifier, self)._validate_estimator(
            default=DecisionTreeClassifier())


class DaggingRegressor(BaseDagging, RegressorMixin):
    """A Dagging regressor.
    This meta regressor creates a number of disjoint, stratified folds out of
    the data and feeds each chunk of data to a copy of the supplied base
    regressor. Predictions are made via hard or soft voting.
    Useful for base regressor that are quadratic or worse in time behavior,
    regarding number of instances in the training data.

    Parameters
    ----------
    base_estimator : object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.
    n_estimators : int, optional (default=3)
        The number of base estimators in the ensemble.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.
    estimators_ : list of estimators
        The collection of fitted base estimators.
    References
    ----------
    .. [1] Ting, K. M., Witten, I. H.: Stacking Bagged and Dagged Models.
           In: Fourteenth international Conference on Machine Learning,
           San Francisco, CA, 367-375, 1997
    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 random_state=None):
        super(DaggingRegressor, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            random_state=random_state)

    def predict(self, X):
        """ Predict class labels for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.
        """
        check_is_fitted(self, 'estimators_')
        predictions = []
        for estimator in self.estimators_:
            predictions.append(estimator.predict(X))
        return np.average(predictions, axis=0)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(DaggingRegressor, self)._validate_estimator(
            default=DecisionTreeRegressor())
