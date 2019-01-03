from ._version import get_versions
from ._dagging import DaggingClassifier, DaggingRegressor  # noqa

__version__ = get_versions()['version']
del get_versions
