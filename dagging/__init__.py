from ._version import get_versions
from ._dagging import Dagging, DaggingClassifier # noqa

__version__ = get_versions()['version']
del get_versions
