===============================
Dagging
===============================

.. image:: https://github.com/chkoar/dagging/workflows/Python%20application/badge.svg
        :target: https://travis-ci.org/chkoar/dagging/actions

.. image:: https://codecov.io/gh/chkoar/dagging/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/chkoar/dagging

.. image:: https://readthedocs.org/projects/dagging/badge/?version=latest
  :target: https://dagging.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/dagging.svg
        :target: https://pypi.python.org/pypi/dagging


Python package implementing the dagging method

* Free software: 3-clause BSD license

Example
-------

.. code-block:: python

    from dagging import DaggingClassifier
    from sklearn.datasets import load_iris 

    # Load Iris from from scikit-learn.
    X, y = load_iris(True)

    model = DaggingClassifier(n_estimators=10, random_state=0)

    # Train the model.
    model.fit(X,y)

    # Accuracy
    print(model.score(X, y))


Dependencies
------------

The dependency requirements are based on the last scikit-learn release:

* scipy
* numpy
* scikit-learn

Installation
------------

dagging is currently available on the PyPi's repository and you can
install it via `pip`::

  pip install -U dagging

If you prefer, you can clone it and run the setup.py file. Use the following
commands to get a copy from GitHub and install all dependencies::

  git clone https://github.com/chkoar/dagging.git
  cd dagging
  pip install .

Or install using pip and GitHub::

  pip install -U git+https://github.com/chkoar/dagging.git
