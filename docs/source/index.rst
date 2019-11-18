.. dagging documentation master file, created by
   sphinx-quickstart on Tue Nov 19 00:45:37 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to dagging's documentation!
===============================

The dagging package implements an ensemble method that is called Dagging [1]_. 
Dagging creates a number of disjoint, stratified folds out of the data and
feeds each chunk of data to a copy of the supplied base classifier.
Predictions are made using the average of class membership probabilities
if the base estimator outputs probabilities otherwise via plurality vote.

.. [1] Ting, K. M., Witten, I. H.: Stacking Bagged and Dagged Models. 
       In: Fourteenth international Conference on Machine Learning,
       San Francisco, CA, 367-375, 1997.



------------------
How to use dagging
------------------

The dagging package inherits from sklearn classes, and thus drops in neatly
next to other sklearn classifiers with an identical calling API. Similarly it
supports input in a variety of formats: an array (or pandas dataframe) of
shape ``(num_samples x num_features)``.

.. code:: python

    import dagging
    from sklearn.datasets import load_iris
    
    data, target = load_iris(return_X_y=True)
    
    model = dagging.DaggingClassifier(random_state=0)
    model.fit(data, target)
    model.predict(data)


.. toctree::
    :maxdepth: 1

    installation_guide
    auto_examples/index
    api

