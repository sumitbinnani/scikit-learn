Conventions
-----------

scikit-learn estimators follow certain rules to make their behavior more
predictive.


Type casting
~~~~~~~~~~~~

Unless otherwise specified, input will be cast to ``float64``::

  >>> import numpy as np
  >>> from sklearn import random_projection

  >>> rng = np.random.RandomState(0)
  >>> X = rng.rand(10, 2000)
  >>> X = np.array(X, dtype='float32')
  >>> X.dtype
  dtype('float32')

  >>> transformer = random_projection.GaussianRandomProjection()
  >>> X_new = transformer.fit_transform(X)
  >>> X_new.dtype
  dtype('float64')

In this example, ``X`` is ``float32``, which is cast to ``float64`` by
``fit_transform(X)``.

Regression targets are cast to ``float64``, classification targets are
maintained::

    >>> from sklearn import datasets
    >>> from sklearn.svm import SVC
    >>> iris = datasets.load_iris()
    >>> clf = SVC()
    >>> clf.fit(iris.data, iris.target)  # doctest: +NORMALIZE_WHITESPACE
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)

    >>> list(clf.predict(iris.data[:3]))
    [0, 0, 0]

    >>> clf.fit(iris.data, iris.target_names[iris.target])  # doctest: +NORMALIZE_WHITESPACE
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)

    >>> list(clf.predict(iris.data[:3]))  # doctest: +NORMALIZE_WHITESPACE
    ['setosa', 'setosa', 'setosa']

Here, the first ``predict()`` returns an integer array, since ``iris.target``
(an integer array) was used in ``fit``. The second ``predict`` returns a string
array, since ``iris.target_names`` was for fitting.


Refitting and updating parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hyper-parameters of an estimator can be updated after it has been constructed
via the :func:`sklearn.pipeline.Pipeline.set_params` method. Calling ``fit()``
more than once will overwrite what was learned by any previous ``fit()``::

  >>> import numpy as np
  >>> from sklearn.svm import SVC

  >>> rng = np.random.RandomState(0)
  >>> X = rng.rand(100, 10)
  >>> y = rng.binomial(1, 0.5, 100)
  >>> X_test = rng.rand(5, 10)

  >>> clf = SVC()
  >>> clf.set_params(kernel='linear').fit(X, y)  # doctest: +NORMALIZE_WHITESPACE
  SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
  >>> clf.predict(X_test)
  array([1, 0, 1, 1, 0])

  >>> clf.set_params(kernel='rbf').fit(X, y)  # doctest: +NORMALIZE_WHITESPACE
  SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
  >>> clf.predict(X_test)
  array([0, 0, 0, 1, 0])

Here, the default kernel ``rbf`` is first changed to ``linear`` after the
estimator has been constructed via ``SVC()``, and changed back to ``rbf`` to
refit the estimator and to make a second prediction.
