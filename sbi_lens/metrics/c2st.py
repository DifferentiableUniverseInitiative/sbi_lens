from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
import numpy as np


def c2st(X, Y, seed, n_folds):
  """ Binary classifier with 2 hidden layers of 10x dim each,
    following the architecture of Benchmarking Simulation-Based Inference
    https://github.com/sbi-benchmark/sbibm/blob/main/sbibm/metrics/c2st.py

    Parameters
    ----------
    X : Array
        First sample
    Y : Array
        Second sample
    seed : int
        Seed for sklearn
    n_folds : int
         Number of folds

    Returns
    -------
    Array
        Score
    """

  X_mean = np.mean(X, axis=0)
  X_std = np.std(X, axis=0)
  X = (X - X_mean) / X_std
  Y = (Y - X_mean) / X_std

  ndim = X.shape[1]

  clf = MLPClassifier(
      activation="relu",
      hidden_layer_sizes=(10 * ndim, 10 * ndim),
      max_iter=10000,
      solver="adam",
      random_state=seed,
  )

  data = np.concatenate((X, Y))
  target = np.concatenate((
      np.zeros((X.shape[0], )),
      np.ones((Y.shape[0], )),
  ))

  shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
  scores = cross_val_score(clf, data, target, cv=shuffle, scoring="accuracy")

  scores = np.asarray(np.mean(scores)).astype(np.float32)
  return scores
