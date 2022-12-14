import numpy as np
import pandas as pd
from sklearn import model_selection


class CustomPredefinedSplit(model_selection.BaseCrossValidator):
    """
    Predefined split cross-validator. Provides train/test indices to split data into train/test sets using a predefined
    scheme specified by explicit test indices.

    In contrast to `sklearn.model_selection.PredefinedSplit`, samples can be in the test set of more than one split.

    In methods `split()` etc., parameters `X`, `y` and `groups` only exist for compatibility, but are always ignored.

    Parameters
    ----------
    test_folds : list of array-like
        Indices of test samples for each split. The number of splits equals the length of the list.
        Note that the test sets do not have to be mutually disjoint.
    """

    def __init__(self, test_folds=None):
        assert test_folds is not None
        if isinstance(test_folds, (np.ndarray, pd.Series)):
            test_folds = [test_folds]
        self.test_folds = test_folds

    @classmethod
    def from_data(cls, X: pd.DataFrame, columns) -> 'CustomPredefinedSplit':
        if isinstance(columns, (str, int)):
            columns = [columns]
        assert all(X[c].dtype.kind == 'b' for c in columns)
        return cls(test_folds=[np.flatnonzero(X[c].values) for c in columns])

    def _iter_test_indices(self, X=None, y=None, groups=None):
        for indices in self.test_folds:
            yield indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.test_folds)
