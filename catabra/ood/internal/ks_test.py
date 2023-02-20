from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..base import FeaturewiseOODDetector
from ..utils import make_standard_transformer
from catabra.util import io
from scipy.stats import ks_2samp


class KSTest(FeaturewiseOODDetector):
    """
    Two sample Kolmogorov-Smirnov test.
    Refer to: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    Hypothesis test for the following question:
    "How likely is it that we would see two sets of samples like this if they were drawn from the same (but unknown)
    probability distribution?"
    """

    def __init__(self, subset=1, p_val=0.05, random_state: int=None, verbose=True):
        """
        Initialization of KS Test
        :param: p_val: p-value to decide statistical significance
        """
        super().__init__(subset=subset, verbose=verbose)
        self._p_val = p_val
        self._random_state = np.random.randint(1000) if random_state is None else random_state
        # self._transformer = make_standard_transformer()
        self._subset_indices = None
        self._num_cols: Optional[np.ndarray] = None
        self._train_data: pd.DataFrame = None

    @property
    def p_val(self):
        return self._p_val

    @property
    def random_state(self):
        return self._random_state

    @property
    def num_cols(self) -> Optional[np.ndarray]:
        return self._num_cols

    def _fit_transformer(self, X: pd.DataFrame):
        # self._transformer.fit(X)
        cnts = X.apply(lambda x: x.nunique())
        X = X.drop(list(cnts[cnts <= 2].index), axis=1)
        self._num_cols = X.select_dtypes(np.number).columns.values

    def _transform(self, X: pd.DataFrame):
        # return self._transformer.transform(X)
        return X[self._num_cols]

    def _fit_transformed(self, X: pd.DataFrame, y: pd.Series):
        self._train_data = X

    def _predict_transformed(self, X):
        return ((1 - self._predict_proba_transformed(X)) <= self._p_val).astype(int)

    def _predict_proba_transformed(self, X):

        results = pd.Series(np.arange(self._train_data.shape[1]), index=self._train_data.columns)
        progress = tqdm(total=X.shape[1])

        def __apply_ks_test(i: int):
            ks = ks_2samp(self._train_data.iloc[:, i], X.iloc[:, i])[1]
            progress.update()
            return ks

        return 1 - results.apply(__apply_ks_test)
