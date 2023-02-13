from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from ..base import OODDetector
from ..utils import make_standard_transformer
from catabra.util import io
from scipy.stats import ks_2samp


class KSTest(OODDetector):
    """
    Two sample Kolmogorov-Smirnov test.
    Refer to: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    Hypothesis test for the following question:
    "How likely is it that we would see two sets of samples like this if they were drawn from the same (but unknown)
    probability distribution?"
    """
    _data_artefact_file = 'ks_data_artefact.h5'

    def __init__(self, subset=1, p_val=0.05, random_state: int=None, verbose=True):
        """
        Initialization of Autoencoder
        :param: p_val: p-value to decide statistical significance
        """
        super().__init__(subset=subset, verbose=verbose)
        self._p_val = p_val
        self._random_state = np.random.randint(1000) if random_state is None else random_state
        self._transformer = make_standard_transformer()
        self._subset_indices = None


    @property
    def p_val(self):
        return self._p_val

    @property
    def random_state(self):
        return self._random_state

    def _fit_transformer(self, X: pd.DataFrame):
        self._transformer.fit(X)

    def _transform(self, X: pd.DataFrame):
        return self._transformer.transform(X)

    def _fit_transformed(self, X: pd.DataFrame, y: pd.Series):
        io.write_df(X, self._data_artefact_file)

    def _predict_transformed(self, X):
        return (self._p_val <= 1 - self._predict_proba_transformed(X)).astype(int)

    def _predict_proba_transformed(self, X):
        orig_data = io.read_df(self._data_artefact_file)

        results = pd.Series(np.arange(orig_data.shape[1]), index=orig_data.columns)
        return 1 - results.apply(lambda i: ks_2samp(orig_data.iloc[:, i], X.iloc[:, i])[1])
