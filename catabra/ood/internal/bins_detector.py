from typing import Union, Optional, List

import numpy as np
import pandas as pd

from catabra.ood import OODDetector
from catabra.util.logging import log


class BinsDetector(OODDetector):
    """
    Simple OOD detector that distributes the training set into equally sized bins.
    A sample is considered OOD if it falls within one such bin
    """

    def __init__(self, subset=1, bins: Union[None, pd.DataFrame, int]=None, random_state: int=None, verbose=True):
        """
        :param bins: Number of bins for each column
        if int each column uses the same amount of bins
        defaults to 2 * std for each columns
        """
        super().__init__(subset=subset, random_state=random_state, verbose=verbose)
        self._bins: Union[None, int, pd.Series] = bins
        self._num_cols: Optional[np.ndarray[str]] = None
        self._empty_bins: Optional[pd.DataFrame] = None
        self._min_max: Optional[pd.DataFrame] = None


    @property
    def empty_bins(self) -> Union[None, int, pd.DataFrame]:
        return self._empty_bins

    @property
    def bins(self) -> Union[None, pd.DataFrame]:
        return self._bins

    @property
    def num_cols(self) -> Optional[np.ndarray[str]]:
        return self._num_cols

    def _fit_transformer(self, X: pd.DataFrame):
        self._num_cols = X.select_dtypes(np.number).columns.values

    def _transform(self, X: pd.DataFrame):
        return X[self._num_cols]

    def _fit_transformed(self, X: pd.DataFrame, y: pd.Series):
        std = X.std()
        self._min_max = pd.DataFrame({'min': X.min(), 'max': X.max()})
        if self._bins is None:
            span = self._min_max['max'] - self._min_max['min']
            self._bins = pd.Series(np.round(span / (2 * std)), index=X.columns)

        def _get_empty_bins(col: pd.Series):
            bins = int(self._bins[col.name] if isinstance(self._bins, pd.Series) else self._bins)
            cnts, edges = np.histogram(col.dropna(), bins)
            zero_bins = np.where(np.array(cnts) == 0)[0]
            empties = list(zip(edges[zero_bins], edges[zero_bins + 1]))
            if len(empties) == 0:
                return []
            else:
                return empties

        self._empty_bins = X.apply(_get_empty_bins)

    def _predict_transformed(self, X: pd.DataFrame):
        return np.any(~self._predict_proba_transformed(X).isna(), axis=0)

    def _predict_proba_transformed(self, X) -> pd.DataFrame:
        log('Warning: Bin Detector cannot predict any probabilities. Only whether a samples falls within an empty bin.')

        def _is_in_bin(col: pd.Series):
            bins = np.array([None] * col.shape[0])

            for bin in self._empty_bins[col.name]:
                inside = (col > bin[0]) & (col < bin[1])
                bins[inside] = str(bin)
            bins[col > self._min_max.loc[col.name, 'max']] = '> max'
            bins[col < self._min_max.loc[col.name, 'min']] = '< min'

            return bins

        return X.apply(_is_in_bin)

