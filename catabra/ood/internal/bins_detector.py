from typing import Union, Optional, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from catabra.ood.base import SamplewiseOODDetector


class BinsDetector(SamplewiseOODDetector):
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
        self._num_cols: Optional[np.ndarray] = None
        self._empty_bins: Optional[pd.DataFrame] = None
        self._min_max: Optional[pd.DataFrame] = None

    @property
    def empty_bins(self) -> Union[None, int, pd.DataFrame]:
        return self._empty_bins

    @property
    def bins(self) -> Union[None, pd.DataFrame]:
        return self._bins

    @property
    def num_cols(self) -> Optional[np.ndarray]:
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

        progress = tqdm(total=len(self._num_cols))

        def _get_empty_bins(col: pd.Series):
            bins = int(self._bins[col.name] if isinstance(self._bins, pd.Series) else self._bins)
            cnts, edges = np.histogram(col.dropna(), bins)
            zero_bins = np.where(np.array(cnts) == 0)[0]
            empties = list(zip(edges[zero_bins], edges[zero_bins + 1]))
            progress.update()
            if len(empties) == 0:
                return []
            else:
                return empties

        self._empty_bins = X.apply(_get_empty_bins)

    def _predict_transformed(self, X: pd.DataFrame):
        return self._predict_proba_transformed(X) > 0

    def _predict_proba_transformed(self, X: pd.DataFrame) -> pd.Series:
        """
        Returns distance to bin edges normalized by bin width if value falls within an empty bin
        """
        left, right = self._get_bins_transformed(X)
        X = X.reset_index().drop('index', axis=1)
        left_dist = np.abs(left - X)
        right_dist = np.abs(right - X)
        bin_width = (right - left) / 2

        dist = pd.DataFrame(np.zeros_like(X) * np.nan, columns=X.columns)
        cond_left = np.where((left_dist < right_dist) | (~left_dist.isna() & right_dist.isna()))
        dist.iloc[cond_left[0], cond_left[1]] = left_dist.iloc[cond_left[0], cond_left[1]]
        cond_right = np.where((left_dist >= right_dist) | (left_dist.isna() & ~right_dist.isna()))
        dist.iloc[cond_right[0], cond_right[1]] = right_dist.iloc[cond_right[0], cond_right[1]]

        return (dist / bin_width).fillna(0).max(axis=1)

    def get_bins(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self._get_bins_transformed(self._transform(X))

    def _get_bins_transformed(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        progress = tqdm(total=len(self._num_cols))

        def _is_in_bin(col: pd.Series):
            right_edge = np.array([np.nan] * col.shape[0])
            left_edge = np.array([np.nan] * col.shape[0])

            for bin in self._empty_bins[col.name]:
                inside = (col > bin[0]) & (col < bin[1])
                left_edge[inside] = bin[0]
                right_edge[inside] = bin[1]
            left_edge[col > self._min_max.loc[col.name, 'max']] = self._min_max.loc[col.name, 'max']
            right_edge[col < self._min_max.loc[col.name, 'min']] = self._min_max.loc[col.name, 'min']

            progress.update()
            return np.hstack([left_edge, right_edge])

        stacked = X.apply(_is_in_bin)
        left, right = stacked.iloc[:X.shape[0],:], stacked.iloc[X.shape[0]:,:]
        return left.reset_index().drop('index', axis=1), right.reset_index().drop('index', axis=1)
