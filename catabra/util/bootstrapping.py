#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

from functools import partial
from typing import Optional, Union

import numpy as np
import pandas as pd


class Bootstrapping:
    """
    Class for performing bootstrapping [1], i.e., repeatedly sample with replacement from given data and evaluate
    statistics on each resample to obtain mean, standard deviation, etc. for more robust estimates.

    Parameters
    ----------
    *args: DataFrame | Series | ndarray
        Data, non-empty sequence of DataFrames, Series or arrays of the same length.
    kwargs: dict, optional
        Additional keyword arguments passed to the function `fn` computing the statistics. Like `args`, the values
        of the dict must be DataFrames, Series or arrays of the same length as the elements of `args`.
    fn: optional
        The statistics to compute. Must be None, a function that takes the given `args` as input and returns a
        scalar/array/DataFrame/Series or a (nest) dict/tuple thereof, or a (nested) dict/tuple of such functions.
    seed: int, optional
        Random seed.
    replace: bool, default=True
        Whether to resample with replacement. If False, this does not actually correspond to bootstrapping.
    size: int | float, default=1.
        The size of the resampled data. If <= 1, it is multiplied with the number of samples in the given data.
        Bootstrapping normally assumes that resampled data have the same number of samples as the original data, so
        this parameter should be set to 1.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
    """

    def __init__(self, *args: Union[pd.DataFrame, pd.Series, np.ndarray], kwargs: Optional[dict] = None,
                 fn=None, seed=None, replace: bool = True, size: Union[int, float] = 1.):

        self._args = args
        assert self._args
        assert all(isinstance(a, (pd.Series, pd.DataFrame, np.ndarray)) for a in self._args)
        self._n = len(args[0])
        assert all(len(a) == self._n for a in self._args[1:])
        self._kwargs = {} if kwargs is None else kwargs
        assert all(isinstance(a, (pd.Series, pd.DataFrame, np.ndarray)) for a in self._kwargs.values())
        assert all(len(a) == self._n for a in self._kwargs.values())
        self._fn = fn
        self._rng = np.random.RandomState(seed=seed)
        self._replace = replace
        self._size = size
        self._size_n = int(round(self._n * self._size)) if self._size <= 1 else self._size
        assert 1 <= self._size_n
        assert self._size_n <= self._n or self._replace
        self._results = None
        self._idx = []
        self._seeds = []

    def reset(self, seed=None):
        self._rng = np.random.RandomState(seed=seed)
        self._results = None
        self._idx = []
        self._seeds = []

    def run(self, n_repetitions: int = 100, sample_indices: Optional[np.ndarray] = None) -> 'Bootstrapping':
        """
        Run bootstrapping for a given number of repetitions, and store the results in a list. Results are appended to
        results from previous runs!

        Parameters
        ----------
        n_repetitions: int, default=100
            Number of repetitions.
        sample_indices: ndarray, optional
            Pre-computed sample indices to use in each repetition. If not None, n_repetitions` is ignored and
            `sample_indices` must have shape `(n, size)`.
        """
        if sample_indices is not None:
            assert sample_indices.ndim == 2
            assert sample_indices.shape[1] == self._size_n
        for i in range(n_repetitions):
            if sample_indices is None:
                seed = self._rng.randint(2 ** 32)
                idx = self.subsample(seed=seed)
            else:
                seed = None
                idx = sample_indices[i]
            self._idx.append(idx)
            self._seeds.append(seed)
            res = Bootstrapping._apply_function(
                self._fn,
                [a[idx] if isinstance(a, np.ndarray) else a.iloc[idx] for a in self._args],
                {k: a[idx] if isinstance(a, np.ndarray) else a.iloc[idx] for k, a in self._kwargs.items()}
            )
            if self._results is None:
                self._results = Bootstrapping._init_results(res)
            else:
                Bootstrapping._update_results(self._results, res)
        return self

    def subsample(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Construct a subsample.

        Parameters
        ----------
        seed: int, optional
            Random seed to use.

        Returns
        -------
        ndarray
            Array with subsample indices.
        """

        # create new random state to make result independent of previous calls to this method
        rng = np.random.RandomState(seed=seed)
        return rng.choice(self._n, size=self._size_n, replace=self._replace)

    @property
    def replace(self) -> bool:
        return self._replace

    @property
    def size(self) -> Union[int, float]:
        return self._size

    @property
    def n_runs(self) -> int:
        return len(self._idx)

    @property
    def results(self):
        return self._results

    @property
    def seeds(self) -> list:
        return self._seeds

    def get_sample_indices(self) -> np.ndarray:
        """
        Get sample indices used for resampling the data.

        Returns
        -------
        ndarray
            Array of shape `(n_runs, size)`.
        """
        return np.stack(self._idx, axis=0) if self._idx else None

    def agg(self, func):
        """
        Compute aggregate statistics of the results of the individual runs, like mean, standard deviation, etc.

        Parameters
        ----------
        func:
            The aggregation function to apply.

        Returns
        -------
        Any
            Aggregated results.
        """
        if isinstance(func, str):
            func = getattr(np, func)
        return Bootstrapping._aggregate(self._results, func)

    def mean(self):
        return Bootstrapping._aggregate(self._results, np.mean)

    def std(self):
        return Bootstrapping._aggregate(self._results, np.std)

    def var(self):
        return Bootstrapping._aggregate(self._results, np.var)

    def min(self):
        return Bootstrapping._aggregate(self._results, np.min)

    def max(self):
        return Bootstrapping._aggregate(self._results, np.max)

    def median(self):
        return Bootstrapping._aggregate(self._results, np.median)

    def quantile(self, q=0.5):
        if isinstance(q, (list, tuple, np.ndarray)):
            return [Bootstrapping._aggregate(self._results, partial(np.quantile, q=qi)) for qi in q]
        else:
            return Bootstrapping._aggregate(self._results, partial(np.quantile, q=q))

    def sum(self):
        return Bootstrapping._aggregate(self._results, np.sum)

    def dataframe(self, keys=None) -> Optional[pd.DataFrame]:
        """
        Construct a DataFrame with all results, if possible. Only works for (dicts/tuples of) scalar values.

        Returns
        -------
        DataFrame
            DataFrame whose columns correspond to individual metrics and whose rows correspond to runs, or None.
        """
        if isinstance(self._results, list):
            if all(np.isscalar(r) for r in self._results):
                return pd.Series(self._results).to_frame('')
            else:
                res = {}
        elif isinstance(self._results, tuple):
            res = dict(enumerate(self._results))
        elif isinstance(self._results, dict):
            res = self._results
        else:
            res = {}
        if keys is None:
            keys = res.keys()
        elif not isinstance(keys, (list, set, tuple)):
            keys = [keys]
        res = {k: v for k, v in res.items() if k in keys and isinstance(v, list) and all(np.isscalar(x) for x in v)}
        if res:
            return pd.DataFrame(res)
        else:
            return None

    def describe(self, keys=None) -> Union[pd.Series, pd.DataFrame]:
        """
        Describe the results of the individual runs by computing a predefined set of statistics, similar to pandas'
        `describe()` method. Only works for (dicts/tuples of) scalar values.

        Returns
        -------
        Series | DataFrame
            DataFrame or Series with descriptive statistics.
        """
        df = self.dataframe(keys=keys)
        if df is None:
            return pd.DataFrame(index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
        elif df.shape[1] == 1 and df.columns[0] == '':
            return df.iloc[:, 0].describe()
        else:
            return df.describe()

    @staticmethod
    def _apply_function(fn, args: list, kwargs: dict):
        if fn is None:
            # ignore `kwargs`
            return tuple(args)
        elif isinstance(fn, dict):
            return {k: Bootstrapping._apply_function(v, args, kwargs) for k, v in fn.items()}
        elif isinstance(fn, tuple):
            return tuple(Bootstrapping._apply_function(f, args, kwargs) for f in fn)
        else:
            try:
                return fn(*args, **kwargs)
            except:     # noqa
                # It could be that due to resampling some metrics raise exceptions (e.g., if no condition positives
                # appear anymore). We catch all exceptions and simply return NaN in this case.
                return np.nan

    @staticmethod
    def _init_results(new):
        if isinstance(new, dict):
            return {k: Bootstrapping._init_results(v) for k, v in new.items()}
        elif isinstance(new, tuple):
            return tuple(Bootstrapping._init_results(n) for n in new)
        else:
            return [new]

    @staticmethod
    def _update_results(agg, new):
        if isinstance(new, dict):
            assert isinstance(agg, dict)
            for k, v in new.items():
                Bootstrapping._update_results(agg[k], v)
        elif isinstance(new, tuple):
            assert isinstance(agg, tuple)
            assert len(agg) == len(new)
            for a, n in zip(agg, new):
                Bootstrapping._update_results(a, n)
        else:
            assert isinstance(agg, list)
            agg.append(new)

    @staticmethod
    def _aggregate(res, fn):
        if isinstance(res, dict):
            return {k: Bootstrapping._aggregate(v, fn) for k, v in res.items()}
        elif isinstance(res, tuple):
            return tuple(Bootstrapping._aggregate(r, fn) for r in res)
        elif isinstance(res, list):
            if res:
                if isinstance(res[0], pd.DataFrame):
                    out = pd.DataFrame(index=res[0].index)
                    for c in res[0].columns:
                        out[c] = fn([r[c].values for r in res], axis=0)
                    return out
                elif isinstance(res[0], pd.Series):
                    return pd.Series(fn([r.values for r in res], axis=0), index=res[0].index)
                else:
                    return fn(res, axis=0)
            else:
                return fn(res)
        else:
            return res
