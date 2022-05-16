from typing import Union, Optional
from functools import partial
import numpy as np
import pandas as pd


class Bootstrapping:

    def __init__(self, *args: Union[pd.DataFrame, pd.Series, np.ndarray], fn=None, seed=None, replace: bool = True,
                 size: Union[int, float] = 1.):
        """
        Perform bootstrapping [1], i.e., repeatedly sample with replacement from given data and evaluate statistics
        on each resample to obtain mean, standard deviation, etc. for more robust estimates.

        [1] https://en.wikipedia.org/wiki/Bootstrapping_(statistics)

        :param args: Data, non-empty sequence of DataFrames, Series or arrays of the same length.
        :param fn: The statistics to compute. Must be None, a function that takes the given `args` as input and returns
        a scalar/array/DataFrame/Series or a (nest) dict/tuple thereof, or a (nested) dict/tuple of such functions.
        :param seed: Optional, random seed.
        :param replace: Whether to resample with replacement. If False, this does not actually correspond to
        bootstrapping.
        :param size: The size of the resampled data. If <= 1, it is multiplied with the number of samples in the given
        data. Bootstrapping normally assumes that resampled data have the same number of samples as the original data,
        so this parameter should be set to 1.
        """
        self._args = args
        assert self._args
        assert all(isinstance(a, (pd.Series, pd.DataFrame, np.ndarray)) for a in self._args)
        self._n = len(args[0])
        assert all(len(a) == self._n for a in self._args[1:])
        self._fn = fn
        self._rng = np.random.RandomState(seed=seed)
        self._replace = replace
        self._size = size
        self._size_n = int(round(self._n * self._size)) if self._size <= 1 else self._size
        assert 1 <= self._size_n
        assert self._size_n <= self._n or self._replace
        self._results = None
        self._idx = []

    def reset(self, seed=None):
        self._rng = np.random.RandomState(seed=seed)
        self._results = None
        self._idx = []

    def run(self, n_repetitions: int = 100, sample_indices: Optional[np.ndarray] = None) -> 'Bootstrapping':
        """
        Run bootstrapping for a given number of repetitions, and store the results in a list. Results are appended to
        results from previous runs!
        :param n_repetitions: Number of repetitions.
        :param sample_indices: Optional, pre-computed sample indices to use in each repetition. If not None,
        `n_repetitions` is ignored and `sample_indices` must have shape `(n, size)`.
        """
        if sample_indices is not None:
            assert sample_indices.ndim == 2
            assert sample_indices.shape[1] == self._size_n
        for i in range(n_repetitions):
            idx = self._rng.choice(self._n, size=self._size_n, replace=self._replace) \
                if sample_indices is None else sample_indices[i]
            self._idx.append(idx)
            res = Bootstrapping._apply_function(
                self._fn,
                [a[idx] if isinstance(a, np.ndarray) else a.iloc[idx] for a in self._args]
            )
            if self._results is None:
                self._results = Bootstrapping._init_results(res)
            else:
                Bootstrapping._update_results(self._results, res)
        return self

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

    def get_sample_indices(self) -> np.ndarray:
        """
        Get sample indices used for resampling the data.
        :return: Array of shape `(n_runs, size)`.
        """
        return np.stack(self._idx, axis=0) if self._idx else None

    def agg(self, func):
        """
        Compute aggregate statistics of the results of the individual runs, like mean, standard deviation, etc.
        :param func: The aggregation function to apply.
        :return: Aggregated results.
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

    def describe(self, keys=None) -> Union[pd.Series, pd.DataFrame]:
        """
        Describe the results of the individual runs by computing a predefined set of statistics, similar to pandas'
        `describe()` method. Only works for (dicts/tuples of) scalar values.
        :return: DataFrame or Series with descriptive statistics.
        """
        if isinstance(self._results, list):
            if all(np.isscalar(r) for r in self._results):
                return pd.Series(self._results).describe()
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
            return pd.DataFrame(res).describe()
        else:
            return pd.DataFrame(index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])

    @staticmethod
    def _apply_function(fn, args: list):
        if fn is None:
            return tuple(args)
        elif isinstance(fn, dict):
            return {k: Bootstrapping._apply_function(v, args) for k, v in fn.items()}
        elif isinstance(fn, tuple):
            return tuple(Bootstrapping._apply_function(f, args) for f in fn)
        else:
            return fn(*args)

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
