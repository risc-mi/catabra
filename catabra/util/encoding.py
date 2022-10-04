from typing import Union, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

from . import common as cu
from ..base import io


class Encoder(BaseEstimator):
    """
    Encoder for features- and labels DataFrames. Implements the `BaseEstimator` class of sklearn, with methods `fit()`,
    `transform()` and `inverse_transform()`, and can easily be dumped to and loaded from disk.

    Encoding ensures that:
    * The data type of every feature column is either float, int, bool or categorical. This is achieved by converting
        time-like columns into float and raising exceptions if object data types are found.
    * The data type of every target column is float.
        + In regression tasks, this is achieved by converting numerical data types (float, int, bool, time-like) into
            float, and raising exceptions if other data types are found.
        + In binary classification, this is achieved by representing the negative class by 0.0 and the positive class
            by 1.0. If the original data type is categorical, the negative class corresponds to the first category,
            whereas the positive class corresponds to the second category. If the original data type is not categorical,
            the positive and negative classes are determined through sklearn's `LabelEncoder`.
        + In multiclass classification, this is achieved by representing the `i`-th class by `i`.
        + In multilabel classification, this is achieved by representing the presence of a class by 1.0 and its absence
            by 0.0.
    * Both features and labels may contain NaN values before encoding. These are simply propagated, meaning that
        encoded data may contain NaN values as well!

    `inverse_transform()` decodes encoded data. In the case of classification, it is also able to handle Numpy arrays
    containing class (indices), as returned by `predict()`, as well as class probabilities, as returned by
    `predict_proba()`.
    """

    _YEAR_RES = pd.Timedelta(365.2525, unit='d')
    _DAY_RES = pd.Timedelta(1, unit='d')
    _HOUR_RES = pd.Timedelta(1, unit='h')
    _MIN_RES = pd.Timedelta(1, unit='m')
    _SEC_RES = pd.Timedelta(1, unit='s')

    def __init__(self, classify: bool = True):
        self._classify = classify
        self._task = None
        self._features = None
        self._targets = None

    @property
    def classify(self) -> bool:
        return self._classify

    @property
    def task_(self) -> Optional[str]:
        return self._task

    @property
    def n_features_(self) -> Optional[int]:
        return None if self._features is None else len(self._features)

    @property
    def n_targets_(self) -> Optional[int]:
        return None if self._targets is None else len(self._targets)

    @property
    def feature_names_(self) -> Optional[list]:
        return None if self._features is None else [f['name'] for f in self._features]

    @property
    def target_names_(self) -> Optional[list]:
        return None if self._targets is None else [f['name'] for f in self._targets]

    def get_target_or_class_names(self) -> Optional[list]:
        """
        Convenience method for getting the names of the targets or, in case of multiclass classification, the names of
        the individual classes.
        :return: List of target- or class names.
        """
        if self._task == 'multiclass_classification':
            if self._targets is not None:
                assert len(self._targets) == 1
                return list(self._targets[0]['dtype']['categories'])
        else:
            return self.target_names_

    def get_dtype(self, name: str) -> Optional[dict]:
        if self._features is None:
            raise ValueError('Encoder must be fit before method `get_dtype()` can be called.')
        dt = None
        for f in self._features:
            if name == f['name']:
                dt = f
                break
        if dt is not None:
            return dt.get('dtype')
        elif self._targets is not None:
            for f in self._targets:
                if name == f['name']:
                    dt = f
                    break
            if dt is not None:
                return dt.get('dtype')
        raise ValueError('Unknown column name: ' + name)

    def fit(self, x: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> 'Encoder':
        self._features = [self._fit_series(x[c]) for c in x.columns]

        if y is None:
            self._targets = None
            self._task = None
        else:
            if y.shape[1] == 0:
                self._targets = []
                self._task = None
            else:
                if self._classify:
                    self._task = 'multilabel_classification' if y.shape[1] > 1 else 'multiclass_classification'
                else:
                    self._task = 'regression'
                self._targets = [self._fit_series(y[c], target=True, n_targets=y.shape[1]) for c in y.columns]

        return self

    def _fit_series(self, s: pd.Series, target: bool = False, n_targets: int = 0) -> dict:
        info = dict(name=s.dtype.name, kind=s.dtype.kind)
        if s.dtype.name == 'category':
            if target:
                if self._classify:
                    if n_targets == 1:
                        if len(s.cat.categories) <= 2:
                            self._task = 'binary_classification'
                    elif len(s.cat.categories) > 2:
                        raise ValueError('In multilabel classification, categorical targets can have at most 2'
                                         f' categories, but found {len(s.cat.categories)} categories for "{s.name}".')
                else:
                    raise ValueError('In regression, targets cannot be categorical.')
            info['categories'] = list(s.cat.categories)
            info['ordered'] = s.cat.ordered
        elif s.dtype.kind == 'M':
            # datetime
            if target and self._classify:
                raise ValueError('In classification, targets cannot be datetime-like.')
            start = s.min()
            info['start'] = start
            span = s.max() - start
            if span > Encoder._YEAR_RES:
                info['resolution'] = 'year'
            elif span > Encoder._DAY_RES:
                info['resolution'] = 'day'
            elif span > Encoder._HOUR_RES:
                info['resolution'] = 'hour'
            elif span > Encoder._MIN_RES:
                info['resolution'] = 'minute'
            else:
                info['resolution'] = 'second'
        elif s.dtype.kind == 'm':
            # timedelta
            if target and self._classify:
                raise ValueError('In classification, targets cannot be datetime-like.')
            mean = s.abs().mean()
            if mean > Encoder._YEAR_RES:
                info['resolution'] = 'year'
            elif mean > Encoder._DAY_RES:
                info['resolution'] = 'day'
            elif mean > Encoder._HOUR_RES:
                info['resolution'] = 'hour'
            elif mean > Encoder._MIN_RES:
                info['resolution'] = 'minute'
            else:
                info['resolution'] = 'second'
        elif s.dtype.kind == 'O':
            raise ValueError(f'Cannot encode column "{s.name}" with object data type.')
        elif target:
            if self._classify:
                if s.dtype.kind == 'f' and (s[s.notna()] != s[s.notna()].astype('int64')).any():
                    raise ValueError('In classification, targets must have discrete values.')
                else:
                    le = LabelEncoder()
                    le.fit([v for v in s.unique() if pd.notna(v)])
                    categories = le.classes_.tolist()
                    if n_targets == 1:
                        if len(categories) <= 2:
                            self._task = 'binary_classification'
                    elif len(categories) > 2:
                        raise ValueError('In multilabel classification, categorical targets can have at most 2'
                                         f' categories, but found {len(categories)} categories for "{s.name}".')
                    info['categories'] = categories
            elif s.dtype.kind in 'iub':
                # convert into float
                info['name'] = 'float64'
                info['kind'] = 'f'
            elif s.dtype.kind != 'f':
                raise ValueError('In regression, data types of targets must be numerical,'
                                 f' but found {s.dtype.name}.')
        return dict(name=s.name, dtype=info)

    def transform(self, *, inplace: bool = True, **kwargs: Optional[pd.DataFrame]):
        """
        Transform features- and/or labels DataFrames.
        :param inplace: Whether to modify the given data in place.
        :param kwargs: The data to transform, with keys "x" (features), "y" (labels) or "data" (features+labels).
        :return: The transformed DataFrame(s), either a single DataFrame if only one of "x" or "y" is passed, or a
        pair of DataFrames in the same order as in the argument dict. If "data" is passed, returns the pair of encoded
        features and labels.
        """
        out = []
        for k, v in kwargs.items():
            if k in ('x', 'y'):
                if v is not None:
                    info = self._features if k == 'x' else self._targets
                    if info is None:
                        raise ValueError('Encoder must be fit before it can be applied to data.')
                    columns = [f['name'] for f in info]
                    missing = [c for c in columns if c not in v.columns]
                    if missing:
                        raise ValueError('The following columns are missing: ' + cu.repr_list(missing, brackets=False))
                    if v.shape[1] != len(columns) or any(c1 != c2 for c1, c2 in zip(columns, v.columns)):
                        v = v.reindex(columns, axis=1)
                        copy = False
                    else:
                        copy = not inplace
                    for f in info:
                        c = f['name']
                        s = self._transform_series(v[c], f['dtype'], target=(k == 'y'))
                        if s is not None:
                            if copy:
                                v = v.copy()
                                copy = False
                            v[c] = s
                out.append(v)
            elif k == 'data':
                if v is None:
                    out += [None, None]
                elif self._targets is None:
                    # `data` only contains features
                    out += [self.transform(inplace=inplace, x=v), None]
                else:
                    target_columns = [f['name'] for f in self._targets]
                    missing = [c for c in target_columns if c not in v.columns]
                    if missing:
                        raise ValueError('The following columns are missing: ' + cu.repr_list(missing, brackets=False))
                    y = v[target_columns].copy()
                    out += [self.transform(inplace=inplace, x=v), self.transform(inplace=True, y=y)]
            else:
                raise TypeError(f"transform() got an unexpected keyword argument '{k}'")

        return out[0] if len(out) == 1 else tuple(out)

    def _transform_series(self, s: pd.Series, dt: dict, target: bool = False) -> Optional[pd.Series]:
        if target and self._classify:
            out = pd.Series(name=s.name, index=s.index, data=None, dtype='float32')
            for i, c in enumerate(dt['categories']):
                out[s == c] = i
            return out

        name = dt['name']
        kind = dt['kind']
        if name == 'category':
            if s.dtype.name == 'category':
                categories = dt['categories']
                if len(categories) != len(s.cat.categories) \
                        or any(c1 != c2 for c1, c2 in zip(categories, s.cat.categories)):
                    raise ValueError(f'Categories of "{s.name}" do not match.')
            else:
                raise ValueError(f'Data type of "{s.name}" should be categorical, but found {s.dtype.name}.')
        elif kind == 'M':
            if s.dtype.kind == 'M':
                return (s - pd.Timestamp(dt['start'])) / Encoder._get_resolution(dt['resolution'])
            else:
                raise ValueError(f'Data type of "{s.name}" should be datetime, but found {s.dtype.name}.')
        elif kind == 'm':
            if s.dtype.kind == 'm':
                return s / Encoder._get_resolution(dt['resolution'])
            else:
                raise ValueError(f'Data type of "{s.name}" should be timedelta, but found {s.dtype.name}.')
        elif s.dtype.name == 'category' or s.dtype.kind in 'Mm':
            raise ValueError(f'Data type of "{s.name}" should be {name}, but found {s.dtype.name}.')
        elif s.dtype.name != name:
            return s.astype(name)
        return None

    @classmethod
    def _get_resolution(cls, res: str) -> pd.Timedelta:
        if res == 'year':
            return cls._YEAR_RES
        elif res == 'day':
            return cls._DAY_RES
        elif res == 'hour':
            return cls._HOUR_RES
        elif res == 'minute':
            return cls._MIN_RES
        elif res == 'second':
            return cls._SEC_RES
        raise ValueError(f'Invalid resolution: "{res}".')

    def fit_transform(self, x: pd.DataFrame, y: Optional[pd.DataFrame] = None, inplace: bool = True) \
            -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        return self.fit(x, y=y).transform(x=x, y=y, inplace=inplace)

    def inverse_transform(self, *, inplace: bool = True, **kwargs: Union[pd.DataFrame, np.ndarray, None]):
        """
        Back-transform features- and/or labels DataFrames.
        :param inplace: Whether to modify the given data in place.
        :param kwargs: The data to transform back, with keys "x" (features) or "y" (labels).
        :return: The back-transformed DataFrame(s), either a single DataFrame if only one of "x" or "y" is passed, or a
        pair of DataFrames in the same order as in the argument dict.
        """
        out = []

        for k, v in kwargs.items():
            if k == 'x':
                if v is not None:
                    if self._features is None:
                        raise ValueError('Encoder must be fit before it can be applied to data.')
                    copy = not inplace
                    for f in self._features:
                        c = f['name']
                        if c in v.columns:
                            s = self._inverse_transform_series(v[c], f['dtype'])
                            if s is not None:
                                if copy:
                                    v = v.copy()
                                    copy = False
                                v[c] = s
                out.append(v)
            elif k == 'y':
                if v is not None:
                    if self._targets is None:
                        raise ValueError('Encoder must be fit before it can be applied to data.')
                    if isinstance(v, np.ndarray):
                        if v.ndim == 1:
                            v = v.reshape((-1, 1))
                        elif v.ndim != 2:
                            raise ValueError(f'Target array must have rank 1 or 2, but found {v.ndim}.')
                        if self._classify and self._task != 'multilabel_classification':
                            # 1 column, or as many columns as there are classes
                            classes = self._targets[0]['dtype']['categories']
                            if v.shape[1] == 1:
                                if v.dtype.kind == 'f' and not ((v < 0.).any() or (1. < v).any()) \
                                        and ((0. < v) & (v < 1.)).any() and self._task == 'binary_classification':
                                    # probability of positive class
                                    v = pd.DataFrame(data=v, columns=classes[1:2])
                                else:
                                    v = pd.DataFrame(data=v, columns=[self._targets[0]['name']])
                            elif v.shape[1] == len(classes):
                                v = pd.DataFrame(data=v, columns=classes)
                            else:
                                raise ValueError(f'Target array must have 1 or {len(classes)} columns,'
                                                 f' but found {v.shape[1]}.')
                        elif v.shape[1] == len(self._targets):
                            v = pd.DataFrame(data=v, columns=[f['name'] for f in self._targets])
                        else:
                            raise ValueError(
                                f'Target array must have {len(self._targets)} columns, but found {v.shape[1]}.')
                        copy = False
                    else:
                        copy = not inplace

                    if self._classify and self._task != 'multilabel_classification':
                        name = self._targets[0]['name']
                        if name in v.columns:
                            s = self._inverse_transform_series(v[name], self._targets[0]['dtype'], target=True)
                            if s is not None:
                                if copy:
                                    v = v.copy()
                                v[name] = s
                    else:
                        for f in self._targets:
                            c = f['name']
                            if c in v.columns:
                                s = self._inverse_transform_series(v[c], f['dtype'], target=True)
                                if s is not None:
                                    if copy:
                                        v = v.copy()
                                        copy = False
                                    v[c] = s
                out.append(v)
            else:
                raise TypeError(f"inverse_transform() got an unexpected keyword argument '{k}'")

        return out[0] if len(out) == 1 else tuple(out)

    @classmethod
    def _inverse_transform_series(cls, s: pd.Series, dt: dict, target: bool = False) -> Optional[pd.Series]:
        name = dt['name']
        kind = dt['kind']
        categories = dt.get('categories')
        if categories is not None:
            if s.dtype.kind == 'f':
                if not ((s < 0.).any() or (1. < s).any()) and ((0. < s) & (s < 1.)).any():
                    if len(categories) == 2:
                        # probability of "positive" class => leave unchanged
                        return None
                    else:
                        raise ValueError(f'Cannot decode "{s.name}": single probabilities for more than 2 categories.')
                elif target:
                    # `s` contains indices
                    s = s.fillna(-1).astype(np.int32)
            if target and s.dtype.kind in 'iu' and (-1 <= s).all() and (s < len(categories)).all():
                s = pd.Series(pd.Categorical.from_codes(s, categories=categories, ordered=dt.get('ordered', False)),
                              index=s.index, name=s.name)
                if name == 'category' or (kind not in 'fOmM' and s.isna().any()):
                    return s
                else:
                    return s.astype(name)
            elif s.dtype.name == 'category':
                if len(categories) != len(s.cat.categories) \
                        or any(c1 != c2 for c1, c2 in zip(categories, s.cat.categories)):
                    raise ValueError(f'Categories of "{s.name}" do not match.')
                elif name != 'category':
                    return s.astype(name)
            else:
                if (s.isna() | s.isin(categories)).all():
                    if name == 'category':
                        return s.astype(pd.CategoricalDtype(categories=categories, ordered=dt.get('ordered', False)))
                    elif s.dtype.name != name:
                        return s.astype(name)
                else:
                    raise ValueError(f'Categories of "{s.name}" do not match.')
        elif kind == 'M':
            if s.dtype.kind == 'f':
                return s * Encoder._get_resolution(dt['resolution']) + pd.Timestamp(dt['start'])
            elif s.dtype.kind != 'M':
                raise ValueError(f'Data type of "{s.name}" should be datetime or float, but found {s.dtype.name}.')
        elif kind == 'm':
            if s.dtype.kind == 'f':
                return s * Encoder._get_resolution(dt['resolution'])
            elif s.dtype.kind != 'm':
                raise ValueError(f'Data type of "{s.name}" should be timedelta or float, but found {s.dtype.name}.')
        elif s.dtype.name == 'category' or s.dtype.kind in 'Mm':
            raise ValueError(f'Data type of "{s.name}" should be {name}, but found {s.dtype.name}.')
        elif s.dtype.name != name:
            return s.astype(name)
        return None

    def dump(self, file: Union[str, Path]):
        dct = dict(
            classify=self._classify,
            task=self._task,
            features=self._features,
            targets=self._targets
        )
        io.dump(io.to_json(dct), file)

    @classmethod
    def load(cls, file: Union[str, Path]) -> 'Encoder':
        dct = io.load(file)
        encoder = Encoder()
        value = dct.get('classify')
        if value is not None:
            encoder._classify = value
        value = dct.get('task')
        if value is not None:
            encoder._task = value
        value = dct.get('features')
        if value is not None:
            encoder._features = value
        value = dct.get('targets')
        if value is not None:
            encoder._targets = value
        return encoder
