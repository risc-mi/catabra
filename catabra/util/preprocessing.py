#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

from typing import Union, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn import preprocessing as skl_preprocessing
from sklearn.utils.validation import check_is_fitted
from sklearn.base import TransformerMixin, BaseEstimator, clone


class MinMaxScaler(skl_preprocessing.MinMaxScaler):

    def __init__(self, fit_bool: bool = True, **kwargs):
        """
        Transform data by scaling each feature to a given range. The only difference to
        `sklearn.preprocessing.MinMaxScaler` is parameter `fit_bool` that, when set to False, does not fit this scaler
        on boolean features but rather uses 0 and 1 as fixed minimum and maximum values. This ensures that False is
        always mapped to `feature_range[0]` and True is always mapped to `feature_range[1]`. Otherwise, if the training
        data only contains True values, True would be mapped to `feature_range[0]` and False to
        `feature_range[0] - feature_range[1]`.
        The behavior on other numerical data types is not affected by this.

        Note that `sklearn.preprocessing.MaxAbsScaler` always maps False to 0 and True to 1, so there is no need for
        an analogous subclass.

        :param fit_bool: Whether to fit this scaler on boolean features. If True, the behavior is identical to
        `sklearn.preprocessing.MinMaxScaler`.
        """
        super(MinMaxScaler, self).__init__(**kwargs)
        self.fit_bool = fit_bool

    def partial_fit(self, X: Union[np.ndarray, pd.DataFrame], y=None) -> 'MinMaxScaler':
        if isinstance(X, np.ndarray):
            if self.fit_bool or X.dtype.kind != 'b':
                super(MinMaxScaler, self).partial_fit(X, y=y)
            else:
                if hasattr(self, 'n_samples_seen_'):
                    self.data_min_ = np.minimum(self.data_min_, 0.)
                    self.data_max_ = np.maximum(self.data_max_, 1.)
                    self.n_samples_seen_ += X.shape[0]
                else:
                    self.data_min_ = np.zeros(X.shape[1], dtype=np.float64)
                    self.data_max_ = np.ones(X.shape[1], dtype=np.float64)
                    self.n_samples_seen_ = X.shape[0]

                self.data_range_ = self.data_max_ - self.data_min_
                self.scale_ = ((self.feature_range[1] - self.feature_range[0]) / self.data_range_)
                self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        else:
            updates = []
            if not self.fit_bool:
                for i, c in enumerate(X.columns):
                    if X[c].dtype.kind == 'b':
                        data_min = 0
                        data_max = 1
                        if hasattr(self, 'data_min_'):
                            data_min = np.minimum(self.data_min_[i], data_min)
                            data_max = np.maximum(self.data_max_[i], data_max)
                        updates.append((i, data_min, data_max))

            super(MinMaxScaler, self).partial_fit(X, y=y)

            for i, data_min, data_max in updates:
                self.data_min_[i] = data_min
                self.data_max_[i] = data_max
                self.data_range_[i] = data_max - data_min
                self.scale_[i] = ((self.feature_range[1] - self.feature_range[0]) / self.data_range_[i])
                self.min_[i] = self.feature_range[0] - data_min * self.scale_[i]

        return self


class OneHotEncoder(skl_preprocessing.OneHotEncoder):

    def __init__(self, drop_na: bool = False, drop=None, handle_unknown=None, **kwargs):
        """
        Encode categorical features as a one-hot numeric array. The only difference to
        `sklearn.preprocessing.OneHotEncode` is parameter `drop_na` that, when set to True, allows to drop NaN
        categories. More precisely, no separate columns representing NaN categories are added upon transformation,
        resembling the behavior of `pandas.get_dummies()`.
        :param drop_na: Whether to drop NaN categories. If False, the behavior is identical to
        `sklearn.preprocessing.OneHotEncode`.
        :param drop: Categories to drop. If `drop_na` is True, this parameter must be set to None.
        :param handle_unknown: How to handle unknown categories. If `drop_na` is True, this parameter must be set to
        "ignore". None defaults to "ignore" if `drop_na` is True and to "error" otherwise.
        """
        self._drop_na = drop_na
        if self._drop_na:
            assert drop is None
            if handle_unknown is None:
                handle_unknown = 'ignore'
            else:
                assert handle_unknown == 'ignore'
        elif handle_unknown is None:
            handle_unknown = 'error'

        super(OneHotEncoder, self).__init__(drop=drop, handle_unknown=handle_unknown, **kwargs)

    @property
    def drop_na(self) -> bool:
        return self._drop_na

    def fit(self, X, y=None) -> 'OneHotEncoder':
        super(OneHotEncoder, self).fit(X, y=y)
        if self._drop_na:
            assert self.drop_idx_ is None
            drop_idx = []
            for categories in self.categories_:
                mask = pd.isna(categories)
                if mask.any():
                    drop_idx.append(np.flatnonzero(mask)[-1])
                else:
                    drop_idx.append(None)
            if not all(i is None for i in drop_idx):
                self.drop_idx_ = np.asarray(drop_idx, dtype=np.object)

        return self


class NumCatTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, num_transformer=None, cat_transformer=None, bool: str = 'passthrough', obj: str = 'drop',
                 timedelta: str = 'num', timestamp: str = 'num'):
        """
        Transform numerical and categorical columns of a DataFrame separately.

        The order of columns may change compared to the input: numerical columns come first, followed by categorical
        columns, followed by passed-through columns.

        :param num_transformer: The transformer to apply to numerical columns, or "passthrough" or "drop". Must
        implement `fit()` and `transform()`. Class instances are cloned before being fit to data, to ensure that the
        given instances are left unchanged.
        :param cat_transformer: The transformer to apply to categorical columns, or "passthrough" or "drop". Must
        implement `fit()` and `transform()`. Class instances are cloned before being fit to data, to ensure that the
        given instances are left unchanged.
        :param bool: How to treat boolean columns. One of "num", "cat", "passthrough" or "drop".
        :param obj: How to treat columns with object data type. One of "num", "cat", "passthrough" or "drop".
        :param timedelta: How to treat timedelta columns. One of "num", "cat", "passthrough", "drop", "[ns]", "[us]",
        "[ms]", "[s]", "[m]", "[h]", "[d]", "[w]" or "[y]". A string representing a temporal resolution means that
        timedelta columns are first converted into floats by dividing by the given resolution, and then treating the
        result as numeric. This is useful if `num_transformer` does not natively support timedelta values.
        :param timestamp: How to treat timestamp/datetime columns. Same possibilities as for `timedelta`.
        """

        self.num_transformer = num_transformer or 'passthrough'
        self.cat_transformer = cat_transformer or 'passthrough'
        self.bool = bool
        self.obj = obj
        self._timedelta = timedelta
        self._timestamp = timestamp
        self._timedelta_resolution = NumCatTransformer._get_resolution(self._timedelta)
        self._timestamp_resolution = NumCatTransformer._get_resolution(self._timestamp)

    @property
    def timedelta(self) -> str:
        return self._timedelta

    @property
    def timestamp(self) -> str:
        return self._timestamp

    @property
    def timedelta_resolution(self) -> Optional[pd.Timedelta]:
        return self._timedelta_resolution

    @property
    def timestamp_resolution(self) -> Optional[pd.Timedelta]:
        return self._timestamp_resolution

    def fit(self, X: pd.DataFrame, y=None) -> 'NumCatTransformer':
        self.num_cols_: list = []
        self.cat_cols_: list = []
        self.passthrough_cols_: list = []

        def _add_to_list(_c, _spec: str, _allow_temporal: bool):
            if _spec == 'num':
                self.num_cols_.append(_c)
            elif _spec == 'cat':
                self.cat_cols_.append(_c)
            elif _spec == 'passthrough':
                self.passthrough_cols_.append(_c)
            elif _allow_temporal and _spec in ('[ns]', '[us]', '[ms]', '[s]', '[m]', '[h]', '[d]', '[w]', '[y]'):
                self.num_cols_.append(_c)

        for c in X.columns:
            if X[c].dtype.name == 'category':
                self.cat_cols_.append(c)
            elif X[c].dtype.kind == 'b':
                _add_to_list(c, self.bool, False)
            elif X[c].dtype.kind == 'O':
                _add_to_list(c, self.obj, False)
            elif X[c].dtype.kind == 'M':
                _add_to_list(c, self._timestamp, True)
            elif X[c].dtype.kind == 'm':
                _add_to_list(c, self._timedelta, True)
            elif X[c].dtype.kind in 'uif':
                self.num_cols_.append(c)

        if self.cat_cols_ and not isinstance(self.cat_transformer, str):
            self.cat_transformer_ = self.cat_transformer() if type(self.cat_transformer) is type else clone(self.cat_transformer)
            self.cat_transformer_.fit(X[self.cat_cols_])
        else:
            self.cat_transformer_ = None
            if self.cat_cols_ and self.cat_transformer == 'passthrough':
                self.passthrough_cols_ = self.cat_cols_ + self.passthrough_cols_
                self.cat_cols_ = []

        if self.num_cols_ and not isinstance(self.num_transformer, str):
            self.num_transformer_ = self.num_transformer() if type(self.num_transformer) is type else clone(self.num_transformer)
            X_num, _ = self._prepare_num(X[self.num_cols_])
            self.num_transformer_.fit(X_num)
        else:
            self.num_transformer_ = None
            if self.num_cols_ and self.num_transformer == 'passthrough':
                self.passthrough_cols_ = self.num_cols_ + self.passthrough_cols_
                self.num_cols_ = []

        return self

    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        check_is_fitted(self)
        self._validate_input(X)

        if self.num_transformer_ is not None:
            X_num, _ = self._prepare_num(X[self.num_cols_])
            num = self.num_transformer_.transform(X_num)
            num, num_df = self._postproc_num(num, X.index)
        else:
            num = num_df = None

        if self.cat_transformer_ is not None:
            cat = self.cat_transformer_.transform(X[self.cat_cols_])
            cat, cat_df = self._postproc_cat(cat, X.index)
        else:
            cat = cat_df = None

        return NumCatTransformer._combine_results(num, num_df, cat, cat_df, X[self.passthrough_cols_])

    def _validate_input(self, df: pd.DataFrame):
        diff = [c for c in self.num_cols_ + self.cat_cols_ + self.passthrough_cols_ if c not in df.columns]
        if diff:
            raise ValueError('X lacks columns ' + str(diff))

    def _prepare_num(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        copied = False
        if self._timedelta_resolution is not None:
            for c in df.columns:
                if df[c].dtype.kind == 'm':
                    if not copied:
                        df = df.copy()
                        copied = True
                    df[c] = df[c] / self._timedelta_resolution
        if self._timestamp_resolution is not None:
            for c in df.columns:
                if df[c].dtype.kind == 'M':
                    if not copied:
                        df = df.copy()
                        copied = True
                    df[c] = (df[c] - pd.Timestamp(0)) / self._timestamp_resolution

        return df, copied

    def _postproc_num(self, num, index: pd.Index):
        num_df = None
        if hasattr(num, 'toarray'):
            num = num.toarray()
        if isinstance(num, pd.DataFrame):
            assert (num.index == index).all()
            num_df = num
        elif num.shape[1] == len(self.num_cols_):
            num_df = pd.DataFrame(index=index, columns=self.num_cols_, data=num)
        elif isinstance(self.num_transformer_, skl_preprocessing.KBinsDiscretizer):
            if self.num_transformer_.encode in ('onehot', 'onehot-dense') \
                    and self.num_transformer_.n_bins_.sum() == num.shape[1] \
                    and len(self.num_transformer_.n_bins_) == len(self.num_cols_):
                num_df = pd.DataFrame(
                    index=index,
                    columns=[f'{col}_{b}' for col, n in zip(self.num_cols_, self.num_transformer_.n_bins_)
                             for b in range(n)],
                    data=num
                )
        return num, num_df

    def _postproc_cat(self, cat, index: pd.Index):
        cat_df = None
        if hasattr(cat, 'toarray'):
            cat = cat.toarray()
        if isinstance(cat, pd.DataFrame):
            assert (cat.index == index).all()
            cat_df = cat
        elif isinstance(self.cat_transformer_, skl_preprocessing.OneHotEncoder):
            if self.cat_transformer_.drop_idx_ is not None:
                features_out = []
                for i, categories in enumerate(self.cat_transformer_.categories_):
                    j = self.cat_transformer_.drop_idx_[i]
                    if j is None:
                        features_out.append(categories)
                    elif j == 0:
                        features_out.append(categories[1:])
                    elif j + 1 == len(categories):
                        features_out.append(categories[:-1])
                    else:
                        features_out.append(np.concatenate([categories[:j], categories[j + 1:]]))
            else:
                features_out = self.cat_transformer_.categories_
            if len(self.cat_cols_) == len(features_out) and cat.shape[1] == sum(len(f) for f in features_out):
                cat_df = pd.DataFrame(
                    index=index,
                    columns=[f'{col}_{f}' for col, feats in zip(self.cat_cols_, features_out) for f in feats],
                    data=cat.toarray() if hasattr(cat, 'toarray') else cat
                )
        elif cat.shape[1] == len(self.cat_cols_):
            cat_df = pd.DataFrame(index=index, columns=self.cat_cols_, data=cat)
        return cat, cat_df

    @staticmethod
    def _combine_results(num, num_df, cat, cat_df, passthrough_df) -> Union[pd.DataFrame, np.ndarray]:
        if (num is not None and num_df is None) or (cat is not None and cat_df is None):
            # return array
            arrs = []
            if num is not None:
                arrs.append(num)
            elif num_df is not None:
                arrs.append(num_df.values)
            if cat is not None:
                arrs.append(cat)
            elif cat_df is not None:
                arrs.append(cat_df.values)
            if not passthrough_df.empty:
                arrs.append(passthrough_df.values)
            return np.hstack(arrs)
        else:
            # return DataFrame
            dfs = []
            if num_df is not None:
                dfs.append(num_df)
            if cat_df is not None:
                dfs.append(cat_df)
            if not passthrough_df.empty:
                dfs.append(passthrough_df)
            return pd.concat(dfs, axis=1, sort=False)

    @staticmethod
    def _get_resolution(spec: str) -> Optional[pd.Timedelta]:
        if spec == '[ns]':
            return pd.Timedelta(1, unit='ns')
        elif spec == '[us]':
            return pd.Timedelta(1, unit='us')
        elif spec == '[ms]':
            return pd.Timedelta(1, unit='ms')
        elif spec == '[s]':
            return pd.Timedelta(1, unit='s')
        elif spec == '[m]':
            return pd.Timedelta(1, unit='m')
        elif spec == '[h]':
            return pd.Timedelta(1, unit='h')
        elif spec == '[d]':
            return pd.Timedelta(1, unit='d')
        elif spec == '[w]':
            return pd.Timedelta(7, unit='d')
        elif spec == '[y]':
            return pd.Timedelta(365.2525, unit='d')
        return None


class FeatureFilter(BaseEstimator, TransformerMixin):
    """
    Simple transformer that ensures that list of features is identical to features seen during fit.
    Only applicable to DataFrames.
    """

    def fit(self, X: pd.DataFrame, y=None) -> 'FeatureFilter':
        self.columns_ = X.columns
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        if X.shape[1] == len(self.columns_) and (X.columns == self.columns_).all():
            return X
        else:
            return X.reindex(self.columns_, axis=1)


# convenience functions

def ordinal_encode(X: pd.DataFrame, dtype=np.float64, obj: str = 'passthrough') -> Union[pd.DataFrame, np.ndarray]:
    return NumCatTransformer(
        num_transformer='passthrough',
        cat_transformer=skl_preprocessing.OrdinalEncoder(dtype=dtype),
        obj=obj
    ).fit_transform(X)


def one_hot_encode(X: pd.DataFrame, drop=None, dtype=np.float64, handle_unknown: str = 'error',
                   obj: str = 'passthrough') -> Union[pd.DataFrame, np.ndarray]:
    return NumCatTransformer(
        num_transformer='passthrough',
        cat_transformer=skl_preprocessing.OneHotEncoder(drop=drop, sparse=False, dtype=dtype,
                                                        handle_unknown=handle_unknown),
        obj=obj
    ).fit_transform(X)


def k_bins_discretize(X: pd.DataFrame, n_bins: int = 5, encode: str = 'onehot', strategy: str = 'quantile',
                      timedelta: str = 'num', timestamp: str = 'passthrough') -> Union[pd.DataFrame, np.ndarray]:
    return NumCatTransformer(
        num_transformer=skl_preprocessing.KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy),
        cat_transformer='passthrough',
        obj='passthrough',
        timedelta=timedelta,
        timestamp=timestamp,
    ).fit_transform(X)


def binarize(X: pd.DataFrame, threshold: float = 0, timedelta: str = 'num',
             timestamp: str = 'passthrough') -> Union[pd.DataFrame, np.ndarray]:
    return NumCatTransformer(
        num_transformer=skl_preprocessing.Binarizer(threshold=threshold),
        cat_transformer='passthrough',
        obj='passthrough',
        timedelta=timedelta,
        timestamp=timestamp,
    ).fit_transform(X)


def scale(X: pd.DataFrame, strategy: str = 'standard', bool: str = 'passthrough', timedelta: str = 'num',
          timestamp: str = 'passthrough', **kwargs) -> Union[pd.DataFrame, np.ndarray]:
    if strategy == 'standard':
        num_transformer = skl_preprocessing.StandardScaler(**kwargs)
    elif strategy == 'robust':
        num_transformer = skl_preprocessing.RobustScaler(**kwargs)
    elif strategy == 'minmax':
        num_transformer = skl_preprocessing.MinMaxScaler(**kwargs)
    elif strategy == 'maxabs':
        num_transformer = skl_preprocessing.MaxAbsScaler(**kwargs)
    else:
        raise ValueError('Scaling strategy must be one of "standard", "robust", "minmax" or "maxabs",'
                         f' but got "{strategy}".')
    return NumCatTransformer(
        num_transformer=num_transformer,
        cat_transformer='passthrough',
        bool=bool,
        obj='passthrough',
        timedelta=timedelta,
        timestamp=timestamp,
    ).fit_transform(X)
