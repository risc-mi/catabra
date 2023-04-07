#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from catabra.util import io


def convert_object_dtypes(df: pd.DataFrame, inplace: bool = True, max_categories: int = 100) -> pd.DataFrame:
    """
    Convert "object" data types in `df` into other data types, if possible. In particular, this includes timedelta,
    datetime, categorical and string types, in that order. String types are not supported in all Pandas versions.

    Parameters
    ----------
    df: DataFrame
        The DataFrame.
    inplace: bool, default=True
        Whether to modify `df` in place. Note that if no column in `df` can be converted, it is returned as-is even if
        `inplace` is False.
    max_categories: int, default=100
        The maximum number of allowed categories when converting on object column into a categorical column.

    Returns
    -------
    DataFrame
        DataFrame with converted data types.
    """
    for c in df.columns:
        s0 = df[c]
        if s0.dtype.kind == 'O' and s0.dtype.name not in ('category', 'string'):
            s = None
            na_mask = s0.isna()
            if na_mask.all():
                s = np.nan
            else:
                try:
                    s = pd.to_timedelta(s0)
                except ValueError:
                    try:
                        s = pd.to_datetime(s0)
                    except ValueError:
                        n = s0.nunique()
                        if n <= max_categories and n * 10 <= len(s0) - na_mask.sum():
                            s = pd.Categorical(s0)
                        elif (s0.str.len().notna() | na_mask).all():
                            # all entries are either strings or NA
                            try:
                                s = s0.astype('string')
                            except:     # noqa
                                pass
            if s is not None:
                if not inplace:
                    df = df.copy()
                    inplace = True
                df[c] = s
    return df


def set_index(df: pd.DataFrame, inplace: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Set the row index of the given DataFrame to an ID column, unless it contains IDs already, and return a list of
    other potential ID columns.

    Parameters
    ----------
    df:
        The DataFrame.
    inplace: bool, default=True
        Whether to modify `df` in place.

    Returns
    -------
    tuple
        Pair `(df, id_cols)`, where `df` is the new DataFrame and `id_cols` is a list of potential ID columns.
    """
    id_cols = []
    for c in df.columns:
        s0 = df[c]
        if s0.dtype.kind == 'i':
            n = s0.nunique()
        elif s0.dtype.kind == 'f' and ((np.floor(s0) == s0) | s0.isna()).all():
            n = s0.nunique()
        else:
            n = 0
        if n * 20 >= max(1, len(s0)):
            id_cols.append(c)
    if len(id_cols) and df.columns[0] == id_cols[0] and df.index.nlevels == 1 and df.index.name is None \
            and (df.index == pd.RangeIndex(len(df))).all() and df[id_cols[0]].is_unique:
        if inplace:
            df.set_index(id_cols[0], append=False, drop=True, inplace=True)
        else:
            df = df.set_index(id_cols[0], append=False, drop=True, inplace=False)
        del id_cols[0]
    return df, id_cols


def merge_tables(tables: Iterable[Union[pd.DataFrame, str, io.Path]]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Merge the given tables by left-joining them on ID columns.

    Parameters
    ----------
    tables: Iterable
        The tables to merge, an iterable of DataFrames or paths to tables. Function `convert_object_dtypes()` is
        automatically applied to tables read from files.

    Returns
    -------
    tuple
        The pair `(df, id_cols)`, where `df` is the merged DataFrame and `id_cols` is the list of potential ID columns.
    """
    df = None
    id_cols = None
    for i, fn in enumerate(tables):
        if isinstance(fn, pd.DataFrame):
            df_aux = fn
            inplace = False
        else:
            if not isinstance(fn, io.Path):
                fn = io.Path(fn)
            df_aux = io.read_df(fn)
            df_aux = convert_object_dtypes(df_aux, inplace=True)
            inplace = True
        df_aux, id_cols_aux = set_index(df_aux, inplace=inplace)
        if df is None:
            df = df_aux
            id_cols = id_cols_aux
        else:
            ok = False
            if df.index.name is not None and df.index.name == df_aux.index.name:
                if df_aux.index.is_unique:
                    df = df.join(df_aux, how='left')
                    id_cols += id_cols_aux
                    ok = True
            elif df.index.name in id_cols_aux:
                if df_aux[df.index.name].is_unique:
                    df = df.join(df_aux.set_index(df.index.name), how='left')
                    id_cols += [c for c in id_cols_aux if c != df.index.name]
                    ok = True
            elif df_aux.index.name in id_cols:
                if df_aux.index.is_unique:
                    df = df.join(df_aux, on=df_aux.index.name, how='left')
                    id_cols += id_cols_aux
                    ok = True
            if not ok:
                raise RuntimeError(
                    f'Table "{fn.name if isinstance(fn, io.Path) else i}" cannot be joined with other tables.'
                )
    return df, id_cols


def train_test_split(df: pd.DataFrame, by: str) -> Tuple[Dict[str, np.ndarray], Optional[str]]:
    """
    Split the given DataFrame into train- and test set(s), by a given column.

    Parameters
    ----------
    df: DataFrame
        The DataFrame.
    by: str
        The name of the column to split by. Must have bool or categorical data type.

    Returns
    -------
    tuple
        Pair `(split_masks, train_key)`, where

        * `split_masks` is a dict mapping string-keys to masks corresponding to non-overlapping portions of `df`.
        * `train_key` is the key (in `split_masks`) containing the training set, or None if the training set could not
           be determined.
    """
    if by not in df.columns:
        raise ValueError(f'"{by}" is no column of the given DataFrame.')
    s = df[by]
    if s.dtype.kind == 'b':
        train = 'train' in by.lower()
        test = 'test' in by.lower() or 'val' in by.lower()
        if train == test:
            if s.all():
                return {by: s.values}, None
            elif not s.any():
                return {'not_' + by: ~s.values}, None
            else:
                return {'not_' + by: ~s.values, by: s.values}, None
        elif train:
            if s.all():
                return {by: s.values}, by
            else:
                return {by: s.values, 'not_' + by: ~s.values}, by
        else:
            if s.any():
                return {'not_' + by: ~s.values, by: s.values}, 'not_' + by
            else:
                return {'not_' + by: ~s.values}, 'not_' + by
    elif s.dtype.name == 'category':
        train = [c for c in s.cat.categories if 'train' in c.lower()]
        if len(train) == 0:
            train = [c for c in s.cat.categories if not ('test' in c.lower() or 'val' in c.lower())]
        if len(train) == 1:
            train = train[0]
            mask = s == train
            out = {str(train): mask.values}
            if not mask.all():
                for c in s.cat.categories:
                    if c != train:
                        c_mask = s == c
                        if c_mask.any():
                            out[c] = c_mask.values
            return out, str(train)
        else:
            test = {}
            for c in s.cat.categories:
                c_mask = s == c
                if c_mask.any():
                    test[c] = c_mask.values
            return test, None
    else:
        raise ValueError(
            f'The data type of train-test-split column "{by}" must be bool or categorical, but found {s.dtype.name}.'
        )
