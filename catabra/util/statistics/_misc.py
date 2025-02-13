#  Copyright (c) 2022-2025. RISC Software GmbH.
#  All rights reserved.

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from catabra_lib.statistics import chi_square, mann_whitney_u

from catabra.util import logging
from catabra.util.io import Path, convert_rows_to_str, make_path, write_df, write_dfs


def calc_numeric_statistics(df: pd.DataFrame, target: list, classify: bool) -> dict:
    """
    Calculate descriptive statistics for numeric features for a specific dataframe

    Parameters
    ----------
    df: DataFrame
        The main dataframe.
    target: list
        The target labels; stored in list
    classify: bool
        Is true, if classification task. False for regression task

    Returns
    -------
    dict
        Dictionary with statistics (for numeric features) for entire dataset, each target and (in case of
        classification) each label.
    """
    df = df.copy()
    dict_stat = {'overall': df.describe().T}
    if classify:
        for label_ in target:
            df[label_] = df[label_].astype(str).fillna('NaN')
            df_stat_cat = pd.DataFrame()
            df_stat_temp = df.groupby(label_).describe()

            unique = df[label_].unique()
            tests = np.empty((len(dict_stat['overall']), len(unique)), dtype=np.float64)
            for j, v in enumerate(unique):
                if j == 1 and len(unique) == 2:
                    # Mann-Whitney U test is symmetric => no need to calculate it twice
                    tests[:, 1] = tests[:, 0]
                else:
                    mask = df[label_] == v
                    for i, col_ in enumerate(dict_stat['overall'].index):
                        tests[i, j] = mann_whitney_u(df.loc[mask, col_], df.loc[~mask, col_])
            tests = pd.Series(index=pd.MultiIndex.from_product([dict_stat['overall'].index, unique]),
                              data=tests.reshape(-1))

            for col_ in list(dict_stat['overall'].index):
                temp = df_stat_temp.iloc[:, df_stat_temp.columns.get_level_values(0) == col_]
                temp.columns = temp.columns.droplevel()
                arrays = [[col_ for _ in list(temp.index)], list(temp.index)]
                index = pd.MultiIndex.from_tuples(list(zip(*arrays)), names=["Feature", label_])
                df_stat_cat = pd.concat([df_stat_cat, temp.set_index(index)])

            df_stat_cat['count'] = df_stat_cat['count'].astype(int)
            df_stat_cat['mann_whitney_u'] = tests
            dict_stat[label_] = df_stat_cat

    return dict_stat


def create_non_numeric_statistics(df: pd.DataFrame, target: list, name_: str = '') -> pd.DataFrame:
    """
    Calculate descriptive statistics for non-numeric features for a specific dataframe

    Parameters
    ----------
    df: DataFrame
        The main dataframe.
    target: list
        The target labels; stored in list
    name_: str, default=''
        Name of the label. Used for naming the columns

    Returns
    -------
    DataFrame
        Returns a dataframe with statistics (for non-numeric features)
    """
    df_stat_cat = pd.DataFrame()
    for col_ in target + [c for c in df.columns if c not in target]:
        if df[col_].dtype.name in ['bool', 'category'] or (col_ in target):
            series = df[col_].astype(str).fillna('NaN')
            temp = pd.DataFrame(series.value_counts()).rename(columns={col_: name_ + 'count'})
            temp = temp.join(
                pd.DataFrame(series.value_counts(normalize=True) * 100).rename(columns={col_: name_ + '%'}))

            arrays = [[col_ for _ in list(temp.index)], list(temp.index)]
            index = pd.MultiIndex.from_tuples(list(zip(*arrays)), names=["Feature", "Value"])
            temp = temp.set_index(index)
            df_stat_cat = pd.concat([df_stat_cat, temp])

    return df_stat_cat


def calc_non_numeric_statistics(df: pd.DataFrame, target: list, classify: bool) -> dict:
    """
    Calculate non-numeric descriptive statistics

    Parameters
    ----------
    df: DataFrame
        The main dataframe.
    target: list
        The target labels; stored in list
    classify: bool
        Is true, if classification task. False for regression task

    Returns
    -------
    dict
        Dictionary with descriptive statistics (for non-numeric features) for overall dataset, each target and (in case
        of classification) each label.
    """
    dict_non_num_stat = {'overall': create_non_numeric_statistics(df, [t_ for t_ in target if classify])}

    if classify:
        for label_ in target:
            unique = df[label_].unique()
            df_label = []
            for value_ in unique:
                mask = df[label_] == value_ if pd.notna(value_) else df[label_].isnull()
                df_value = \
                    create_non_numeric_statistics(df[mask], [l_ for l_ in [label_] if classify], str(value_) + ' - ')
                df_value = df_value.reindex(dict_non_num_stat['overall'].index, fill_value=0)

                idx = dict_non_num_stat['overall'].index.levels[0]
                s = pd.Series(index=idx, data=[chi_square(df.loc[mask, col_], df.loc[~mask, col_]) for col_ in idx])
                s.name = str(value_) + ' - chi_square'
                s = s.reindex(df_value.index.get_level_values(0))
                s[s.index.duplicated()] = np.nan
                s.index = df_value.index

                df_label.append(df_value.join(s, how='left'))

            dict_non_num_stat[label_] = pd.concat(df_label, axis=1, sort=False)

    return dict_non_num_stat


def save_descriptive_statistics(df: pd.DataFrame, target: list, classify: bool, fn: Union[str, Path],
                                corr_threshold: int = 200):
    """
    Calculate and save descriptive statistics including correlation information to disk.

    Parameters
    ----------
    df: DataFrame
        The main dataframe.
    target: list
        The target labels; stored in list.
    classify: bool
        Is true, if classification task. False for regression task
    fn: str | Path
        The directory where to save the statistics files
    corr_threshold: int, default=200
        Maximum number of columns for which a correlation-DataFrame is computed.
    """
    fn = make_path(fn)

    num_stats, non_num_stats, corr = calc_descriptive_statistics(df, target, classify, corr_threshold=corr_threshold)
    # calculate and save descriptive statistics & correlations
    cols_to_str = list(df.select_dtypes(include=['timedelta64[ns]']).columns)
    convert_rows_to_str(num_stats, cols_to_str, inplace=True, skip=['count', 'mann_whitney_u'])

    write_dfs(num_stats, fn / 'statistics_numeric.xlsx')
    write_dfs(non_num_stats, fn / 'statistics_non_numeric.xlsx')
    if corr is not None:
        write_df(corr, fn / 'correlations.xlsx')

    # delete temp variables and end function
    del num_stats, non_num_stats
    logging.log('Saving descriptive statistics completed')


def calc_descriptive_statistics(df: pd.DataFrame, target: list, classify: bool, corr_threshold: int = 200) \
        -> Tuple[dict, dict, Optional[pd.DataFrame]]:
    """
    Calculate and return descriptive statistics including correlation information

    Parameters
    ----------
    df: DataFrame
        The main dataframe.
    target: list
        The target labels; stored in list
    classify: bool
        Is true, if classification task. False for regression task
    corr_threshold: int, default=200
        Maximum number of columns for which a correlation-DataFrame is computed.

    Returns
    -------
    tuple
        Tuple of numeric and non-numeric statistics (separate dictionaries) and correlation-DataFrame
    """
    if classify:
        convert = [c for c in target if df[c].dtype.name != 'category']
        if convert:
            df = df.copy()
            df[convert] = df[convert].astype('category')

    # calculate descriptive statistics
    dict_stat = calc_numeric_statistics(df, target, classify)
    dict_non_num_stat = calc_non_numeric_statistics(df, target, classify)

    if df.shape[1] <= corr_threshold:
        try:
            corr = df.corr(numeric_only=True)
        except TypeError:
            corr = df.corr()
    else:
        corr = None

    return dict_stat, dict_non_num_stat, corr
