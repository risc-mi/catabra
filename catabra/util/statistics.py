import pandas as pd
import numpy as np
from ..util.io import Path, make_path, write_df, write_dfs
from ..util import logging
from typing import Union, Tuple


def calc_numeric_statistics(df: pd.DataFrame, target: list, classify: bool) -> dict:
    """
    Calculate descriptive statistics for numeric features for a specific dataframe
    :param df: The main dataframe.
    :param target: The target labels; stored in list
    :param classify: Is true, if classification task. False for regression task
    :return: Dictionary with statistics (for numeric features) for entire dataset, each target and
    (in case of classification) each label.
    """
    df_temp = df.copy()
    dict_stat = {'overall': df.describe().T}
    if classify:
        for label_ in target:
            dict_stat[label_] = df.groupby(label_).describe().T
            df_temp[label_] = 'Overall'
            dict_stat[label_] = dict_stat[label_].join(df_temp.groupby(label_).describe().T)

    del df_temp
    return dict_stat


def create_non_numeric_statistics(df: pd.DataFrame, target: list, name_: str = '') -> pd.DataFrame:
    """
    Calculate descriptive statistics for non-numeric features for a specific dataframe
    :param df: The main dataframe.
    :param target: The target labels; stored in list
    :param name_: Name of the label. Used for naming the columns
    :return: Returns a dataframe with statistics (for non-numeric features)
    """
    df_stat_cat = pd.DataFrame()
    for col_ in target + [c for c in df.columns if c not in target]:
        if df[col_].dtype.name in ['bool', 'category'] or (col_ in target):
            temp = pd.DataFrame(df[col_].value_counts()).rename(columns={col_: name_ + 'count'})
            temp = temp.join(
                pd.DataFrame(df[col_].value_counts(normalize=True) * 100).rename(columns={col_: name_ + '%'}))

            arrays = [[col_ for s in list(temp.index)], list(temp.index)]
            index = pd.MultiIndex.from_tuples(list(zip(*arrays)), names=["Feature", "Value"])
            temp = temp.set_index(index)
            df_stat_cat = pd.concat([df_stat_cat, temp])

    del temp
    return df_stat_cat


def calc_non_numeric_statistics(df: pd.DataFrame, target: list, classify: bool) -> dict:
    """
    Calculate non-numeric descriptive statistics
    :param df: The main dataframe.
    :param target: The target labels; stored in list
    :param classify: Is true, if classification task. False for regression task
    :return: Dictionary with descriptive statistics (for non-numeric features) for overall dataset,
    each target and (in case of classification) each label.
    """
    dict_non_num_stat = {'overall': create_non_numeric_statistics(df, [t_ for t_ in target if classify])}

    for label_ in target:
        c_ = df[label_].notnull()
        df_non_num_stat = create_non_numeric_statistics(df[c_], [l_ for l_ in [label_] if classify], 'Overall - ')

        if classify:
            for value_ in df[label_].unique():
                mask = df[label_] == value_
                df_non_num_stat = df_non_num_stat.join(create_non_numeric_statistics(df[mask],
                                                                                     [l_ for l_ in [label_] if classify],
                                                                                     str(value_) + ' - '))

        dict_non_num_stat[label_] = df_non_num_stat

    del df_non_num_stat
    return dict_non_num_stat


def save_descriptive_statistics(df: pd.DataFrame, target: list, classify: bool, fn: Union[str, Path]):
    """
    Calculate and save descriptive statistics including correlation information to disk.
    :param df: The main dataframe.
    :param target: The target labels; stored in list
    :param classify: Is true, if classification task. False for regression task
    :param fn: The directory where to save the statistics files
    """
    fn = make_path(fn)
    if classify:
        df[target] = df[target].astype('category')

    # calculate and save descriptive statistics & correlations
    dict_stat = calc_numeric_statistics(df, target, classify)
    cols_to_str = list(df.select_dtypes(include=['timedelta64[ns]']).columns)
    dict_stat = convert_to_str(dict_stat, cols_to_str)
    write_dfs(dict_stat, fn / 'statistics_numeric.xlsx')

    dict_non_num_stat = calc_non_numeric_statistics(df, target, classify)
    write_dfs(dict_non_num_stat, fn / 'statistics_non_numeric.xlsx')

    write_df(df.corr(), fn / 'correlations.xlsx')

    # delete temp variables and end function
    del dict_stat, dict_non_num_stat
    logging.log(f'Saving descriptive statistics completed')


def calc_descriptive_statistics(df: pd.DataFrame, target: list, classify: bool) -> Tuple[dict, dict, pd.DataFrame]:
    """
    Calculate and return descriptive statistics including correlation information
    :param df: The main dataframe.
    :param target: The target labels; stored in list
    :param classify: Is true, if classification task. False for regression task
    :return: Tuple of numeric and non-numeric statistics (separate dictionaries) and correlation-DataFrame
    """
    if classify:
        df[target] = df[target].astype('category')

    # calculate descriptive statistics
    dict_stat = calc_numeric_statistics(df, target, classify)
    dict_non_num_stat = calc_non_numeric_statistics(df, target, classify)

    return dict_stat, dict_non_num_stat, df.corr()


def convert_to_str(d: dict, cols_to_str: list) -> dict:
    for key_ in list(d.keys()):
        for c_ in cols_to_str:
            d[key_].iloc[d[key_].index == c_, 1:] = d[key_].iloc[d[key_].index == c_, 1:].astype(str)
    return d
