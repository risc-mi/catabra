import pandas as pd
import numpy as np
from ..util.io import Path, make_path, write_df, write_dfs
from ..util import logging
from typing import Union


def calc_numeric_statistics(df: pd.DataFrame, target: list, classify: bool) -> dict:
    """
    Calculates descriptive statistics for numeric features
    :param df: The main dataframe.
    :param target: The target labels; stored in list
    :param classify: Is true, if classification task. False for regression task
    :return: Dictionary with statistics (for numeric features) for entire dataset and each train/test split.
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
    Calculate and save descriptive statistics including correlation information to disk.
    :param df: The main dataframe.
    :param target: The target labels; stored in list
    :param name_: Is true, if classification task. False for regression task
    :return: Returns a dataframe with statistics (for non-numeric features) for a specific split or label
    """
    df_stat_cat = pd.DataFrame()
    for col_ in target + [c for c in df.columns if c not in target]:
        if df[col_].dtype.name in ['bool', 'category'] or (col_ in target):
            temp = pd.DataFrame(df[col_].value_counts()).rename(columns={col_: name_ + 'count'})
            temp = temp.join(
                pd.DataFrame(df[col_].value_counts(normalize=True) * 100).rename(columns={col_: name_ + '%'}))

            arrays = [[col_ for s in list(temp.index)], list(temp.index)]
            tuples = list(zip(*arrays))
            index = pd.MultiIndex.from_tuples(tuples, names=["Feature", "Value"])
            temp = temp.set_index(index)
            df_stat_cat = pd.concat([df_stat_cat, temp])

    del temp
    return df_stat_cat


def calc_non_numeric_statistics(df: pd.DataFrame, target: list, classify: bool) -> dict:
    """
    Calculate and save descriptive statistics including correlation information to disk.
    :param df: The main dataframe.
    :param target: The target labels; stored in list
    :param classify: Is true, if classification task. False for regression task
    :return: Dictionary with statistics (for non-numeric features) for entire dataset and each train/test split.
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


def save_descriptive_statistics(df: pd.DataFrame, target: list, classify: bool, split_masks: list,
                                fn: Union[str, Path]):
    """
    Calculate and save descriptive statistics including correlation information to disk.
    :param df: The main dataframe.
    :param target: The target labels; stored in list
    :param classify: Is true, if classification task. False for regression task
    :param split_masks: Contains the mask information about the train/test splits
    :param fn: The directory where to save the statistics files
    """
    fn = make_path(fn)

    # calculate and save descriptive statistics & correlations for overall dataset
    dict_stat = calc_numeric_statistics(df, target, classify)
    write_dfs(dict_stat, fn / 'statistics_numeric.xlsx')

    write_df(df.corr(), fn / 'correlations.xlsx')

    dict_non_num_stat = calc_non_numeric_statistics(df, target, classify)
    write_dfs(dict_non_num_stat, fn / 'statistics_non_numeric.xlsx')

    # calculate and save descriptive statistics & correlations for individual splits
    if split_masks is not None:
        for key_, mask_ in split_masks.items():
            for value_ in np.unique(mask_):

                mask_red = mask_ == value_
                df_temp = df[mask_red].copy()

                dict_stat = calc_numeric_statistics(df_temp, target, classify)
                write_dfs(dict_stat, fn / (key_ + '_' + str(value_)) / 'statistics_numeric.xlsx')

                write_df(df_temp.corr(),  fn / (key_ + '_' + str(value_)) / 'correlations.xlsx')

                dict_non_num_stat = calc_non_numeric_statistics(df_temp, target, classify)
                write_dfs(dict_non_num_stat, fn / (key_ + '_' + str(value_)) / 'statistics_non_numeric.xlsx')

    # delete temp variables and end function
    del df_temp, dict_stat, dict_non_num_stat, mask_red
    logging.log(f'Saving descriptive statistics completed')
