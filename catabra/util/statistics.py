import pandas as pd
from catabra.base.io import Path, make_path, write_df, write_dfs, convert_rows_to_str
from ..base import logging
from typing import Union, Tuple, Optional


def calc_numeric_statistics(df: pd.DataFrame, target: list, classify: bool) -> dict:
    """
    Calculate descriptive statistics for numeric features for a specific dataframe
    :param df: The main dataframe.
    :param target: The target labels; stored in list
    :param classify: Is true, if classification task. False for regression task
    :return: Dictionary with statistics (for numeric features) for entire dataset, each target and
    (in case of classification) each label.
    """
    df = df.copy()
    dict_stat = {'overall': df.describe().T}
    if classify:
        for label_ in target:
            df[label_] = df[label_].astype(str).fillna('NaN')
            df_stat_cat = pd.DataFrame()
            df_stat_temp = df.groupby(label_).describe()

            for col_ in list(dict_stat['overall'].index):
                temp = df_stat_temp.iloc[:, df_stat_temp.columns.get_level_values(0) == col_]
                temp.columns = temp.columns.droplevel()
                arrays = [[col_ for _ in list(temp.index)], list(temp.index)]
                index = pd.MultiIndex.from_tuples(list(zip(*arrays)), names=["Feature", label_])
                df_stat_cat = pd.concat([df_stat_cat, temp.set_index(index)])

            df_stat_cat['count'] = df_stat_cat['count'].astype(int)
            dict_stat[label_] = df_stat_cat

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
    :param df: The main dataframe.
    :param target: The target labels; stored in list
    :param classify: Is true, if classification task. False for regression task
    :return: Dictionary with descriptive statistics (for non-numeric features) for overall dataset,
    each target and (in case of classification) each label.
    """
    dict_non_num_stat = {'overall': create_non_numeric_statistics(df, [t_ for t_ in target if classify])}

    for label_ in target:
        df_non_num_stat = create_non_numeric_statistics(df, [l_ for l_ in [label_] if classify], 'Overall - ')

        if classify:
            for value_ in df[label_].unique():
                mask = df[label_] == value_ if pd.notna(value_) else df[label_].isnull()
                df_non_num_stat = df_non_num_stat.join(create_non_numeric_statistics(df[mask],
                                                                                     [l_ for l_ in [label_] if classify],
                                                                                     str(value_) + ' - '))

        dict_non_num_stat[label_] = df_non_num_stat

    return dict_non_num_stat


def save_descriptive_statistics(df: pd.DataFrame, target: list, classify: bool, fn: Union[str, Path],
                                corr_threshold: int = 200):
    """
    Calculate and save descriptive statistics including correlation information to disk.
    :param df: The main dataframe.
    :param target: The target labels; stored in list
    :param classify: Is true, if classification task. False for regression task
    :param fn: The directory where to save the statistics files
    :param corr_threshold: Maximum number of columns for which a correlation-DataFrame is computed.
    """
    fn = make_path(fn)

    num_stats, non_num_stats, corr = calc_descriptive_statistics(df, target, classify, corr_threshold=corr_threshold)
    # calculate and save descriptive statistics & correlations
    cols_to_str = list(df.select_dtypes(include=['timedelta64[ns]']).columns)
    convert_rows_to_str(num_stats, cols_to_str, inplace=True, skip=['count'])

    write_dfs(num_stats, fn / 'statistics_numeric.xlsx')
    write_dfs(non_num_stats, fn / 'statistics_non_numeric.xlsx')
    if corr is not None:
        write_df(corr, fn / 'correlations.xlsx')

    # delete temp variables and end function
    del num_stats, non_num_stats
    logging.log(f'Saving descriptive statistics completed')


def calc_descriptive_statistics(df: pd.DataFrame, target: list, classify: bool, corr_threshold: int = 200) \
        -> Tuple[dict, dict, Optional[pd.DataFrame]]:
    """
    Calculate and return descriptive statistics including correlation information
    :param df: The main dataframe.
    :param target: The target labels; stored in list
    :param classify: Is true, if classification task. False for regression task
    :param corr_threshold: Maximum number of columns for which a correlation-DataFrame is computed.
    :return: Tuple of numeric and non-numeric statistics (separate dictionaries) and correlation-DataFrame
    """
    if classify:
        convert = [c for c in target if df[c].dtype.name != 'category']
        if convert:
            df = df.copy()
            df[convert] = df[convert].astype('category')

    # calculate descriptive statistics
    dict_stat = calc_numeric_statistics(df, target, classify)
    dict_non_num_stat = calc_non_numeric_statistics(df, target, classify)

    return dict_stat, dict_non_num_stat, (df.corr() if df.shape[1] <= corr_threshold else None)
