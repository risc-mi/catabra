#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from catabra.util.longitudinal import resample_interval

from .util import compare_dataframes, create_random_data, resample_interval_slow

seed = 16873


@pytest.mark.parametrize(
    ids=[
        'random',
        'regular_non_overlapping',
        'regular',
        'non_overlapping',
        'infinite',
        'na_windows'
    ],
    argnames=[
        'pattern',
        'include_df_start',
        'include_df_stop',
        'na_windows'
    ],
    argvalues=[
        ('random', True, True, False),
        ('regular non-overlapping', False, True, False),
        ('regular', True, False, False),
        ('non-overlapping', True, True, False),
        ('no_start', True, False, False),
        ('regular non-overlapping', True, False, False)
    ]
)
def test_windows(pattern: str, include_df_start: bool, include_df_stop: bool, na_windows: bool):
    if pattern == 'mixed':
        dfs = []
        windows = []
        i = 0
        for p in ['random', 'regular', 'regular non-overlapping', 'non-overlapping']:
            df1, windows1 = create_random_data(25000, 2500, attributes=['attr_' + str(i) for i in range(1, 7)],
                                               intervals=True, window_pattern=p, seed=seed)
            df1['entity'] += i
            windows1[('entity', '')] += i
            i = max(df1['entity'].max(), windows1[('entity', '')].max()) + 1
            dfs.append(df1)
            windows.append(windows1)
        df = pd.concat(dfs, axis=0, sort=False)
        windows = pd.concat(windows, axis=0, sort=False)
    elif pattern == 'no_start':
        df, windows = create_random_data(100000, 1000, attributes=['attr_' + str(i) for i in range(1, 7)],
                                         intervals=True, window_pattern='random', seed=seed)
        windows.drop([('timestamp', 'start')], axis=1, inplace=True)
    else:
        df, windows = create_random_data(100000, 1000, attributes=['attr_' + str(i) for i in range(1, 7)],
                                         intervals=True, window_pattern=pattern, seed=seed)

    if not include_df_start:
        df.drop('start', axis=1, inplace=True)
    elif not include_df_stop:
        df.drop('stop', axis=1, inplace=True)

    if na_windows:
        rng = np.random.RandomState(seed=seed)
        windows.drop([('timestamp', 'stop')], axis=1, inplace=True)
        windows.loc[rng.randint(2, size=len(windows), dtype=np.bool), ('timestamp', 'start')] = None

    kwargs = dict(
        attributes=['attr_1', 'attr_2', 'attr_5', 'attr_4'],
        entity_col='entity',
        time_col='timestamp',
        start_col='start',
        stop_col='stop',
        attribute_col='attribute',
        value_col='value'
    )
    out1 = resample_interval(df, windows, **kwargs)
    ground_truth = resample_interval_slow(df, windows, **kwargs)
    compare_dataframes(out1, ground_truth)


@pytest.mark.slow
def test_windows_mixed():
    # extracted to own tests to mark it as slow
    test_windows('mixed', True, True, False)


def test_single_entity_attribute():
    df, windows = create_random_data(100000, 1000, n_entities=1, intervals=True, window_pattern='random',
                                     value_dtype='int', seed=seed)

    kwargs = dict(
        start_col='start',
        stop_col='stop',
        time_col='timestamp',
        value_col='value'
    )
    out1 = resample_interval(df, windows, **kwargs)
    ground_truth = resample_interval_slow(df, windows, **kwargs)
    compare_dataframes(out1, ground_truth)


@pytest.mark.slow
def test_large():
    df, _ = create_random_data(10000000, 1, n_entities=20000, attributes=['attr_' + str(i) for i in range(1, 50)],
                               intervals=True, window_pattern='random', time_dtype='timestamp', seed=seed)
    attrs = df['attribute'].value_counts().index
    duration = (df['stop'].max() - df['start'].min()) * 0.1
    windows1 = df.loc[df['attribute'] == attrs[2], ['entity', 'stop']].copy()
    windows1.columns = pd.MultiIndex.from_tuples([('entity', ''), ('timestamp', 'stop')])
    windows2 = windows1.copy()
    windows2[('timestamp', 'stop')] = windows2[('timestamp', 'stop')] - 0.5 * duration
    windows = pd.concat([windows1, windows2], axis=0, sort=False)
    windows[('timestamp', 'start')] = windows[('timestamp', 'stop')] - duration

    kwargs = dict(
        entity_col='entity',
        start_col='start',
        stop_col='stop',
        time_col='timestamp',
        attribute_col='attribute',
        value_col='value',
        attributes=[attrs[0], attrs[1], attrs[2], attrs[-2], attrs[-1]]
    )

    out1 = resample_interval(df, windows, **kwargs)
    ddf = dd.from_pandas(df.set_index('entity'), npartitions=32, sort=True)
    out_ddf1 = resample_interval(ddf, windows, **kwargs)
    compare_dataframes(out1, out_ddf1)

    out_ddf2 = resample_interval(ddf, dd.from_pandas(windows.reset_index(drop=True), npartitions=8), **kwargs)
    assert isinstance(out_ddf2, dd.DataFrame)
    out_ddf2 = dd.compute(out_ddf2)[0]
    out_ddf2.sort_index(inplace=True)           # re-establish correct order of rows
    out_ddf2.index = out1.index
    compare_dataframes(out1, out_ddf2)