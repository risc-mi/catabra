#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from catabra.util.longitudinal import resample_eav

from .util import compare_dataframes, create_random_data, resample_eav_slow

seed = 93584


def test_corner_cases():
    df = pd.DataFrame(
        data=dict(
            entity=[1000] * 10 + [1001] * 2,
            timestamp=[0, 10, 20, 30, 5, 9, 25, 34, 7, 17, 1, -1],
            attribute=['attr_1', 'attr_1', 'attr_1', 'attr_1', 'attr_2', 'attr_2',
                       'attr_2', 'attr_2', 'attr_3', 'attr_3', 'attr_2', 'attr_2'],
            value=[np.nan, 85.7, 151.4, 137.8, np.nan, 49., 50., 50., 0.1, 0.2, 38.7, 36.1]
        )
    )

    # no data, only NaN values, both "attr_3" values, only one "attr_3" value
    windows = pd.DataFrame(
        data=dict(
            entity=[1000] * 4 + [1001],
            start=[-1, 0, 1, 7, 0],
            duration=[1, 8, 40, 10, 2]
        )
    )
    windows['stop'] = windows['start'] + windows['duration']
    windows.columns = pd.MultiIndex.from_tuples(
        [('entity', ''), ('timestamp', 'start'), ('duration', ''), ('timestamp', 'stop')]
    )

    agg = {'attr_1': 'p25', 'attr_2': ['sum', 'count'], 'attr_3': ['r-1', 'r0', 't-1', 't0', 'r1']}
    out1 = resample_eav(
        df,
        windows,
        agg,
        entity_col='entity',
        time_col='timestamp',
        attribute_col='attribute',
        value_col='value',
        include_start=True,
        include_stop=False
    )
    ground_truth = resample_eav_slow(
        df,
        windows,
        agg,
        entity_col='entity',
        time_col='timestamp',
        attribute_col='attribute',
        value_col='value',
        include_start=True,
        include_stop=False
    )
    compare_dataframes(out1, ground_truth)


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
        'include_start',
        'include_stop',
        'na_windows'
    ],
    argvalues=[
        ('random', True, False, False),
        ('regular non-overlapping', False, True, False),
        ('regular', True, True, False),
        ('non-overlapping', False, False, False),
        ('no_start', True, False, False),
        ('regular non-overlapping', True, False, False)
    ]
)
def test_windows(pattern: str, include_start: bool, include_stop: bool, na_windows: bool):
    if pattern == 'mixed':
        dfs = []
        windows = []
        i = 0
        for p in ['random', 'regular', 'regular non-overlapping', 'non-overlapping']:
            df1, windows1 = create_random_data(2500, 250, attributes=['attr_' + str(i) for i in range(1, 7)],
                                               window_pattern=p, seed=seed)
            df1['entity'] += i
            windows1[('entity', '')] += i
            i = max(df1['entity'].max(), windows1[('entity', '')].max()) + 1
            dfs.append(df1)
            windows.append(windows1)
        df = pd.concat(dfs, axis=0, sort=False)
        windows = pd.concat(windows, axis=0, sort=False)
    elif pattern == 'no_start':
        df, windows = create_random_data(10000, 100, attributes=['attr_' + str(i) for i in range(1, 7)],
                                         window_pattern='random', seed=seed)
        windows.drop([('timestamp', 'start')], axis=1, inplace=True)
    else:
        df, windows = create_random_data(10000, 100, attributes=['attr_' + str(i) for i in range(1, 7)],
                                         window_pattern=pattern, seed=seed)

    if na_windows:
        rng = np.random.RandomState(seed=seed)
        windows.drop([('timestamp', 'stop')], axis=1, inplace=True)
        windows.loc[rng.randint(2, size=len(windows), dtype=bool), ('timestamp', 'start')] = None

    agg = {'attr_1': ['mean', 'size', 'skew', 'sem'], 'attr_3': ['p25', 'p50', 'p99'], 'attr_4': ['p50', 'r-1', 't0'],
           'attr_6': ['r0', 't-1', 'r2']}
    kwargs = dict(
        entity_col='entity',
        time_col='timestamp',
        attribute_col='attribute',
        value_col='value',
        include_start=include_start,
        include_stop=include_stop
    )
    out1 = resample_eav(df, windows, agg, optimize='time', **kwargs)
    out2 = resample_eav(df, windows, agg, optimize='memory', **kwargs)
    ground_truth = resample_eav_slow(df, windows, agg, **kwargs)
    compare_dataframes(out1, ground_truth)
    compare_dataframes(out2, ground_truth)


@pytest.mark.slow
def test_windows_mixed():
    return test_windows('mixed', True, False, False)


@pytest.mark.slow
def test_single_entity_attribute():
    df, windows = create_random_data(10000, 100, n_entities=1, window_pattern='random', value_dtype='int', seed=seed)

    agg = {'attr_1': ['mean', 'size', 'std', 'p25', 'p50', 'p99', 'r-1', 't0']}
    kwargs = dict(
        time_col='timestamp',
        value_col='value',
        include_start=True,
        include_stop=True
    )
    out1 = resample_eav(df, windows, agg, optimize='time', **kwargs,)
    out2 = resample_eav(df, windows, agg, optimize='memory', **kwargs)
    ground_truth = resample_eav_slow(df, windows, agg, **kwargs)
    compare_dataframes(out1, ground_truth)
    compare_dataframes(out2, ground_truth)


def test_categorical():
    df, windows = create_random_data(10000, 100, attributes=['attr_1', 'attr_2'], value_dtype='category',
                                     window_pattern='random', seed=seed)

    agg = {'attr_1': ['mode', 'mode_count', 'size', 'r-1', 't0'], 'attr_2': ['mode', 'count', 'r2']}
    kwargs = dict(
        entity_col='entity',
        time_col='timestamp',
        attribute_col='attribute',
        value_col='value',
        include_start=True,
        include_stop=True
    )
    out1 = resample_eav(df, windows, agg, optimize='time', **kwargs)
    out2 = resample_eav(df, windows, agg, optimize='memory', **kwargs)
    ground_truth = resample_eav_slow(df, windows, agg, **kwargs)
    compare_dataframes(out1, ground_truth)
    compare_dataframes(out2, ground_truth)


def test_custom_agg(seed=None):
    df, windows = create_random_data(10000, 100, window_pattern='random', seed=seed)

    def frac_between_0_1(x: pd.DataFrame):
        return x['value'].between(0, 1).groupby(level=0).mean().to_frame('frac_in[0, 1]')

    def frac_between_1_10(x: pd.DataFrame):
        return x['value'].between(1, 10).groupby(level=0).mean()

    agg = {'attr_1': [frac_between_0_1, frac_between_1_10]}
    kwargs = dict(
        entity_col='entity',
        time_col='timestamp',
        value_col='value',
        include_start=True,
        include_stop=False
    )
    out1 = resample_eav(df, windows, agg, **kwargs)
    ground_truth = resample_eav_slow(df, windows, agg, **kwargs)
    compare_dataframes(out1, ground_truth)


def test_include_stop():
    df, _ = create_random_data(10000, 1, n_entities=200, attributes=['attr_' + str(i) for i in range(1, 10)],
                               window_pattern='random', time_dtype='timestamp', seed=seed)
    attrs = df['attribute'].value_counts().index
    duration = (df['timestamp'].max() - df['timestamp'].min()) * 0.1
    windows1 = df.loc[df['attribute'] == attrs[1], ['entity', 'timestamp']].copy()
    windows1.columns = pd.MultiIndex.from_tuples([('entity', ''), ('timestamp', 'stop')])
    windows2 = windows1.copy()
    windows2[('timestamp', 'stop')] = windows2[('timestamp', 'stop')] - 0.5 * duration
    windows = pd.concat([windows1, windows2], axis=0, sort=False)
    windows[('timestamp', 'start')] = windows[('timestamp', 'stop')] - duration

    agg = {attrs[0]: ['sum', 'max'], attrs[1]: ['r-1', 'r-2', 't-1', 't-2'], attrs[-2]: 'size', attrs[-1]: 'min'}
    kwargs = dict(
        entity_col='entity',
        time_col='timestamp',
        attribute_col='attribute',
        value_col='value',
        include_start=True,
        include_stop=True
    )
    out1 = resample_eav(df, windows, agg, optimize='time', **kwargs)
    out2 = resample_eav(df, windows, agg, optimize='memory', **kwargs)
    ground_truth = resample_eav_slow(df, windows, agg, **kwargs)
    compare_dataframes(out1, ground_truth)
    compare_dataframes(out2, ground_truth)

    # set `include_stop` to False
    kwargs = dict(
        entity_col='entity',
        time_col='timestamp',
        attribute_col='attribute',
        value_col='value',
        include_start=True,
        include_stop=False
    )
    out1 = resample_eav(df, windows, agg, optimize='time', **kwargs)
    out2 = resample_eav(df, windows, agg, optimize='memory', **kwargs)
    ground_truth = resample_eav_slow(df, windows, agg, **kwargs)
    compare_dataframes(out1, ground_truth)
    compare_dataframes(out2, ground_truth)


@pytest.mark.manual
@pytest.mark.parametrize(
    argnames=['dd'],
    argvalues=[(dd,), (None,)],
    ids=['dask', 'pandas']
)
def test_large(dd):
    df, _ = create_random_data(10000000, 1, n_entities=20000, attributes=['attr_' + str(i) for i in range(1, 50)],
                               window_pattern='random', time_dtype='timestamp', seed=seed)
    attrs = df['attribute'].value_counts().index
    duration = (df['timestamp'].max() - df['timestamp'].min()) * 0.1
    windows1 = df.loc[df['attribute'] == attrs[2], ['entity', 'timestamp']].copy()
    windows1.columns = pd.MultiIndex.from_tuples([('entity', ''), ('timestamp', 'stop')])
    windows2 = windows1.copy()
    windows2[('timestamp', 'stop')] = windows2[('timestamp', 'stop')] - 0.5 * duration
    windows = pd.concat([windows1, windows2], axis=0, sort=False)
    windows[('timestamp', 'start')] = windows[('timestamp', 'stop')] - duration

    agg = {attrs[0]: ['sum', 'max'], attrs[1]: 'min', attrs[2]: ['r-1', 'r-2', 't-1', 't-2'], attrs[-2]: 'size',
           attrs[-1]: 'median'}
    kwargs = dict(
        entity_col='entity',
        time_col='timestamp',
        attribute_col='attribute',
        value_col='value',
        include_start=False,
        include_stop=False
    )

    out1 = resample_eav(df, windows, agg, optimize='time', **kwargs)
    if dd is None:
        out2 = resample_eav( df, windows, agg, optimize='memory', **kwargs)
        # don't compute ground truth, as this would take way too long
        compare_dataframes(out1, out2)
    else:
        # _fun_name='resample_eav(dask, pandas)'
        ddf = dd.from_pandas(df.set_index('entity'), npartitions=32, sort=True)
        out_ddf1 = resample_eav(ddf, windows, agg, optimize='time', **kwargs)
        compare_dataframes(out1, out_ddf1)
        # 'resample_eav(dask, dask)'
        out_ddf2 = resample_eav(ddf, dd.from_pandas(windows.reset_index(drop=True), npartitions=8), agg,
                           optimize='time', **kwargs)
        assert isinstance(out_ddf2, dd.DataFrame)
        out_ddf2 = dd.compute(out_ddf2, _prefix='  ', _fun_name='dask.compute')[0]
        out_ddf2.sort_index(inplace=True)           # re-establish correct order of rows
        out_ddf2.index = out1.index
        compare_dataframes(out1, out_ddf2)