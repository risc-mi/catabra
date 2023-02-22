#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import pandas as pd
import numpy as np

from ..util import profile
from .util import resample_eav_slow, compare_dataframes, create_random_data
from catabra.util.longitudinal import resample_eav


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
    kwargs = dict(
        entity_col='entity',
        time_col='timestamp',
        attribute_col='attribute',
        value_col='value',
        include_start=True,
        include_stop=False
    )
    out1 = profile(resample_eav, df, windows, agg, **kwargs, _prefix='  ')
    ground_truth = resample_eav_slow(df, windows, agg, **kwargs)
    compare_dataframes(out1, ground_truth)


def _test_windows(pattern: str, include_start: bool = True, include_stop: bool = False, seed=None,
                  return_dfs: bool = False, na_windows: bool = False):
    if pattern == 'mixed':
        dfs = []
        windows = []
        i = 0
        for p in ['random', 'regular', 'regular non-overlapping', 'non-overlapping']:
            df1, windows1 = create_random_data(25000, 2500, attributes=['attr_' + str(i) for i in range(1, 7)],
                                               window_pattern=p, seed=seed)
            df1['entity'] += i
            windows1[('entity', '')] += i
            i = max(df1['entity'].max(), windows1[('entity', '')].max()) + 1
            dfs.append(df1)
            windows.append(windows1)
        df = pd.concat(dfs, axis=0, sort=False)
        windows = pd.concat(windows, axis=0, sort=False)
    elif pattern == 'no_start':
        df, windows = create_random_data(100000, 1000, attributes=['attr_' + str(i) for i in range(1, 7)],
                                         window_pattern='random', seed=seed)
        windows.drop([('timestamp', 'start')], axis=1, inplace=True)
    else:
        df, windows = create_random_data(100000, 1000, attributes=['attr_' + str(i) for i in range(1, 7)],
                                         window_pattern=pattern, seed=seed)

    if na_windows:
        rng = np.random.RandomState(seed=seed)
        windows.drop([('timestamp', 'stop')], axis=1, inplace=True)
        windows.loc[rng.randint(2, size=len(windows), dtype=np.bool), ('timestamp', 'start')] = None

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
    out1 = profile(resample_eav, df, windows, agg, optimize='time', **kwargs,
                   _prefix='  ', _include_kwargs=['optimize'])
    out2 = profile(resample_eav, df, windows, agg, optimize='memory', **kwargs,
                   _prefix='  ', _include_kwargs=['optimize'])
    ground_truth = resample_eav_slow(df, windows, agg, **kwargs)
    if return_dfs:
        return df, out1, out2, ground_truth
    else:
        compare_dataframes(out1, ground_truth)
        compare_dataframes(out2, ground_truth)


def test_random_windows(**kwargs):
    return _test_windows('random', **kwargs)


def test_regular_non_overlapping_windows(**kwargs):
    return _test_windows('regular non-overlapping', include_start=False, include_stop=True, **kwargs)


def test_regular_windows(**kwargs):
    return _test_windows('regular', include_start=True, include_stop=True, **kwargs)


def test_non_overlapping_windows(**kwargs):
    return _test_windows('non-overlapping', include_start=False, include_stop=False, **kwargs)


def test_mixed_windows(**kwargs):
    return _test_windows('mixed', **kwargs)


def test_infinite_windows(**kwargs):
    return _test_windows('no_start', **kwargs)


def test_na_windows(**kwargs):
    return _test_windows('regular non-overlapping', **kwargs)


def test_single_entity_attribute(seed=None):
    df, windows = create_random_data(100000, 1000, n_entities=1, window_pattern='random', value_dtype='int', seed=seed)

    agg = {'attr_1': ['mean', 'size', 'std', 'p25', 'p50', 'p99', 'r-1', 't0']}
    kwargs = dict(
        time_col='timestamp',
        value_col='value',
        include_start=True,
        include_stop=True
    )
    out1 = profile(resample_eav, df, windows, agg, optimize='time', **kwargs,
                   _prefix='  ', _include_kwargs=['optimize'])
    out2 = profile(resample_eav, df, windows, agg, optimize='memory', **kwargs,
                   _prefix='  ', _include_kwargs=['optimize'])
    ground_truth = resample_eav_slow(df, windows, agg, **kwargs)
    compare_dataframes(out1, ground_truth)
    compare_dataframes(out2, ground_truth)


def test_categorical(seed=None):
    df, windows = create_random_data(100000, 1000, attributes=['attr_1', 'attr_2'], value_dtype='category',
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
    out1 = profile(resample_eav, df, windows, agg, optimize='time', **kwargs,
                   _prefix='  ', _include_kwargs=['optimize'])
    out2 = profile(resample_eav, df, windows, agg, optimize='memory', **kwargs,
                   _prefix='  ', _include_kwargs=['optimize'])
    ground_truth = resample_eav_slow(df, windows, agg, **kwargs)
    compare_dataframes(out1, ground_truth)
    compare_dataframes(out2, ground_truth)


def test_custom_agg(seed=None):
    df, windows = create_random_data(100000, 1000, window_pattern='random', seed=seed)

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
    out1 = profile(resample_eav, df, windows, agg, **kwargs, _prefix='  ')
    ground_truth = resample_eav_slow(df, windows, agg, **kwargs)
    compare_dataframes(out1, ground_truth)


def test_include_stop(seed=None):
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
    out1 = profile(resample_eav, df, windows, agg, optimize='time', **kwargs,
                   _prefix='  ', _include_kwargs=['optimize'])
    out2 = profile(resample_eav, df, windows, agg, optimize='memory', **kwargs,
                   _prefix='  ', _include_kwargs=['optimize'])
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
    out1 = profile(resample_eav, df, windows, agg, optimize='time', **kwargs,
                   _prefix='  ', _include_kwargs=['optimize'])
    out2 = profile(resample_eav, df, windows, agg, optimize='memory', **kwargs,
                   _prefix='  ', _include_kwargs=['optimize'])
    ground_truth = resample_eav_slow(df, windows, agg, **kwargs)
    compare_dataframes(out1, ground_truth)
    compare_dataframes(out2, ground_truth)


def test_large(seed=None, dd=None):
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

    out1 = profile(resample_eav, df, windows, agg, optimize='time', **kwargs,
                   _prefix='  ', _include_kwargs=['optimize'])
    if dd is None:
        out2 = profile(resample_eav, df, windows, agg, optimize='memory', **kwargs,
                       _prefix='  ', _include_kwargs=['optimize'])
        # don't compute ground truth, as this would take way too long
        compare_dataframes(out1, out2)
    else:
        ddf = dd.from_pandas(df.set_index('entity'), npartitions=32, sort=True)
        out_ddf1 = profile(resample_eav, ddf, windows, agg, optimize='time', **kwargs,
                           _prefix='  ', _fun_name='resample_eav(dask, pandas)')
        compare_dataframes(out1, out_ddf1)

        out_ddf2 = profile(resample_eav, ddf, dd.from_pandas(windows.reset_index(drop=True), npartitions=8), agg,
                           optimize='time', **kwargs, _prefix='  ', _fun_name='resample_eav(dask, dask)')
        assert isinstance(out_ddf2, dd.DataFrame)
        out_ddf2 = profile(dd.compute, out_ddf2, _prefix='  ', _fun_name='dask.compute')[0]
        out_ddf2.sort_index(inplace=True)           # re-establish correct order of rows
        out_ddf2.index = out1.index
        compare_dataframes(out1, out_ddf2)


def test_dask(seed=None):
    try:
        import dask.dataframe as dd
    except ImportError:
        print('Failed to test Dask: package not found.')
        return

    return test_large(seed=seed, dd=dd)


def main(seed=None):
    tic = pd.Timestamp.now()
    rng = np.random.RandomState(seed=seed)

    profile(test_corner_cases)

    s = rng.randint(2 ** 31)
    profile(test_random_windows, seed=s, _include_kwargs=['seed'])

    s = rng.randint(2 ** 31)
    profile(test_regular_non_overlapping_windows, seed=s, _include_kwargs=['seed'])

    s = rng.randint(2 ** 31)
    profile(test_regular_windows, seed=s, _include_kwargs=['seed'])

    s = rng.randint(2 ** 31)
    profile(test_non_overlapping_windows, seed=s, _include_kwargs=['seed'])

    s = rng.randint(2 ** 31)
    profile(test_mixed_windows, seed=s, _include_kwargs=['seed'])

    s = rng.randint(2 ** 31)
    profile(test_infinite_windows, seed=s, _include_kwargs=['seed'])

    s = rng.randint(2 ** 31)
    profile(test_na_windows, seed=s, _include_kwargs=['seed'])

    s = rng.randint(2 ** 31)
    profile(test_single_entity_attribute, seed=s, _include_kwargs=['seed'])

    s = rng.randint(2 ** 31)
    profile(test_categorical, seed=s, _include_kwargs=['seed'])

    s = rng.randint(2 ** 31)
    profile(test_custom_agg, seed=s, _include_kwargs=['seed'])

    s = rng.randint(2 ** 31)
    profile(test_include_stop, seed=s, _include_kwargs=['seed'])

    s = rng.randint(2 ** 31)
    profile(test_large, seed=s, _include_kwargs=['seed'])

    s = rng.randint(2 ** 31)
    profile(test_dask, seed=s, _include_kwargs=['seed'])

    print('Tests finished successfully!')
    print('    elapsed time:', pd.Timestamp.now() - tic)


if __name__ == '__main__':
    main()
