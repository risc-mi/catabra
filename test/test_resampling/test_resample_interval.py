import pandas as pd
import numpy as np

from ..util import profile
from .util import resample_interval_slow, compare_dataframes, create_random_data
from catabra.util.resampling import resample_interval


def _test_windows(pattern: str, include_df_start: bool = True, include_df_stop: bool = True, seed=None,
                  return_dfs: bool = False, na_windows: bool = False):
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
    out1 = profile(resample_interval, df, windows, **kwargs, _prefix='  ')
    ground_truth = resample_interval_slow(df, windows, **kwargs)
    if return_dfs:
        return df, out1, ground_truth
    else:
        compare_dataframes(out1, ground_truth)


def test_random_windows(**kwargs):
    return _test_windows('random', **kwargs)


def test_regular_non_overlapping_windows(**kwargs):
    return _test_windows('regular non-overlapping', include_df_start=False, **kwargs)


def test_regular_windows(**kwargs):
    return _test_windows('regular', include_df_stop=False, **kwargs)


def test_non_overlapping_windows(**kwargs):
    return _test_windows('non-overlapping', **kwargs)


def test_mixed_windows(**kwargs):
    return _test_windows('mixed', **kwargs)


def test_infinite_windows(**kwargs):
    return _test_windows('no_start', include_df_stop=False, **kwargs)


def test_na_windows(**kwargs):
    return _test_windows('regular non-overlapping', include_df_stop=False, **kwargs)


def test_single_entity_attribute(seed=None):
    df, windows = create_random_data(100000, 1000, n_entities=1, intervals=True, window_pattern='random',
                                     value_dtype='int', seed=seed)

    kwargs = dict(
        start_col='start',
        stop_col='stop',
        time_col='timestamp',
        value_col='value'
    )
    out1 = profile(resample_interval, df, windows, **kwargs, _prefix='  ')
    ground_truth = resample_interval_slow(df, windows, **kwargs)
    compare_dataframes(out1, ground_truth)


def test_large(seed=None, dd=None):
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

    out1 = profile(resample_interval, df, windows, **kwargs, _prefix='  ')
    if dd is None:
        # don't compute ground truth, as this would take way too long
        pass
    else:
        ddf = dd.from_pandas(df.set_index('entity'), npartitions=32, sort=True)
        out_ddf1 = profile(resample_interval, ddf, windows, **kwargs,
                           _prefix='  ', _fun_name='resample_interval(dask, pandas)')
        compare_dataframes(out1, out_ddf1)

        out_ddf2 = profile(resample_interval, ddf, dd.from_pandas(windows.reset_index(drop=True), npartitions=8),
                           **kwargs, _prefix='  ', _fun_name='resample_interval(dask, dask)')
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
    profile(test_large, seed=s, _include_kwargs=['seed'])

    s = rng.randint(2 ** 31)
    profile(test_dask, seed=s, _include_kwargs=['seed'])

    print('Tests finished successfully!')
    print('    elapsed time:', pd.Timestamp.now() - tic)


if __name__ == '__main__':
    main()
