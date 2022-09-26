import pandas as pd
import numpy as np

from ..util import profile
from .util import create_random_data
from catabra.util.time_series import group_temporal


def _check_result(grp: pd.Series, df: pd.DataFrame, orig_index: pd.Index, orig_columns: pd.Index):
    assert df.shape == (len(orig_index), len(orig_columns))
    assert df.index.nlevels == orig_index.nlevels
    assert df.columns.nlevels == orig_columns.nlevels
    assert (df.index == orig_index).all()
    assert (df.columns == orig_columns).all()
    assert len(grp) == len(df)
    assert (grp.index == df.index).all()


def test_no_group(seed=None):
    df, _ = create_random_data(100000, 1, n_entities=1, intervals=True, seed=seed)

    kwargs = dict(
        start_col='start',
        stop_col='stop',
        distance=(df['start'].max() - df['start'].min()) * 0.01
    )

    orig_index = df.index.copy()
    orig_columns = df.columns.copy()
    grp = profile(group_temporal, df, **kwargs, _prefix='  ')
    _check_result(grp, df, orig_index, orig_columns)


def test_all_columns(seed=None):
    df, _ = create_random_data(100000, 1, n_entities=500, intervals=True, seed=seed)

    kwargs = dict(
        group_by='entity',
        start_col='start',
        stop_col='stop',
        distance=(df['start'].max() - df['start'].min()) * 0.05
    )

    orig_index = df.index.copy()
    orig_columns = df.columns.copy()
    grp = profile(group_temporal, df, **kwargs, _prefix='  ')
    _check_result(grp, df, orig_index, orig_columns)


def test_mixed_index_columns(seed=None):
    df, _ = create_random_data(100000, 1, n_entities=100, attributes=[f'attr_{i}' for i in range(10)],
                               intervals=True, seed=seed, time_dtype='float')
    df['value'].fillna(0, inplace=True)
    df.set_index(['value', 'entity'], inplace=True)

    kwargs = dict(
        group_by=['attribute', 1],
        start_col='start',
        stop_col='stop',
        distance=(df['start'].max() - df['start'].min()) * 0.1
    )

    orig_index = df.index.copy()
    orig_columns = df.columns.copy()
    grp = profile(group_temporal, df, **kwargs, _prefix='  ')
    _check_result(grp, df, orig_index, orig_columns)


def test_isolated(seed=None):
    df, _ = create_random_data(100000, 1, n_entities=100, attributes=[f'attr_{i}' for i in range(10)],
                               intervals=False, seed=seed, time_dtype='timestamp')
    df.set_index('entity', inplace=True)

    kwargs = dict(
        group_by=[0, 'attribute'],
        time_col='timestamp',
        distance=(df['timestamp'].max() - df['timestamp'].min()) * 0.1
    )

    orig_index = df.index.copy()
    orig_columns = df.columns.copy()
    grp = profile(group_temporal, df, **kwargs, _prefix='  ')
    _check_result(grp, df, orig_index, orig_columns)


def test_integer(seed=None):
    df, _ = create_random_data(100000, 1, n_entities=100, intervals=False, seed=seed, time_dtype='int')
    df.set_index('entity', inplace=True)

    kwargs = dict(
        group_by=0,
        time_col='timestamp',
        distance=(df['timestamp'].max() - df['timestamp'].min()) * 0.005
    )

    orig_index = df.index.copy()
    orig_columns = df.columns.copy()
    grp = profile(group_temporal, df, **kwargs, _prefix='  ')
    _check_result(grp, df, orig_index, orig_columns)


def test_na(seed=None):
    df, _ = create_random_data(100000, 1, n_entities=500, intervals=True, seed=seed, time_dtype='timestamp')
    rng = np.random.RandomState(seed)
    df.loc[rng.uniform(size=len(df)) < 0.1, 'start'] = None
    df.loc[rng.uniform(size=len(df)) < 0.1, 'stop'] = None

    kwargs = dict(
        group_by='entity',
        start_col='start',
        stop_col='stop',
        distance=(df['stop'].max() - df['stop'].min()) * 0.1
    )

    orig_index = df.index.copy()
    orig_columns = df.columns.copy()
    grp = profile(group_temporal, df, **kwargs, _prefix='  ')
    _check_result(grp, df, orig_index, orig_columns)


def test_multi_columns(seed=None):
    df, _ = create_random_data(100000, 1, n_entities=500, attributes=['attr_0', 'attr_1'], intervals=False, seed=seed)
    df.columns = pd.MultiIndex.from_tuples([(c, '' if c in ('entity', 'attribute', 'timestamp') else str(i))
                                            for i, c in enumerate(df.columns)])

    kwargs = dict(
        group_by=[('entity', ''), ('attribute', '')],
        time_col=('timestamp', ''),
        distance=(df['timestamp'].max() - df['timestamp'].min()) * 0.1
    )

    orig_index = df.index.copy()
    orig_columns = df.columns.copy()
    grp = profile(group_temporal, df, **kwargs, _prefix='  ')
    _check_result(grp, df, orig_index, orig_columns)


def test_large(seed=None):
    df, _ = create_random_data(10000000, 1, n_entities=20000, attributes=['attr_' + str(i) for i in range(1, 50)],
                               intervals=True, time_dtype='timestamp', seed=seed)

    kwargs = dict(
        group_by=['entity', 'attribute'],
        start_col='start',
        stop_col='stop',
        distance=(df['start'].max() - df['start'].min()) * 0.1
    )

    orig_index = df.index.copy()
    orig_columns = df.columns.copy()
    grp = profile(group_temporal, df, **kwargs, _prefix='  ')
    _check_result(grp, df, orig_index, orig_columns)


def main(seed=None):
    # The following tests only check whether `group_temporal()` can be applied to the given arguments without error.
    # Only rudimentary correctness checks of the results are performed.

    tic = pd.Timestamp.now()
    rng = np.random.RandomState(seed=seed)

    s = rng.randint(2 ** 31)
    profile(test_no_group, seed=s, _include_kwargs=['seed'])

    s = rng.randint(2 ** 31)
    profile(test_all_columns, seed=s, _include_kwargs=['seed'])

    s = rng.randint(2 ** 31)
    profile(test_mixed_index_columns, seed=s, _include_kwargs=['seed'])

    s = rng.randint(2 ** 31)
    profile(test_isolated, seed=s, _include_kwargs=['seed'])

    s = rng.randint(2 ** 31)
    profile(test_integer, seed=s, _include_kwargs=['seed'])

    s = rng.randint(2 ** 31)
    profile(test_na, seed=s, _include_kwargs=['seed'])

    s = rng.randint(2 ** 31)
    profile(test_multi_columns, seed=s, _include_kwargs=['seed'])

    s = rng.randint(2 ** 31)
    profile(test_large, seed=s, _include_kwargs=['seed'])

    print('Tests finished successfully!')
    print('    elapsed time:', pd.Timestamp.now() - tic)


if __name__ == '__main__':
    main()
