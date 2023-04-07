#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import numpy as np
import pandas as pd
import pytest

from catabra.util.longitudinal import group_temporal

from .util import create_random_data

# -- helpers -----------------------------------------------------------------------------------------------------------

seed = 38409


def _check_result(grp: pd.Series, df: pd.DataFrame, orig_index: pd.Index, orig_columns: pd.Index):
    assert df.shape == (len(orig_index), len(orig_columns))
    assert df.index.nlevels == orig_index.nlevels
    assert df.columns.nlevels == orig_columns.nlevels
    assert (df.index == orig_index).all()
    assert (df.columns == orig_columns).all()
    assert len(grp) == len(df)
    assert (grp.index == df.index).all()


# ----------------------------------------------------------------------------------------------------------------------


def test_no_group():
    df, _ = create_random_data(100000, 1, n_entities=1, intervals=True, seed=seed)
    orig_index = df.index.copy()
    orig_columns = df.columns.copy()
    grp = group_temporal(
        df,
        start_col='start',
        stop_col='stop',
        distance=(df['start'].max() - df['start'].min()) * 0.01
    )
    _check_result(grp, df, orig_index, orig_columns)


def test_all_columns():
    df, _ = create_random_data(100000, 1, n_entities=500, intervals=True, seed=seed)
    orig_index = df.index.copy()
    orig_columns = df.columns.copy()
    grp = group_temporal(
        df,
        group_by='entity',
        start_col='start',
        stop_col='stop',
        distance=(df['start'].max() - df['start'].min()) * 0.05
    )
    _check_result(grp, df, orig_index, orig_columns)


def test_mixed_index_columns():
    df, _ = create_random_data(100000, 1, n_entities=100, attributes=[f'attr_{i}' for i in range(10)],
                               intervals=True, seed=seed, time_dtype='float')
    df['value'].fillna(0, inplace=True)
    df.set_index(['value', 'entity'], inplace=True)
    orig_index = df.index.copy()
    orig_columns = df.columns.copy()
    grp = group_temporal(
        df,
        group_by=['attribute', 1],
        start_col='start',
        stop_col='stop',
        distance=(df['start'].max() - df['start'].min()) * 0.1
    )
    _check_result(grp, df, orig_index, orig_columns)


def test_isolated():
    df, _ = create_random_data(100000, 1, n_entities=100, attributes=[f'attr_{i}' for i in range(10)],
                               intervals=False, seed=seed, time_dtype='timestamp')
    df.set_index('entity', inplace=True)
    orig_index = df.index.copy()
    orig_columns = df.columns.copy()
    grp = group_temporal(
        df,
        group_by=[0, 'attribute'],
        time_col='timestamp',
        distance=(df['timestamp'].max() - df['timestamp'].min()) * 0.1
    )
    _check_result(grp, df, orig_index, orig_columns)


def test_integer():
    df, _ = create_random_data(100000, 1, n_entities=100, intervals=False, seed=seed, time_dtype='int')
    df.set_index('entity', inplace=True)
    orig_index = df.index.copy()
    orig_columns = df.columns.copy()
    grp = group_temporal(
        df,
        group_by=0,
        time_col='timestamp',
        distance=(df['timestamp'].max() - df['timestamp'].min()) * 0.005
    )
    _check_result(grp, df, orig_index, orig_columns)


def test_na():
    df, _ = create_random_data(100000, 1, n_entities=500, intervals=True, seed=seed, time_dtype='timestamp')
    rng = np.random.RandomState(seed)
    df.loc[rng.uniform(size=len(df)) < 0.1, 'start'] = None
    df.loc[rng.uniform(size=len(df)) < 0.1, 'stop'] = None
    orig_index = df.index.copy()
    orig_columns = df.columns.copy()
    grp = group_temporal(
        df,
        group_by='entity',
        start_col='start',
        stop_col='stop',
        distance=(df['stop'].max() - df['stop'].min()) * 0.1
    )

    _check_result(grp, df, orig_index, orig_columns)


def test_multi_columns():
    df, _ = create_random_data(100000, 1, n_entities=500, attributes=['attr_0', 'attr_1'], intervals=False, seed=seed)
    df.columns = pd.MultiIndex.from_tuples([(c, '' if c in ('entity', 'attribute', 'timestamp') else str(i))
                                            for i, c in enumerate(df.columns)])

    orig_index = df.index.copy()
    orig_columns = df.columns.copy()
    grp = group_temporal(
        df,
        group_by=[('entity', ''), ('attribute', '')],
        time_col=('timestamp', ''),
        distance=(df['timestamp'].max() - df['timestamp'].min()) * 0.1
    )
    _check_result(grp, df, orig_index, orig_columns)


@pytest.mark.slow
def test_large():
    df, _ = create_random_data(10000000, 1, n_entities=20000, attributes=['attr_' + str(i) for i in range(1, 50)],
                               intervals=True, time_dtype='timestamp', seed=seed)
    orig_index = df.index.copy()
    orig_columns = df.columns.copy()
    grp = group_temporal(
        df,
        group_by=['entity', 'attribute'],
        start_col='start',
        stop_col='stop',
        distance=(df['start'].max() - df['start'].min()) * 0.1
    )
    _check_result(grp, df, orig_index, orig_columns)