#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import numpy as np
import pandas as pd
import pytest

from catabra.util.longitudinal import prev_next_values

from .util import create_random_data

# -- helpers -----------------------------------------------------------------------------------------------------------


def _check_result_partial(df: pd.DataFrame, orig_index, orig_columns):
    assert df.shape == (len(orig_index), len(orig_columns))
    assert df.index.nlevels == orig_index.nlevels
    assert df.columns.nlevels == orig_columns.nlevels

# ----------------------------------------------------------------------------------------------------------------------


def test_no_group(seed=None):
    df, _ = create_random_data(100000, 1, n_entities=1, seed=seed)
    orig_index = df.index.copy()
    orig_columns = df.columns.copy()
    out = prev_next_values(
        df,
        sort_by='timestamp',
        columns={
            'value': dict(prev_name='value_prev', next_name='value_next')
        },
        first_indicator_name='is_first',
        inplace=False,
        keep_sorted=False
    )
    _check_result_partial(df, orig_index, orig_columns)
    assert (df.index == orig_index).all()
    assert (df.columns == orig_columns).all()
    assert len(out) == len(df)
    assert (out.index == df.index).all()
    assert out.loc[out['is_first'], 'value_prev'].isna().all()


def test_all_columns(seed=None):
    df, _ = create_random_data(100000, 1, n_entities=500, attributes=['attr_1', 'attr_2', 'attr_3'], seed=seed,
                               time_dtype='timestamp')

    orig_index = df.index.copy()
    orig_columns = df.columns.copy()
    out = prev_next_values(
        df,
        sort_by='timestamp',
        group_by=['entity', 'attribute'],
        columns={
            'value': dict(prev_name='value_prev'),
            'timestamp': dict(prev_name='timestamp_prev')
        },
        first_indicator_name='is_first',
        inplace=False,
        keep_sorted=True
    )

    _check_result_partial(df, orig_index, orig_columns)
    assert (df.index == orig_index).all()
    assert (df.columns == orig_columns).all()
    assert len(out) == len(df)
    assert out.loc[out['is_first'], ['value_prev', 'timestamp_prev']].isna().all().all()
    assert (out.index == df.sort_values(['entity', 'attribute', 'timestamp']).index).all()


def test_mixed_index_columns(seed=None):
    df, _ = create_random_data(100000, 1, n_entities=100, attributes=[f'attr_{i}' for i in range(10)],
                               seed=seed, time_dtype='float')
    df['value'].fillna(0, inplace=True)
    df.set_index(['entity', 'timestamp'], inplace=True)
    orig_index = df.index.copy()
    orig_columns = df.columns.copy()
    out = prev_next_values(
        df,
        sort_by=1,
        group_by=['attribute', 0],
        columns={
            'value': dict(next_name='value_next')
        },
        last_indicator_name='is_last',
        inplace=True,
        keep_sorted=False
    )
    assert out is df
    assert df.shape == (len(orig_index), len(orig_columns) + 2)
    assert df.index.nlevels == orig_index.nlevels
    assert df.columns.nlevels == orig_columns.nlevels
    assert (df.index == orig_index).all()
    assert df.loc[df['is_last'], 'value_next'].isna().all()


def test_integer(seed=None):
    df, _ = create_random_data(100000, 1, n_entities=100, seed=seed, value_dtype='int', time_dtype='int')
    df.set_index('entity', inplace=True)
    orig_index = df.index.copy()
    orig_columns = df.columns.copy()
    out = prev_next_values(
        df,
        sort_by='timestamp',
        group_by=0,
        columns={
            'value': dict(prev_name='value_prev', prev_fill=0)
        },
        first_indicator_name='is_first',
        last_indicator_name='is_last',
        inplace=True,
        keep_sorted=True
    )
    assert out is df
    assert len(df) == len(orig_index)
    assert df.index.nlevels == orig_index.nlevels
    assert df.columns.nlevels == orig_columns.nlevels
    assert df['value_prev'].dtype.kind == 'i'
    assert (df.loc[df['is_first'], 'value_prev'] == 0).all()


def test_na(seed=None):
    df, _ = create_random_data(100000, 1, n_entities=500, intervals=True, seed=seed, time_dtype='timestamp')
    rng = np.random.RandomState(seed)
    df.loc[rng.uniform(size=len(df)) < 0.2, 'timestamp'] = None
    orig_index = df.index.copy()
    orig_columns = df.columns.copy()
    out = prev_next_values(
        df,
        sort_by='timestamp',
        group_by='entity',
        columns={
            'timestamp': dict(next_name='timestamp_next')
        }
    )
    _check_result_partial(df, orig_index, orig_columns)
    assert (df.index == orig_index).all()
    assert (df.columns == orig_columns).all()
    assert len(out) == len(df)
    assert (out.index == df.index).all()


def test_multi_columns(seed=None):
    df, _ = create_random_data(100000, 1, n_entities=500, attributes=['attr_0', 'attr_1'], seed=seed)
    df.columns = pd.MultiIndex.from_product([df.columns, ['']])
    orig_index = df.index.copy()
    orig_columns = df.columns.copy()
    out = prev_next_values(
        df,
        sort_by=('timestamp', ''),
        group_by=[('entity', ''), ('attribute', '')],
        columns={
            ('value', ''): dict(prev_name='prev', next_name='next')
        }
    )
    _check_result_partial(df, orig_index, orig_columns)
    assert (df.index == orig_index).all()
    assert (df.columns == orig_columns).all()
    assert len(out) == len(df)
    assert (out.index == df.index).all()
    assert out.columns.nlevels == 1


@pytest.mark.slow
def test_large(seed=None):
    df, _ = create_random_data(10000000, 1, n_entities=20000, attributes=['attr_' + str(i) for i in range(1, 50)],
                               time_dtype='timestamp', seed=seed)
    df.set_index('timestamp', inplace=True)
    orig_index = df.index.copy()
    orig_columns = df.columns.copy()
    out = prev_next_values(
        df,
        sort_by=0,
        group_by=['entity', 'attribute'],
        columns={
            'value': dict(prev_name='value_prev'),
            'entity': dict(next_name='entity_next', next_fill=-1)
        }
    )
    _check_result_partial(df, orig_index, orig_columns)
    assert (df.index == orig_index).all()
    assert (df.columns == orig_columns).all()
    assert len(out) == len(df)
    assert (out.index == df.index).all()
