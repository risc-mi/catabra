#  Copyright (c) 2023. RISC Software GmbH.
#  All rights reserved.

import numpy as np
import pandas as pd
import pytest

from catabra.util.longitudinal import make_windows, resample_eav, resample_interval

_df = pd.DataFrame(
    data=dict(
        entity=[0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 4],
        start=pd.Timestamp.now() + pd.to_timedelta([0, 5, 47, 369, 12, 124, 1, 1, 3, 201, 30], unit='m'),
        stop=pd.Timestamp.now() + pd.to_timedelta([5, 42, 324, 369, 130, 130, 2, 3, 9, 210, 98], unit='m'),
        value=np.random.uniform(2.5, 25, size=11)
    )
)


@pytest.mark.parametrize(
    ids=[
        'None',
        'start',
        'stop'
    ],
    argnames=[
        'anchor'
    ],
    argvalues=[
        (None,),
        ('start',),
        ('stop',)
    ]
)
def test_no_df(anchor):
    mw = make_windows(start_rel=pd.Timedelta('-1 hour'), duration=pd.Timedelta('10 minutes'), anchor=anchor)
    windows = mw(df=_df, entity_col='entity', time_col='start')

    if anchor is None:
        anchor = 'start'

    assert len(windows) == len(_df)
    assert (windows.index == _df.index).all()
    assert (windows[('entity', '')] == _df['entity']).all()
    assert (windows[('start', 'start')] == _df[anchor] - pd.Timedelta('1 hour')).all()
    assert (windows[('start', 'stop')] == _df[anchor] - pd.Timedelta('50 minutes')).all()


def test_groupby():
    start = pd.Timestamp.now()
    stop = start + pd.Timedelta('2 hours')
    mw = make_windows(
        df='df.groupby(entity_col)[time_col].min().to_frame()',
        start=start,
        stop=stop
    )
    windows = mw(df=_df, entity_col='entity', time_col='start')

    df_aux = _df.groupby('entity')['start'].min()

    assert len(windows) == len(df_aux)
    assert (windows.index == df_aux.index).all()
    assert (windows[('entity', '')] == df_aux.index).all()
    assert (windows[('start', 'start')] == start).all()
    assert (windows[('start', 'stop')] == stop).all()


def test_df():
    df = pd.DataFrame(
        data=dict(
            subject_id=[0, 0, 1, 1, 2, 2, 3, 4],
            timestamp=pd.Timestamp.now() + pd.to_timedelta([200, 400, 15, 150, 78, 87, 0, 40], unit='m')
        ),
        index=[5, 0, 3, 7, 1, 4, 6, 2]
    )
    start_rel = pd.Series(pd.to_timedelta([-167, -206, 7, -13, 0, -9, 2, 39], unit='m'))

    mw = make_windows(
        df=df,
        entity='subject_id',
        start_rel=start_rel,
        anchor='timestamp'
    )
    windows = mw(df=_df, entity_col='entity', start_col='start', stop_col='stop')

    assert len(windows) == len(df)
    assert (windows.index == df.index).all()
    assert (windows['entity'] == df['subject_id']).all()
    assert (windows['start'] == df['timestamp'] + start_rel).all()
    assert 'stop' not in windows.columns


def test_resample_eav():
    res = resample_eav(
        _df,
        make_windows(start_rel=pd.Timedelta('-15 minutes')),
        agg={'attr': 'mean'},
        entity_col='entity',
        time_col='start',
        value_col='value'
    )

    assert len(res) == len(_df)


def test_resample_interval():
    res = resample_interval(
        _df,
        make_windows(
            df='df.groupby(entity_col)[stop_col].max().to_frame("anchor")',
            stop_rel=pd.Timedelta(0),
            duration=pd.Timedelta('30 minutes'),
            anchor='anchor'
        ),
        entity_col='entity',
        value_col='value',
        start_col='start',
        stop_col='stop'
    )

    assert len(res) == _df['entity'].nunique()


def test_invalid_args():
    mw = make_windows(start='start', start_rel=pd.Timedelta(0))
    try:
        mw(df=_df, entity_col='entity', time_col='start')
    except ValueError:
        pass
    else:
        assert False, 'both start and start_rel given'

    mw = make_windows(entity='entity', anchor='start', duration=pd.Timedelta('10 minutes'))
    try:
        mw(df=_df, entity_col='entity', time_col='start')
    except ValueError:
        pass
    else:
        assert False, 'no endpoints given'

    mw = make_windows(start='start', stop='stop', duration=pd.Timedelta('10 minutes'))
    try:
        mw(df=_df, entity_col='entity', time_col='start')
    except ValueError:
        pass
    else:
        assert False, 'both endpoints and duration given'

    mw = make_windows(start_rel=pd.Timedelta(0))
    try:
        mw(df=_df, entity_col='entity', start_col='start', stop_col='stop')
    except ValueError:
        pass
    else:
        assert False, 'no anchor given'

    mw = make_windows(start=pd.to_datetime([58963, 47260], unit='h'))
    try:
        mw(df=_df, entity_col='entity', time_col='start')
    except ValueError:
        pass
    else:
        assert False, 'start is Index'

    mw = make_windows(entity=[0, 2, 1, 0, 0, 3])
    try:
        mw(df=_df, entity_col='entity', time_col='start')
    except ValueError:
        pass
    else:
        assert False, 'entity is list'

    mw = make_windows(entity=pd.Series([1, 2, 3], index=[0, 0, 1]),
                      start=pd.Series(pd.Timestamp.now(), index=[1, 0, 0]))
    try:
        mw(df=_df, entity_col='entity', time_col='start')
    except ValueError:
        pass
    else:
        assert False, 'non-unique row index'

    mw = make_windows(entity=pd.Series([1, 2, 3], index=[0, 1, 2]),
                      start=pd.Series(pd.Timestamp.now(), index=[1, 2, 3]))
    try:
        mw(df=_df, entity_col='entity', time_col='start')
    except ValueError:
        pass
    else:
        assert False, 'non-matching row index'

    mw = make_windows(entity=0, start=pd.Timestamp.now())
    try:
        mw(df=_df, entity_col='entity', time_col='start')
    except ValueError:
        pass
    else:
        assert False, 'unknown row index'
