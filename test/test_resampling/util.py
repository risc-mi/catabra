from typing import Optional, Tuple
import pandas as pd
import numpy as np
import re


def resample_eav_slow(df: pd.DataFrame, windows: pd.DataFrame, agg: dict = None, entity_col=None, time_col=None,
                      attribute_col=None, value_col=None, include_start: bool = True, include_stop: bool = False) \
        -> pd.DataFrame:
    df = df.sort_values(time_col)
    if entity_col is None:
        df_entities = None
    elif entity_col in df.columns:
        df_entities = df[entity_col].values
    elif entity_col == df.index.name:
        df_entities = df.index
    else:
        assert False

    if entity_col is None:
        windows_entities = None
    elif (entity_col, '') in windows.columns:
        windows_entities = windows[(entity_col, '')].values
    elif entity_col == windows.index.name:
        windows_entities = windows.index
    else:
        assert False
    if (time_col, 'start') in windows.columns:
        windows_start = windows[(time_col, 'start')].values
    else:
        windows_start = None
    if (time_col, 'stop') in windows.columns:
        windows_stop = windows[(time_col, 'stop')].values
    else:
        windows_stop = None

    _PERCENTILE = r'p[0-9]+(\.[0-9]+)?'
    _RANK = r'[rt]-?[0-9]+'

    new_data = {}
    for i in range(len(windows)):
        if entity_col is None:
            mask = np.ones(len(df), dtype=np.bool)
        else:
            mask = df_entities == windows_entities[i]
        if windows_start is not None:
            if include_start:
                mask &= df[time_col] >= windows_start[i]
            else:
                mask &= df[time_col] > windows_start[i]
        if windows_stop is not None:
            if include_stop:
                mask &= df[time_col] <= windows_stop[i]
            else:
                mask &= df[time_col] < windows_stop[i]
        df0 = df[mask]
        for attr, func in agg.items():
            if not isinstance(func, (list, tuple)):
                func = [func]
            if attribute_col is None:
                df1 = df0
            else:
                df1 = df0[df0[attribute_col] == attr]
            for fn in func:
                val = None
                agg_name = fn
                if df1.empty:
                    if fn in ('count', 'size', 'nunique', 'mode_count'):
                        val = 0.
                elif fn == 'mode':
                    val = df1[value_col].value_counts().index[0]
                elif fn == 'mode_count':
                    val = df1[value_col].value_counts().iloc[0]
                elif fn == 'size':
                    val = float(len(df1))
                elif fn == 'count':
                    val = float(df1[value_col].count())
                elif fn == 'nunique':
                    val = float(df1[value_col].nunique())
                elif re.fullmatch(_PERCENTILE, fn):
                    p = float(fn[1:])
                    if 0 <= p <= 100:
                        val = df1[value_col].quantile(p / 100)
                    else:
                        agg_name = None
                elif re.fullmatch(_RANK, fn):
                    r = int(fn[1:])
                    agg_name = fn[0] + str(r)
                    if r < 0:
                        r += len(df1)
                    if 0 <= r < len(df1):
                        if fn[0] == 'r':
                            val = df1[value_col].iloc[r]
                        else:
                            val = df1[time_col].iloc[r]
                else:
                    val = getattr(df1[value_col], fn)()

                if agg_name is not None:
                    new_data.setdefault((attr, agg_name), []).append(val)

    out = pd.DataFrame(index=windows.index, data=new_data)
    for c in out.columns:
        if out[c].dtype.kind in 'iu':
            out[c] = out[c].astype(np.float64)      # output data type cannot be int
    return pd.concat([windows, out], axis=1, sort=False)


def resample_interval_slow(df: pd.DataFrame, windows: pd.DataFrame, attributes: list = None, entity_col=None,
                           start_col=None, stop_col=None, attribute_col=None, value_col=None, time_col=None,
                           epsilon=1e-7) -> pd.DataFrame:
    if entity_col is None:
        df_entities = None
    elif entity_col in df.columns:
        df_entities = df[entity_col].values
    elif entity_col == df.index.name:
        df_entities = df.index
    else:
        assert False
    if attribute_col in df.columns:
        if attributes is None:
            attributes = df[attribute_col].unique()
    else:
        if attributes is None:
            attributes = ['sum']
        else:
            assert len(attributes) == 1
    df_na = df[value_col].isna().values
    if start_col in df.columns and stop_col in df.columns:
        df_dur = (df[stop_col] - df[start_col]).values
        df_nonzero_dur = (df[stop_col] > df[start_col]).values
        df_na |= df[start_col].isna() | df[stop_col].isna()
        try:
            df_inf = np.isposinf(df_dur)
        except:     # noqa
            df_inf = np.zeros(len(df), dtype=np.bool)
    else:
        df_dur = None
        df_nonzero_dur = None
        df_inf = np.ones(len(df), dtype=np.bool)
        if start_col in df.columns:
            df_na |= df[start_col].isna()
        else:
            df_na |= df[stop_col].isna()

    if entity_col is None:
        windows_entities = None
    elif (entity_col, '') in windows.columns:
        windows_entities = windows[(entity_col, '')].values
    elif entity_col in windows.columns:
        windows_entities = windows[entity_col].values
    elif entity_col == windows.index.name:
        windows_entities = windows.index
    else:
        assert False
    if (time_col, 'start') in windows.columns:
        windows_start = windows[(time_col, 'start')].values
    elif start_col in windows.columns:
        windows_start = windows[start_col].values
    else:
        windows_start = None
    if (time_col, 'stop') in windows.columns:
        windows_stop = windows[(time_col, 'stop')].values
    elif stop_col in windows.columns:
        windows_stop = windows[stop_col].values
    else:
        windows_stop = None

    out = pd.DataFrame(index=windows.index, columns=attributes, data=0, dtype=np.float64)
    for i in range(len(windows)):
        if windows_start is not None and pd.isna(windows_start[i]):
            continue
        if windows_stop is not None and pd.isna(windows_stop[i]):
            continue

        if entity_col is None:
            entity_mask = np.ones(len(df), dtype=np.bool)
        else:
            entity_mask = df_entities == windows_entities[i]
        for a in attributes:
            if attribute_col in df.columns:
                mask = entity_mask & (df[attribute_col] == a)
            else:
                mask = entity_mask
            df0 = df[mask]

            if windows_start is None:
                if start_col in df0.columns:
                    inter_start = df0[start_col].values
                else:
                    inter_start = None
            else:
                if start_col in df0.columns:
                    inter_start = np.maximum(df0[start_col].values, windows_start[i])
                else:
                    inter_start = windows_start[i]

            if windows_stop is None:
                if stop_col in df0.columns:
                    inter_stop = df0[stop_col].values
                else:
                    inter_stop = None
            else:
                if stop_col in df0.columns:
                    inter_stop = np.minimum(df0[stop_col].values, windows_stop[i])
                else:
                    inter_stop = windows_stop[i]

            vs = df0[value_col].values.astype(out[a].dtype, copy=True)
            if inter_start is None or inter_stop is None:
                # all intersections are infinite
                pass
            else:
                inter_dur = inter_stop - inter_start
                try:
                    inter_inf = np.isposinf(inter_dur)
                except:     # noqa
                    inter_inf = np.zeros(len(df0), dtype=np.bool)
                if df_dur is None:
                    # all observation intervals are infinite
                    vs[~inter_inf] = epsilon * np.sign(vs[~inter_inf])
                else:
                    df0_dur = df_dur[mask]
                    df0_nonzero_dur = df_nonzero_dur[mask]
                    df0_inf = df_inf[mask]
                    vs[df0_nonzero_dur] *= (inter_dur[df0_nonzero_dur] / df0_dur[df0_nonzero_dur]).astype(np.float32)
                    vs[df0_inf & ~inter_inf] = epsilon * np.sign(vs[df0_inf & ~inter_inf])

                vs[inter_stop < inter_start] = 0

            vs[df_na[mask]] = 0
            out[a].values[i] = vs.sum()

    if windows.columns.nlevels == 2:
        out.columns = pd.MultiIndex.from_product([out.columns, ['']])
    return pd.concat([windows, out], axis=1, sort=False)


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, eps: float = 1e-7):
    assert df1.shape == df2.shape, str(df1.shape) + ' vs. ' + str(df2.shape)
    assert (df1.index == df2.index).all()
    assert (df1.columns == df2.columns).all()
    for c in df1.columns:
        assert df1[c].dtype.kind == df2[c].dtype.kind, c
        s1 = df1[c]
        s2 = df2[c]
        if df1[c].dtype.kind == 'M':
            s1 = s1 - pd.Timestamp(0)
            s2 = s2 - pd.Timestamp(0)
        if s1.dtype.kind == 'm':
            s1 = s1.dt.total_seconds()
            s2 = s2.dt.total_seconds()

        if s1.dtype.kind == 'f':
            assert (((s1 - s2).abs() < eps) | (s1.isna() & s2.isna())).all(), c
        else:
            assert ((s1 == s2) | (s1.isna() & s2.isna())).all(), c


def create_random_data(n_observations: int, n_windows: int, n_entities: Optional[int] = None,
                       attributes: Optional[list] = None, time_dtype: str = 'timedelta', value_dtype: str = 'float',
                       intervals: bool = False, window_pattern: str = 'random', seed: Optional[int] = None) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed=seed)
    values = _create_random_series(n_observations, value_dtype, rng)
    times = _create_random_series(n_observations, time_dtype, rng)
    min_ts = times.min()
    max_ts = times.max()
    if value_dtype in ('float', 'timedelta', 'timestamp', 'category', 'str'):
        perm = rng.permutation(n_observations)
        values.iloc[perm[:int(np.power(10., rng.uniform(-1, np.log10(n_observations / 2))))]] = None
    if intervals:
        duration = np.power(20., -rng.uniform(0, 1, size=n_observations))
        duration[rng.choice([False, True], size=n_observations, replace=True, p=[0.67, 0.33])] = 0.
        duration = duration * (max_ts - min_ts)
        df = pd.DataFrame(data=dict(value=values, start=times - duration, stop=times + duration))
    else:
        df = pd.DataFrame(data=dict(value=values, timestamp=times))
    if n_entities is None:
        n_entities = max(2, int(np.round(np.power(10., rng.uniform(0, np.log10(n_windows))))))
    if n_entities > 1:
        df['entity'] = rng.randint(n_entities, size=n_observations)
    if attributes is not None:
        p = rng.uniform(0.1, 1, size=len(attributes))
        p /= p.sum()
        df['attribute'] = rng.choice(attributes, size=n_observations, p=p, replace=True)

    duration = np.power(10., -rng.uniform(0, 1, size=n_windows)) * (max_ts - min_ts)
    start = min_ts + rng.uniform(-0.2, 1, size=n_windows) * (max_ts - min_ts)
    windows = pd.DataFrame(data=dict(start=start, stop=start + duration), index=rng.permutation(n_windows))
    windows.columns = pd.MultiIndex.from_product([['timestamp'], ['start', 'stop']])
    if n_entities > 1:
        windows[('entity', '')] = rng.randint(1, df['entity'].max() + 2, size=n_windows)
        windows[('entity', '')].values[-1] = 0  # entity 0 has only one window

    if window_pattern == 'non-overlapping':
        def _make_non_overlapping(grp: pd.DataFrame) -> pd.DataFrame:
            w = rng.uniform(0.1, 1, size=len(grp))
            w /= w.sum()
            grp[('timestamp', 'start')] = min_ts + np.roll(np.cumsum(w), 1) * (max_ts - min_ts)
            grp[('timestamp', 'start')].values[0] = min_ts - w[0] * (max_ts - min_ts)   # window without observations
            grp[('timestamp', 'stop')] = \
                grp[('timestamp', 'start')] + (rng.uniform(0.75, 1, size=len(w)) * w) * (max_ts - min_ts)
            return grp

        if n_entities > 1:
            windows = windows.groupby([('entity', '')]).apply(_make_non_overlapping)
        else:
            windows = _make_non_overlapping(windows)
    elif window_pattern != 'random':
        def _make_regular(grp: pd.DataFrame, offset_range=(0.5, 0.9)) -> pd.DataFrame:
            d = (max_ts - min_ts) / len(grp)
            o = rng.uniform(*offset_range) * d
            grp[('timestamp', 'start')] = min_ts + rng.uniform(-0.2, 0.8) * (max_ts - min_ts) + o * np.arange(len(grp))
            grp[('timestamp', 'stop')] = grp[('timestamp', 'start')] + d
            return grp

        if window_pattern == 'regular':
            o_rng = (0.5, 0.9)
        else:
            assert window_pattern == 'regular non-overlapping'
            o_rng = (1.1, 1.5)
        if n_entities > 1:
            windows = windows.groupby([('entity', '')]).apply(_make_regular, offset_range=o_rng)
        else:
            _make_regular(windows, offset_range=o_rng)

    return df, windows


def _create_random_series(n: int, dtype: str, rng: np.random.RandomState) -> pd.Series:
    if dtype == 'float':
        distr = rng.choice(['normal', 'exponential', 'uniform'])
        if distr == 'normal':
            return pd.Series(data=rng.normal(loc=rng.uniform(-100, 100), scale=rng.uniform(0.5, 20), size=n))
        elif distr == 'exponential':
            return pd.Series(data=rng.exponential(scale=rng.uniform(0.5, 20), size=n))
        elif distr == 'uniform':
            return pd.Series(data=rng.uniform(-1, 1, size=n))
    elif dtype == 'int':
        distr = rng.choice(['uniform', 'binomial'])
        if distr == 'uniform':
            return pd.Series(data=rng.randint(-rng.randint(0, 10), rng.randint(2, 10), size=n))
        elif distr == 'binomial':
            return pd.Series(data=rng.binomial(rng.randint(2, 20), rng.uniform(0.1, 0.9), size=n))
    elif dtype == 'bool':
        p = rng.uniform(0.1, 0.9)
        return pd.Series(data=rng.choice([False, True], size=n, p=[p, 1 - p], replace=True))
    elif dtype == 'timedelta':
        return pd.to_timedelta(_create_random_series(n, 'float', rng),
                               unit=str(rng.choice(['seconds', 'hours', 'days'])))
    elif dtype == 'timestamp':
        return _create_random_series(n, 'timedelta', rng) + pd.Timestamp.now()
    elif dtype == 'str':
        m = rng.randint(2, 20)
        p = rng.uniform(0.1, 1, size=m)
        p /= p.sum()
        return pd.Series(data=rng.choice(['str_' + str(i) for i in range(m)], size=n, p=p, replace=True))
    elif dtype == 'category':
        m = rng.randint(2, 20)
        p = rng.uniform(0.1, 1, size=m)
        p /= p.sum()
        return pd.Series(
            data=pd.Categorical.from_codes(rng.choice(m, size=n, p=p, replace=True),
                                           categories=['cat_' + str(i) for i in range(m)], ordered=True)
        )
    else:
        assert False
