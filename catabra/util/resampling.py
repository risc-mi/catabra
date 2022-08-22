from typing import Union
import numpy as np
import pandas as pd


def resample_eav(df: pd.DataFrame, windows: pd.DataFrame, agg: dict = None, entity_col=None, time_col=None,
                 attribute_col=None, value_col=None, window_group_col=None, include_start: bool = True,
                 include_stop: bool = False) -> pd.DataFrame:
    """
    Resample data in EAV (entity-attribute-value) format wrt. explicitly passed windows of arbitrary (possibly
    infinite) length.

    :param df: The DataFrame to resample, in EAV format. That means, must have columns `value_col` (contains observed
    values), `time_col` (contains observation times), `attribute_col` (optional; contains attribute identifiers) and
    `entity_col` (optional; contains entity identifiers). Must have one column index level.
    Data types are arbitrary, as long as observation times and entity identifiers can be compared wrt. `<` and `<=`
    (e.g., float, int, time delta, date time). Entity identifiers must not be NA.

    :param windows: The target windows into which `df` is resampled. Must have two column index levels and columns
    `(time_col, "start")` (optional; contains start times of each window), `(time_col, "stop")` (optional; contains end
    times of each window), `(entity_col, "")` (optional; contains entity identifiers) and `(window_group_col, "")`
    (optional; contains information for creating groups of mutually disjoint windows).
    At least one of the two endpoint-columns must be given; if one is missing it is assumed to represent +/- inf.

    :param agg: The aggregations to apply. Must be a dict mapping attribute identifiers to lists of aggregation
    functions, which are applied to all observed values of the respective attribute in each specified window.
    Supported aggregation functions are:
        * "mean": Empirical mean of observed non-NA values
        * "min": Minimum of observed non-NA values; equivalent to "p0"
        * "max": Maximum of observed non-NA values; equivalent to "p100"
        * "median": Median of observed non-NA values; equivalent to "p50"
        * "std": Empirical standard deviation of observed non-NA values
        * "var": Empirical variance of observed non-NA values
        * "sum": Sum of observed non-NA values
        * "size": Number of observations, including NA values
        * "count": Number of non-NA observations
        * "nunique": Number of unique observed non-NA values
        * "mode": Mode of observed non-NA values, i.e., most frequent value; ties are broken randomly but reproducibly
        * "mode_count": Number of occurrences of mode
        * "pxx": Percentile of observed non-NA values; `xx` is an arbitrary float in the interval [0, 100]
        * "rxx": `xx`-th observed value (possibly NA), starting from 0; negative indices count from the end
        * "txx": Time of `xx`-th observed value; negative indices count from the end

    :param entity_col: Name of the column in `df` and `windows` containing entity identifiers. If None, all entries
    are assumed to belong to the same entity. Note that entity identifiers may also be on the row index.

    :param time_col: Name of the column in `df` containing observation times, and also name of column(s) in `windows`
    containing start- and end times of the windows. Note that despite its name the data type of the column is
    arbitrary, as long as its values can be compared wrt. `<` and `<=`.

    :param attribute_col: Name of the column in `df` containing attribute identifiers. If None, all entries are assumed
    to belong to the same attribute; in that case `agg` may only contain one single item.

    :param value_col: Name of the column in `df` containing the observed values.

    :param window_group_col: Name of the column in `windows` containing information for grouping the windows such that
    all windows in a group are mutually disjoint. If None, this information is computed automatically; otherwise, must
    have integral data type and non-negative values. In any case, this column is not present in the result!

    :param include_start: Whether start times of observation windows are part of the windows.

    :param include_stop: Whether end times of observation windows are part of the windows.

    :return: Resampled data. Like `windows`, but with one additional column for each requested aggregation.
    Order of columns is arbitrary, order of rows is exactly as in `windows`.
    """

    assert df.columns.nlevels == 1
    assert time_col in df.columns
    assert value_col in df.columns
    assert attribute_col is None or attribute_col in df.columns
    assert entity_col is None or entity_col in df.columns or entity_col == df.index.name

    tmp_col = '__tmp__'
    assert windows.columns.nlevels == 2
    assert tmp_col not in windows.columns.get_level_values(0)
    assert (time_col, 'start') in windows.columns or (time_col, 'stop') in windows.columns
    assert (time_col, 'start') not in windows.columns or \
           (windows[(time_col, 'start')].notna().all() and windows[(time_col, 'start')].dtype == df[time_col].dtype)
    assert (time_col, 'stop') not in windows.columns or \
           (windows[(time_col, 'stop')].notna().all() and windows[(time_col, 'stop')].dtype == df[time_col].dtype)
    assert entity_col is None or (entity_col, '') in windows.columns or entity_col == windows.index.name
    assert window_group_col is None or (window_group_col, '') in windows.columns

    if not agg or windows.empty:
        # no aggregations or no windows => nothing to do
        return windows.copy()
    else:
        assert attribute_col is not None or len(agg) == 1

    # initialize `out` => faster than successive concatenation of partial results
    data = []
    columns = []
    to_drop = []
    if entity_col is not None and (entity_col, '') not in windows.columns:
        to_drop.append((entity_col, ''))
        data.append(windows.index)
        columns.append((entity_col, ''))
    if window_group_col is not None:
        to_drop.extend([(window_group_col, ''), (tmp_col, 'grp')])
        data.append(windows[(window_group_col, '')])
        columns.append((tmp_col, 'grp'))
    val_val = _get_default_value(df[value_col].dtype)
    time_val = _get_default_value(df[time_col].dtype)
    zero_val = np.array(0, dtype=np.float64)

    # partition aggregations
    standard_agg = {}       # maps attributes to lists of standard aggregation functions
    mode_agg = {}           # maps attributes to pairs `(mode, mode_count)`, where both components have type bool
    quantile_agg = {}       # maps attributes to lists of pairs `(q, f)`, where `q` is scalar and `f` is string
    nn_rank_agg = {}        # maps attributes to dicts with items `r: [val, time]`, where `val` and `time` are bool
    neg_rank_agg = {}       # like `nn_rank_agg`
    for attr, func in agg.items():
        if not isinstance(func, (list, tuple)):
            func = [func]
        if 'mode' in func:
            mode_agg[attr] = (True, 'mode_count' in func)
        elif 'mode_count' in func:
            mode_agg[attr] = (False, True)
        quant = []
        nn_rank = {}
        neg_rank = {}
        for f in func:
            if isinstance(f, str) and len(f) > 1:
                if f[0] == 'p':
                    try:
                        x = float(f[1:])
                        if 0. <= x <= 100.:
                            quant.append((0.01 * x, f))
                            data.append(val_val)
                            columns.append((attr, f))
                        continue
                    except ValueError:
                        pass
                elif f[0] in 'rt':
                    try:
                        x = int(f[1:])
                        if x >= 0:
                            aux = nn_rank.get(x, [False, False])
                            if f[0] == 'r':
                                aux[0] = True
                            else:
                                aux[1] = True
                            nn_rank[x] = aux
                        else:
                            aux = neg_rank.get(x, [False, False])
                            if f[0] == 'r':
                                aux[0] = True
                            else:
                                aux[1] = True
                            neg_rank[x] = aux
                        data.append(val_val if f[0] == 'r' else time_val)
                        columns.append((attr, f[0] + str(x)))
                        continue
                    except ValueError:
                        pass
            if f in ('count', 'size', 'nunique', 'mode_count'):
                data.append(zero_val)
            else:
                data.append(val_val)
            columns.append((attr, f))
            if f not in ('mode', 'mode_count'):
                standard_agg.setdefault(attr, []).append(f)
        if quant:
            quantile_agg[attr] = quant
        if nn_rank:
            nn_rank_agg[attr] = nn_rank
        if neg_rank:
            neg_rank_agg[attr] = neg_rank

    # initialize `out`
    # order of rows in `out` never changes during execution
    out = pd.DataFrame(index=windows.index, data=dict(enumerate(data)))
    out.columns = pd.MultiIndex.from_tuples(columns)
    out = pd.concat([windows, out], axis=1, sort=False, join='inner')

    # restrict `df` to relevant entries
    mask = df[time_col].notna()
    if entity_col is not None:
        if entity_col in df.columns:
            mask &= df[entity_col].isin(out[(entity_col, '')])
        else:
            mask &= df.index.isin(out[(entity_col, '')])
    if attribute_col is not None:
        mask &= df[attribute_col].isin(agg.keys())
    df = df[mask]

    # compute aggregations for which observations and windows do not need to be merged
    if neg_rank_agg:
        if (time_col, 'stop') not in out.columns:
            _resample_eav_ranks_2(df, out, neg_rank_agg, entity_col, time_col, attribute_col, value_col, include_start)
            neg_rank_done = True
        else:
            neg_rank_done = False
    else:
        neg_rank_done = True
    if nn_rank_agg:
        if (time_col, 'start') not in out.columns:
            _resample_eav_ranks_2(df, out, nn_rank_agg, entity_col, time_col, attribute_col, value_col, include_stop)
            nn_rank_done = True
        else:
            nn_rank_done = False
    else:
        nn_rank_done = True

    # return if all requested aggregations have been computed already
    if neg_rank_done and nn_rank_done and not (standard_agg or mode_agg or quantile_agg):
        out.drop(to_drop, axis=1, inplace=True, errors='ignore')
        return out

    # merge windows with observations
    cols = [time_col]
    if entity_col is not None and entity_col in df.columns:
        cols.append(entity_col)
    df0 = df[cols].copy()
    df0.reset_index(inplace=True, drop=entity_col is None or entity_col in df.columns)
    df0['__observation_idx__'] = np.arange(len(df0))
    df0['__window_idx__'] = -1.
    df1 = pd.DataFrame(data={'__window_idx__': np.arange(len(out), dtype=np.float64),
                             '__observation_idx__': -1})

    if neg_rank_agg and (time_col, 'start') in out.columns and (time_col, 'stop') in out.columns:
        # special case: sort by "stop" first to handle negative ranks
        df1[time_col] = out[(time_col, 'stop')].values
        sort_col = [time_col, '__window_idx__' if include_stop else '__observation_idx__']
        if entity_col is not None:
            df1[entity_col] = out[(entity_col, '')].values
            sort_col = [entity_col] + sort_col
        merged = pd.concat([df0, df1], axis=0, sort=False, ignore_index=True)
        merged['__is_observation__'] = merged['__observation_idx__'] >= 0
        merged.sort_values(sort_col, inplace=True)
        _resample_eav_ranks(df, out, merged, neg_rank_agg, entity_col, time_col, attribute_col,
                            value_col, include_start)
        neg_rank_done = True
        if not (nn_rank_agg or standard_agg or mode_agg or quantile_agg):
            out.drop(to_drop, axis=1, inplace=True, errors='ignore')
            return out

    # now proper sorting wrt. "start" if possible, otherwise "stop"
    if (time_col, 'start') in out.columns:
        df1[time_col] = out[(time_col, 'start')].values
        sort_col = [time_col, '__observation_idx__' if include_start else '__window_idx__']
        fill_method = 'ffill'
    else:
        df1[time_col] = out[(time_col, 'stop')].values
        sort_col = [time_col, '__window_idx__' if include_stop else '__observation_idx__']
        fill_method = 'bfill'
    if entity_col is not None:
        df1[entity_col] = out[(entity_col, '')].values
        sort_col = [entity_col] + sort_col
    merged = pd.concat([df0, df1], axis=0, sort=False, ignore_index=True)
    merged['__is_observation__'] = merged['__observation_idx__'] >= 0
    del df0, df1
    merged.sort_values(sort_col, inplace=True)

    # compute remaining rank aggregations
    if neg_rank_agg and not neg_rank_done:
        _resample_eav_ranks(df, out, merged, neg_rank_agg, entity_col, time_col, attribute_col, value_col,
                            include_start)
    if nn_rank_agg and not nn_rank_done:
        _resample_eav_ranks(df, out, merged, nn_rank_agg, entity_col, time_col, attribute_col, value_col,
                            include_stop)

    merged.loc[merged['__window_idx__'] < 0., '__window_idx__'] = np.nan

    if standard_agg or mode_agg or quantile_agg:
        # assign windows to groups s.t. all windows in same group are disjoint
        to_drop.append((tmp_col, 'grp'))
        group_windows(out, entity_col=entity_col, time_col=time_col, target=(tmp_col, 'grp'),
                      include_both_endpoints=include_start and include_stop)

        # set groups of empty windows to -1
        if (time_col, 'start') in out.columns and (time_col, 'stop') in out.columns:
            if include_start and include_stop:
                out.loc[out[(time_col, 'start')] > out[(time_col, 'stop')], (tmp_col, 'grp')] = -1
            else:
                out.loc[out[(time_col, 'start')] >= out[(time_col, 'stop')], (tmp_col, 'grp')] = -1

        all_groups = {g for g in out[(tmp_col, 'grp')].unique() if g >= 0}
        assert all_groups
    else:
        all_groups = set()

    # iterate over groups of disjoint windows
    for grp in all_groups:
        group_mask = out[(tmp_col, 'grp')] == grp
        group_merged = merged[merged['__window_idx__'].isin(np.where(group_mask)[0]) |
                              merged['__window_idx__'].isna()].copy()
        if entity_col is None:
            s = group_merged['__window_idx__'].fillna(method=fill_method)
        else:
            s = group_merged.groupby(entity_col)['__window_idx__'].fillna(method=fill_method)
        s = s.fillna(-1).astype(np.int64)
        assert len(s) == len(group_merged) and (s.index == group_merged.index).all()
        group_merged['__window_idx__'] = s
        group_merged = group_merged[group_merged['__window_idx__'] >= 0]
        mask = group_merged['__observation_idx__'] >= 0
        if (time_col, 'start') in out.columns and (time_col, 'stop') in out.columns:
            # otherwise, if only "stop" occurs in `out`, `merged` has been sorted wrt. "stop" already => nothing to do
            group_merged = group_merged.join(
                pd.Series(data=out[(time_col, 'stop')].values, name='__stop__'),
                on='__window_idx__'
            )
            if include_stop:
                mask &= group_merged[time_col] <= group_merged['__stop__']
            else:
                mask &= group_merged[time_col] < group_merged['__stop__']
        group_merged = group_merged[mask]
        # `group_merged` has columns "__observation_idx__" and "__window_idx__", mapping former to latter; all entries
        # are valid, i.e., >= 0; not all windows may appear

        df0 = df.iloc[group_merged['__observation_idx__'].values].copy()
        df0.index = group_merged['__window_idx__'].values

        # resample `df0`, using windows as new entities
        _resample_eav_no_windows(df0, out, standard_agg, mode_agg, quantile_agg, attribute_col, value_col)

    out.drop(to_drop, axis=1, inplace=True, errors='ignore')
    return out


def resample_interval(df: pd.DataFrame, windows: pd.DataFrame, attributes: list = None, entity_col=None,
                      start_col=None, stop_col=None, attribute_col=None, value_col=None, time_col=None,
                      epsilon=1e-7) -> pd.DataFrame:
    """
    Resample interval-like data wrt. explicitly passed windows of arbitrary (possibly infinite) length. "Interval-like"
    means that each observation is characterized by a start- and stop time rather than a singular timestamp (as in EAV
    data). A typical example of interval-like data are medication records, since medications can be administered over
    longer time periods.

    The only supported resampling aggregation is summing the observed values per time window, scaled by the fraction
    of the length of the intersection of observation interval and time window divided by the total length of the
    observation interval: Let `W = [s, t]` be a time window and let `I = [a, b]` be an observation interval with
    observed value `v`. Then `I` contributes to `W` the value

                   |W n I|
        W_I = v * ---------         (1)
                     |I|

    The overall value of `W` is the sum of `W_I` over all intervals. Of course, all this is computed separately for
    each entity-attribute combination.
    Some remarks on Eq. (1) are in place:
        * If `a = b` both numerator and denominator are 0. In this case the fraction is defined as 1 if `a in W`
            (i.e., `s <= a <= t`) and 0 otherwise.
        * If `I` is infinite and `W n I` is non-empty but finite, `W_I` is set to `epsilon * sign(v)`.
            Note that `W n I` is non-empty even if it is of the form `[x, x]`. This leads to the slightly
            counter-intuitive situation that `W_I = epsilon` if `I` is infinite, and `W_I = 0` if `I` is finite.
        * If `I` and `W n I` are both infinite, the fraction is defined as 1. This is regardless of whether `W n I`
            equals `I` or whether it is a proper subset of it.

    :param df: The DataFrame to resample. Must have columns `value_col` (contains observed values), `start_col`
    (optional; contains start times), `stop_time` (optional; contains end times), `attribute_col` (optional; contains
    attribute identifiers) and `entity_col` (optional; contains entity identifiers). Must have one column index level.
    Data types are arbitrary, as long as times and entity identifiers can be compared wrt. `<` and `<=`
    (e.g., float, int, time delta, date time). Entity identifiers must not be NA. Values must be numeric (float, int,
    bool).
    Although both `start_col` and `stop_col` are optional, at least one must be present. Missing start- and end
    columns are interpreted as -/+ inf.
    All intervals are closed, i.e., start- and end times are included. This is especially relevant for entries whose
    start time equals their end time.

    :param windows: The target windows into which `df` is resampled. Must have either one or two columns index level(s).
    If it has one column index level, must have columns `start_col` (optional; contains start times of each window),
    `stop_col` (optional; contains end times of each window) and `entity_col` (optional; contains entity identifiers).
    If it has two column index levels, the columns must be `(time_col, "start")`, `(time_col, "stop")` and
    `(entity_col, "")`.
    At least one of the two endpoint-columns must be present; if one is missing it is assumed to represent -/+ inf.
    All time windows are closed, i.e., start- and end times are included.

    :param attributes: The attributes to consider. Must be a list-like of attribute identifiers. None defaults to the
    list of all such identifiers present in column `attribute_col`. If `attribute_col` is None but `attributes` is not,
    it must be a singleton list.

    :param entity_col: Name of the column in `df` and `windows` containing entity identifiers. If None, all entries
    are assumed to belong to the same entity. Note that entity identifiers may also be on the row index.

    :param start_col: Name of the column in `df` (and `windows` if it has only one column index level) containing start
    times. If None, all start times are assumed to be -inf. Note that despite its name the data type of the column is
    arbitrary, as long as its values can be compared wrt. `<` and `<=`.

    :param stop_col: Name of the column in `df` (and `windows` if it has only one column index level) containing end
    times. If None, all end times are assumed to be +inf. Note that despite its name the data type of the column is
    arbitrary, as long as its values can be compared wrt. `<` and `<=`.

    :param attribute_col: Name of the column in `df` containing attribute identifiers. If None, all entries are assumed
    to belong to the same attribute.

    :param value_col: Name of the column in `df` containing the observed values.

    :param time_col: Name of the column(s) in `windows` containing start- and end times of the windows. Only needed if
    `windows` has two column index levels, because otherwise these two columns must be called `start_col` and
    `stop_col`, respectively.

    :param epsilon: The value to set `W_I` to if `I` is infinite and `W n I` is non-empty and finite; see Eq. (1) and
    the subsequent remarks for details.

    :return: Resampled data. Like `windows`, but with one additional column for each attribute.
    Order of columns is arbitrary, order of rows is exactly as in `windows`. Number of column index levels is as in
    `windows`.
    """

    # Meaningful aggregations besides "sum" are
    #   * number of intervals with non-empty intersection per window ("count")
    #   * duration of intersections per window ("duration")
    #       => cannot be achieved by setting `value` to interval duration if intervals are infinite
    #   * total duration of intervals with non-empty intersection per window ("total_duration")
    #   * ... any other aggregation on durations of intersections or total durations of intersecting intervals
    # One could consider adding these in the future.

    assert df.columns.nlevels == 1
    assert value_col in df.columns
    assert df[value_col].dtype.kind in 'fiub'
    assert start_col in df.columns or stop_col in df.columns
    assert attribute_col is None or attribute_col in df.columns

    if entity_col is None or entity_col == df.index.name:
        df = df.drop([c for c in df.columns if c not in (start_col, stop_col, value_col, attribute_col)], axis=1)
    else:
        assert entity_col in df.columns
        df = df.set_index(entity_col)
        df.drop([c for c in df.columns if c not in (start_col, stop_col, value_col, attribute_col)],
                axis=1, inplace=True)

    if start_col in df.columns:
        time_dtype = df[start_col].dtype
        if stop_col in df.columns:
            assert df[stop_col].dtype == time_dtype
    else:
        assert stop_col in df.columns
        time_dtype = df[stop_col].dtype

    window_start_col = 'window_start'
    window_stop_col = 'window_stop'
    windows_orig = windows
    if windows.columns.nlevels == 2:
        columns = {c_cur: c_new for c_cur, c_new in [((time_col, 'start'), window_start_col),
                                                     ((time_col, 'stop'), window_stop_col),
                                                     ((entity_col, ''), entity_col)] if c_cur in windows.columns}
    else:
        assert windows.columns.nlevels == 1
        columns = {c_cur: c_new for c_cur, c_new in [(start_col, window_start_col), (stop_col, window_stop_col),
                                                     (entity_col, entity_col)] if c_cur in windows.columns}
    windows = windows[list(columns)].copy()
    windows.columns = list(columns.values())

    assert window_start_col in windows.columns or window_stop_col in windows.columns
    assert window_start_col not in windows.columns or \
           (windows[window_start_col].notna().all() and windows[window_start_col].dtype == time_dtype)
    assert window_stop_col not in windows.columns or \
           (windows[window_stop_col].notna().all() and windows[window_stop_col].dtype == time_dtype)

    if entity_col is None or entity_col in windows.columns:
        windows.reset_index(drop=True, inplace=True)
    else:
        assert entity_col == windows.index.name
        windows.reset_index(inplace=True)

    # restrict `df` to relevant entries
    if start_col in df.columns:
        mask = df[start_col].notna()
        if stop_col in df.columns:
            mask &= df[stop_col].notna() & (df[start_col] <= df[stop_col])
    else:
        mask = df[stop_col].notna()
    if entity_col is not None:
        mask &= df.index.isin(windows[entity_col])
    if attribute_col in df.columns:
        if attributes is None:
            attributes = df[attribute_col].unique()
        else:
            mask &= df[attribute_col].isin(attributes)
    else:
        if attributes is None:
            attributes = ['sum']
        else:
            assert len(attributes) == 1
    if not mask.all():
        df = df[mask].copy()

    if df.empty:
        out = pd.DataFrame(index=windows.index, columns=attributes, data=0, dtype=np.float32)
    elif (start_col in df.columns or window_start_col in windows.columns) \
            and (stop_col in df.columns or window_stop_col in windows.columns):
        if entity_col is None:
            # get all interval-window combinations, resulting in a DataFrame with `len(windows) * len(df)` rows
            aux = pd.DataFrame(index=np.tile(windows.index, len(df)),
                               data={c: np.tile(windows[c].values, len(df)) for c in windows.columns})
            for c in df.columns:
                aux[c] = np.repeat(df[c].values, len(windows))
            out = _resample_interval_aux(aux, start_col, stop_col, value_col, attribute_col, window_start_col,
                                         window_stop_col, epsilon).reindex(windows.index, fill_value=0)
            if isinstance(out, pd.Series):
                out = out.to_frame(attributes[0])
        else:
            _MAX_ROWS = max(len(df), 10000000)  # maximum number of rows of DataFrame to process at once

            # `n_rows` contains the number of rows of `windows.join(df, on=entity_col, how='inner')` for each entity
            n_rows = df.index.value_counts().to_frame(name=entity_col + '_') \
                .join(windows[entity_col].value_counts(), how='inner').prod(axis=1)
            if n_rows.sum() <= _MAX_ROWS:
                partition = np.zeros((len(windows),), dtype=np.int8)
            else:
                partition = partition_series(n_rows, _MAX_ROWS).reindex(windows[entity_col], fill_value=0).values

            # the following could easily be parallelized
            dfs = [
                _resample_interval_aux(windows[partition == g].join(df, on=entity_col, how='inner'),
                                       start_col, stop_col, value_col, attribute_col,
                                       window_start_col, window_stop_col, epsilon)
                for g in range(partition.max() + 1)
            ]
            out = pd.concat(dfs, axis=0, sort=False).reindex(windows.index, fill_value=0)
            if isinstance(out, pd.Series):
                out = out.to_frame(attributes[0])
    else:
        # all intersections are infinite => no need to take windows into account
        if entity_col is None:
            if attribute_col is None:
                out = pd.DataFrame(index=windows.index, data={attributes[0]: df[value_col].sum()})
            else:
                s = df.groupby(attribute_col)[value_col].sum()
                out = pd.DataFrame(index=windows.index, columns=s.index, dtype=s.dtype)
                for a, v in s.iteritems():
                    out[a] = v
        else:
            if attribute_col is not None:
                df.set_index(attribute_col, append=True, inplace=True)
            s = df.groupby(level=list(range(df.index.nlevels)))[value_col].sum()
            if s.index.nlevels == 1:
                s = s.to_frame(attributes[0])
            else:
                s = s.unstack(level=-1, fill_value=0)
            out = s.reindex(windows[entity_col], fill_value=0)
            out.index = windows.index

    # `out` is a DataFrame with one column per attribute and the exact same row index as `windows`
    # (which is a RangeIndex)

    # make sure that all requested attributes appear in `out`
    out = out.reindex(attributes, axis=1, fill_value=0)

    out.sort_index(inplace=True)    # to be on the safe side
    out.index = windows_orig.index
    if windows_orig.columns.nlevels == 2:
        out.columns = pd.MultiIndex.from_product([out.columns, ['']])
    return pd.concat([windows_orig, out], axis=1, sort=False)


def partition_series(s: pd.Series, n, shuffle: bool = True) -> pd.Series:
    """
    Partition a given Series into as few groups as possible, such that the sum of the series' values in each group does
    not exceed a given threshold.
    :param s: The series to partition. The data type should allow for taking sums and comparing elements, and all
    values are assumed to be non-negative.
    :param n: The threshold, of the same data type as `s`.
    :param shuffle: Whether to randomly shuffle `s` before generating the partition. If False, this function is
    deterministic.
    :return: A new series with the same index as `s`, with values from 0 to `g - 1` specifying the group ID of each
    entry (`g` is the total number of groups).

    Note that the partitions returned by this function may not be optimal, since finding optimal partitions is
    computationally difficult.
    Also note that `s` may contain entries whose value exceeds `n`; such entries are put into singleton groups.
    """

    out = pd.Series(index=s.index, data=0, dtype=np.int64)
    if s.sum() <= n:
        return out

    groups = {}
    if shuffle:
        rng = np.random.permutation(len(s))
    else:
        rng = range(len(s))
    m = 0
    for i in rng:
        x = s.iloc[i]
        j = -1
        if x < n:
            for k, v in groups.items():
                if v + x <= n:
                    groups[k] += x
                    j = k
                    break
        if j < 0:
            j = m
            m += 1
            groups[j] = x
        out.iloc[i] = j

    return out


def group_windows(windows: pd.DataFrame, entity_col=None, time_col=None, target=None,
                  include_both_endpoints: bool = False) -> pd.DataFrame:
    """
    Group windows such that each group only contains mutually disjoint windows for each entity.
    :param windows: The windows to group, modified in place. Must have two column index levels.
    :param entity_col: Name of the column in `windows` containing entity identifiers, or None. Actually, the column is
    `(entity_col, "")`.
    :param time_col: Name of the column(s) in `windows` containing start- and end times. Actually, the columns are
    `(time_col, "start")` and `(time_col, "stop")`.
    :param target: Name of the new column containing the group indices. If a string, the new column is `(target, "")`.
    :param include_both_endpoints: Whether both endpoints are part of the respective windows.
    :return: `windows` with the additional column `target`, with non-negative integer values.
    Order of rows is preserved.
    """
    assert time_col is not None
    assert entity_col is None or (entity_col, '') in windows.columns
    if isinstance(target, tuple):
        assert len(target) == 2
    else:
        assert target is not None
        target = (target, '')

    if (time_col, 'start') in windows.columns and (time_col, 'stop') in windows.columns:
        data = dict(start=windows[(time_col, 'start')].values, stop=windows[(time_col, 'stop')].values)
        if target in windows.columns:
            assert (windows[target] >= 0).all()
            if not windows[target].dtype.kind == 'i':
                windows[target] = windows[target].astype(np.int32)
            data['target'] = windows[target].values
        if entity_col is None:
            sort_col = ['start']
            data['entity'] = 0
        else:
            sort_col = ['entity', 'start']
            data['entity'] = windows[(entity_col, '')].values

        windows_aux = pd.DataFrame(data=data)
        windows_aux.sort_values(sort_col, inplace=True)

        if 'target' in windows_aux.columns:
            for g in windows_aux['target'].unique():
                mask = _check_disjoint(windows_aux[windows_aux['target'] == g], 'entity', 'start', 'stop',
                                       include_both_endpoints)
                assert not mask.any(), 'Provided grouping does not make windows mutually disjoint!'
        else:
            _pregroup_windows(windows_aux, 'entity', 'start', 'stop', 'pre_group', include_both_endpoints)
            s = windows_aux.groupby('entity')['pre_group'].agg(['min', 'max'])
            s = s['max'] > s['min']
            if s.any():
                # if offending entities exist, they must be processed manually => SLOW!
                mask = windows_aux['entity'].isin(s[s].index)
                windows_grouped = \
                    windows_aux[mask].groupby('pre_group').agg({'entity': 'first', 'start': 'min', 'stop': 'max'})
                # `windows_grouped` has "pre_group" on the row index and three columns "entity", "start", "stop"

                grp_idx = 0
                windows_grouped['target'] = 0
                windows_grouped['i'] = np.arange(len(windows_grouped))
                mask0 = np.ones((len(windows_grouped),), dtype=bool)
                comp = '__gt__' if include_both_endpoints else '__ge__'
                while mask0.any():
                    indices = []
                    sub = windows_grouped[mask0]
                    mask1 = np.ones((mask0.sum(),), dtype=bool)
                    while mask1.any():
                        cur = sub[mask1].groupby('entity')[['stop', 'i']].first()
                        indices.append(cur['i'])
                        cur.columns = ['stop', 'j']
                        tmp = sub[['entity', 'start', 'i']][mask1].join(cur, on='entity', how='left')
                        mask1 &= (tmp['i'] > tmp['j']) & (getattr(tmp['start'], comp)(tmp['stop']))
                    indices = np.concatenate(indices)
                    windows_grouped['target'].values[indices] = grp_idx
                    grp_idx += 1
                    mask0[mask0] &= ~sub['i'].isin(indices)
                # `windows_grouped` now has an additional column "target"

                windows_aux['target'] = windows_grouped['target'].reindex(windows_aux['pre_group'], fill_value=0).values
                windows_aux.sort_index(inplace=True)
                windows[target] = windows_aux['target'].values
            else:
                # all windows are disjoint => nothing to do here
                windows[target] = 0
    else:
        # replace `target` if it is already present
        if entity_col is None:
            windows[target] = np.arange(len(windows))
        else:
            s = pd.Series(index=windows[(entity_col, '')].values, data=1).groupby(level=0).cumsum() - 1
            assert len(s) == len(windows)
            assert (s.index == windows[(entity_col, '')]).all()
            windows[target] = s.values

    return windows


def grouped_mode(series: pd.Series) -> pd.DataFrame:
    """
    Group the given Series `series` by its row index and compute mode aggregations. If there are more than one most
    common values in a group, the "first" is chosen, i.e., the result is always one single value.
    NaN values are ignored, i.e., the most frequent value of a group is NaN iff all values of the group are NaN.

    Very fast method to compute grouped mode, based on https://stackoverflow.com/a/38216118. Using built-in `mode()`
    function is not possible, because returns list of most common values. Work-around a la `lambda x: x.mode()[0]` is
    **terribly** slow.

    :param series: The Series to aggregate. The number of row index levels is arbitrary.
    :return: The mode-aggregated DataFrame with columns "mode" and "count"
    """

    group_levels = list(range(series.index.nlevels + 1))
    idx = series.groupby(level=group_levels[:-1]).size().index
    mask = series.notna().values
    if mask.any():
        series = series.to_frame('mode')
        series.set_index('mode', append=True, inplace=True)
        df = series[mask].groupby(level=group_levels).size().to_frame('count')
        df.reset_index(level=-1, inplace=True)
        df.sort_values('count', ascending=False, inplace=True, kind='stable')
        df = df.loc[~df.index.duplicated()].reindex(idx)
        df['count'] = df['count'].fillna(0).astype(np.int64)
    else:
        df = series.iloc[:0].to_frame('mode')
        df['count'] = 0

    return df


def _get_default_value(dtype):
    if dtype.name == 'category':
        return pd.Categorical.from_codes([-1], dtype=dtype)
    elif dtype.kind in 'mMf':
        return np.array(None, dtype=dtype)
    elif dtype.kind in 'iu':
        return np.array(None, dtype=np.float64)
    elif dtype.kind == 'b':
        return np.array(None, dtype=np.float32)
    return None


def _check_disjoint(df: pd.DataFrame, entity_col, start_col, stop_col, include_both_endpoints: bool) -> np.ndarray:
    # assumes `df` is sorted wrt. `entity_col` and `start_col`

    if entity_col is None:
        mask = np.ones((len(df),), dtype=np.bool)
    else:
        mask = np.roll(df[entity_col].values, -1) == df[entity_col].values
    mask[-1] = False
    # `mask` now indicates whether next entry belongs to same entity

    next_start = np.roll(df[start_col].values, -1)
    if include_both_endpoints:
        mask &= (df[stop_col] >= next_start)
    else:
        mask &= (df[stop_col] > next_start)
    # `mask` now indicates whether next entry belongs to same entity and starts before current entry ends

    return mask


def _pregroup_windows(df: pd.DataFrame, entity_col, start_col, stop_col, grp_col, include_both_endpoints: bool):
    # assumes `df` is sorted wrt. `entity_col` and `start_col`

    prev_stop = np.roll(df[stop_col].values, 1)
    if include_both_endpoints:
        mask = (df[start_col] <= prev_stop).values
    else:
        mask = (df[start_col] < prev_stop).values
    # `mask` indicates whether previous entry stops after current entry starts

    if entity_col is not None:
        mask |= np.roll(df[entity_col].values, 1) != df[entity_col].values
    mask[0] = False
    # `mask` now indicates whether previous entry stops after current entry starts,
    # or previous entry belongs to different entity

    df[grp_col] = np.cumsum(mask)


def _resample_eav_no_windows(df: pd.DataFrame, out: pd.DataFrame, standard_agg: dict, mode_agg: dict,
                             quantile_agg: dict, attribute_col, value_col) -> None:
    # auxiliary function for resampling EAV DataFrames without reference to windows, i.e., aggregations are computed
    #     over _all_ observations per entity and attribute
    # assumes that `df` contains no redundant attributes and that row index (single level!) corresponds to entities
    # assumes `out` is output DataFrame with all necessary columns; connected to `df` via row index of `df`, which
    #     refers to rows in `out` through `iloc[]`
    # `df` is left unchanged, `out` is modified in place

    if attribute_col is None:
        attr_mask = pd.Series(index=df.index, data=True)
    else:
        attr_mask = None

    for attr, agg in standard_agg.items():
        aux = df[attr_mask if attribute_col is None else (df[attribute_col] == attr)]\
            .groupby(level=0)[value_col].agg(agg)
        if isinstance(aux, pd.Series):
            aux = aux.to_frame()
        for i, a in enumerate(agg):
            out[(attr, a)].values[aux.index.values] = aux.iloc[:, i].values

    for attr, (mode, mode_count) in mode_agg.items():
        aux = grouped_mode(df.loc[attr_mask if attribute_col is None else (df[attribute_col] == attr), value_col])
        if mode:
            if mode_count:
                columns = [((attr, 'mode'), 'mode'), ((attr, 'mode_count'), 'count')]
            else:
                columns = [((attr, 'mode'), 'mode')]
        else:
            columns = [((attr, 'mode_count'), 'count')]
        for c_out, c_aux in columns:
            out[c_out].values[aux.index.values] = aux[c_aux].values

    for attr, quantiles in quantile_agg.items():
        aux = df[attr_mask if attribute_col is None else (df[attribute_col] == attr)] \
            .groupby(level=0)[value_col].quantile(q=[q for q, _ in quantiles]).unstack()
        for i, (_, a) in enumerate(quantiles):
            out[(attr, a)].values[aux.index.values] = aux.iloc[:, i].values


def _resample_eav_ranks(df: pd.DataFrame, out: pd.DataFrame, merged: pd.DataFrame, agg: dict, entity_col, time_col,
                        attribute_col, value_col, include_endpoint: bool) -> None:
    # _extremely_ fast auxiliary function for computing rank-like aggregations per time window
    # assumes all ranks are non-negative, or all ranks are negative
    # assumes `df` contains observations and has columns `time_col` and `value_col`, and optionally `attribute_col`
    #     (unless None)
    # assumes `out` contains observation windows and has one column for each requested rank aggregation, and optionally
    #     `(time_col, "start")`, `(time_col, "stop")` and `(entity_col, "")`; windows may overlap
    # assumes `merged` is combination of observations and windows and has columns `time_col`, "__is_observation__",
    #     "__observation_idx__", "__window_idx__", and optionally `entity_col` (unless None)
    # assumes `merged` is sorted wrt. entities and timestamps, where timestamps of windows are start times if ranks are
    #     non-negative and stop times otherwise
    # assumes `merged` has unique row index
    # assumes windows in `merged` are connected to `out` via "__window_idx__",
    #     which refers to rows in `out` through `iloc[]`
    # input DataFrames except `out` are left unchanged

    signs = list({r >= 0 for ranks in agg.values() for r in ranks})
    assert len(signs) == 1
    if signs[0]:
        # only non-negative ranks => `(time_col, "start")` was used to sort `windows` in `merged`
        other_endpoint = (time_col, 'stop')
        comp = 'lt' if include_endpoint else 'le'
        merged = merged.iloc[::-1]      # revert `merged`, s.t. we can pretend to extract _last_ observations
    else:
        # only negative ranks => `(time_col, "stop")` was used to sort `windows` in `merged`
        other_endpoint = (time_col, 'start')
        comp = 'gt' if include_endpoint else 'ge'
    if other_endpoint not in out.columns:
        other_endpoint = None

    for attr, ranks in agg.items():
        if signs[0]:
            # pretend to extract _last_ observations
            rank_indices = [-r - 1 for r in ranks]
        else:
            rank_indices = list(ranks)

        if attribute_col is None:
            aux = merged
        else:
            aux = merged[merged['__observation_idx__'].isin(np.where(df[attribute_col] == attr)[0])
                         | ~merged['__is_observation__']]
        overall_idx = aux['__is_observation__'].cumsum()
        overall_idx.sort_index(inplace=True)
        overall_idx = overall_idx[~merged['__is_observation__']]

        if entity_col is None:
            grp_idx = overall_idx.copy()
        else:
            grp_idx = aux.groupby(entity_col)['__is_observation__'].cumsum().astype('int32')
            grp_idx = grp_idx[grp_idx.index.isin(overall_idx.index)]    # row index of `df` is unique!
        grp_idx = grp_idx.repeat(len(ranks)).to_frame('grp_idx')
        grp_idx.sort_index(inplace=True)                                # to have same index as `overall_idx`
        grp_idx['__rank__'] = np.tile(np.array(rank_indices, dtype=np.int32), len(grp_idx) // len(ranks))
        grp_idx['__i__'] = overall_idx.repeat(len(ranks)) + grp_idx['__rank__']
        grp_idx.loc[grp_idx['grp_idx'] + grp_idx['__rank__'] < 0, '__i__'] = -1

        sub = aux.loc[aux['__is_observation__'], [time_col, '__observation_idx__']].reset_index(drop=True)
        aux = grp_idx.join(sub, on='__i__', how='left').join(aux[['__window_idx__']], how='left')
        aux['__window_idx__'] = aux['__window_idx__'].astype(np.int64)
        aux['__observation_idx__'] = aux['__observation_idx__'].fillna(-1).astype(sub['__observation_idx__'].dtype)
        aux.set_index('__window_idx__', inplace=True)
        # `aux` has columns `time_col`, "__rank__" and "__observation_idx__"; other columns are obsolete

        if other_endpoint is not None:
            start = pd.Series(data=out[other_endpoint].values).reindex(aux.index, copy=False)
            if include_endpoint:
                mask = getattr(start, comp)(aux[time_col])
            else:
                mask = getattr(start, comp)(aux[time_col])
            aux.loc[mask, '__observation_idx__'] = -1
            aux.loc[mask, time_col] = None

        if any(v for v, _ in ranks.values()):
            aux = aux.join(df[value_col].reset_index(drop=True), on='__observation_idx__')
        assert len(aux) == len(out) * len(ranks)
        aux.sort_index(inplace=True)

        for r, (v, t) in ranks.items():
            if signs[0]:
                r0 = -r - 1
            else:
                r0 = r
            m = aux['__rank__'] == r0
            if v:
                out[(attr, 'r' + str(r))].values[:] = aux.loc[m, value_col].values
            if t:
                out[(attr, 't' + str(r))].values[:] = aux.loc[m, time_col].values


def _resample_eav_ranks_2(df: pd.DataFrame, out: pd.DataFrame, agg: dict, entity_col, time_col,
                          attribute_col, value_col, include_endpoint: bool) -> None:
    # _extremely_ fast auxiliary function for computing rank-like aggregations per time window
    # same assumptions as in function `_resample_eav_ranks()`, except that timestamps of windows used for sorting are
    #     _stop_ times if ranks are non-negative and _start_ times otherwise
    # assumes that windows in `out` are defined by single endpoint

    signs = list({r >= 0 for ranks in agg.values() for r in ranks})
    assert len(signs) == 1
    if signs[0]:
        other_endpoint = (time_col, 'stop')
        comp = 'le' if include_endpoint else 'lt'
    else:
        other_endpoint = (time_col, 'start')
        comp = 'ge' if include_endpoint else 'gt'
    assert other_endpoint in out.columns

    # add ranks to each entity-attribute group
    rank_attrs = list(agg)
    group_cols = []
    if attribute_col is None:
        df0 = df
    else:
        df0 = df[df[attribute_col].isin(rank_attrs)]
        group_cols.append(attribute_col)
    if entity_col is not None:
        if entity_col in df.columns:
            df0 = df0.copy()
        else:
            df0 = df0.reset_index()
        group_cols = [entity_col] + group_cols

    if group_cols:
        ranks = df0[group_cols + [time_col]].groupby(group_cols).rank(method='first', ascending=signs[0])[time_col]
    else:
        ranks = df0[time_col].rank(method='first', ascending=signs[0])
    assert len(ranks) == len(df0)
    ranks.fillna(1., inplace=True)
    ranks = ranks.astype(np.int32)
    if signs[0]:
        ranks -= 1
    else:
        ranks = -ranks
    df0.index = ranks.values

    if attribute_col is None:
        attr_mask = pd.Series(index=df0.index, data=True)

    for attr, ranks in agg.items():
        if attribute_col is not None:
            attr_mask = df0[attribute_col] == attr
        mask = attr_mask & df0.index.isin(list(ranks))
        if mask.any():
            if entity_col is None:
                aux = df0.loc[mask, [value_col, time_col]].set_index(pd.Index(np.zeros((mask.sum(),), dtype=np.int8)),
                                                                     append=True)
                aux = aux.unstack(0).repeat(len(out))
            else:
                aux = df0.loc[mask, [entity_col, value_col, time_col]].set_index(entity_col, append=True)
                aux = aux.unstack(0).reindex(out[(entity_col, '')])
            for r, (v, t) in ranks.items():
                if (time_col, r) in aux.columns:
                    mask = (getattr(aux[(time_col, r)], comp)(out[other_endpoint].values)).values
                    if mask.any():
                        if v:
                            out[(attr, 'r' + str(r))].values[mask] = aux[(value_col, r)].values[mask]
                        if t:
                            out[(attr, 't' + str(r))].values[mask] = aux[(time_col, r)].values[mask]


def _resample_interval_aux(df: pd.DataFrame, start_col, stop_col, value_col, attribute_col,
                           window_start_col, window_stop_col, epsilon) -> Union[pd.Series, pd.DataFrame]:
    # `df` is changed in place
    # assumes that `start_col` or `window_start_col`, as well as `stop_col` or `window_stop_col`, are in `df`
    # assumes that `df[start_col] <= df[stop_col]` if both columns appear in `df`

    if start_col in df.columns and stop_col in df.columns:
        duration = df[stop_col] - df[start_col]
    else:
        duration = None

    if start_col in df.columns:
        if window_start_col in df.columns:
            start = np.maximum(df[start_col], df[window_start_col])
        else:
            start = df[start_col]
    else:
        start = df[window_start_col]

    if stop_col in df.columns:
        if window_stop_col in df.columns:
            stop = np.minimum(df[stop_col], df[window_stop_col])
        else:
            stop = df[stop_col]
    else:
        stop = df[window_stop_col]

    factor = pd.Series(index=df.index, data=0, dtype=np.float32)
    if duration is None:
        try:
            is_inf = np.isposinf(stop - start)
            is_eps = (~is_inf) & (stop >= start)
            factor[is_inf] = 1
        except TypeError:
            is_eps = stop >= start
    else:
        numerator = stop - start
        mask = df[start_col] < df[stop_col]  # we assume that `df[start_col] <= df[stop_col]`
        factor[~mask & (stop >= start)] = 1
        mask &= stop >= start
        try:
            is_num_inf = np.isposinf(numerator)
            is_denom_inf = np.isposinf(duration)
            is_eps = (~is_num_inf) & is_denom_inf & mask
            factor[is_num_inf & is_denom_inf] = 1
            mask &= ~(is_num_inf | is_denom_inf)
        except TypeError:
            is_eps = pd.Series(index=df.index, data=False)
        factor.values[mask.values] = (numerator[mask] / duration[mask]).values

    eps_value = epsilon * np.sign(df.loc[is_eps, value_col])
    df[value_col] = df[value_col] * factor
    df.loc[is_eps, value_col] = eps_value

    if attribute_col is not None:
        df.set_index(attribute_col, append=True, inplace=True)
    s = df.groupby(level=list(range(df.index.nlevels)))[value_col].sum()
    if s.index.nlevels > 1:
        s = s.unstack(level=-1, fill_value=0)

    return s
