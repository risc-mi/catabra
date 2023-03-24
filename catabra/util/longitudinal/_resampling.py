#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

from typing import Union

import numpy as np
import pandas as pd

# maximum number of rows to allow for intermediate DataFrames when optimizing for "time"
MAX_ROWS = 10000000


def resample_eav(df: Union[pd.DataFrame, 'dask.dataframe.DataFrame'], # noqa F821
                 windows: Union[pd.DataFrame, 'dask.dataframe.DataFrame'], # noqa F821
                 agg: dict = None, entity_col=None, time_col=None, attribute_col=None, value_col=None,
                 include_start: bool = True, include_stop: bool = False, optimize: str = 'time') \
        -> Union[pd.DataFrame, 'dask.dataframe.DataFrame']: # noqa F821
    """
    Resample data in EAV (entity-attribute-value) format wrt. explicitly passed windows of arbitrary (possibly
    infinite) length.

    :param df: The DataFrame to resample, in EAV format. That means, must have columns `value_col` (contains observed
    values), `time_col` (contains observation times), `attribute_col` (optional; contains attribute identifiers) and
    `entity_col` (optional; contains entity identifiers). Must have one column index level.
    Data types are arbitrary, as long as observation times and entity identifiers can be compared wrt. `<` and `<=`
    (e.g., float, int, time delta, date time). Entity identifiers must not be NA. Observation times may be NA, but such
    entries are ignored entirely.
    `df` can be a Dask DataFrame as well. In that case, however, `entity_col` must not be None and entities should
    already be on the row index, with known divisions. Otherwise, the row index is set to `entity_col`, which can be
    very costly both in terms of time and memory. Especially if `df` is known to be sorted wrt. entities already, the
    calling function should better take care of this; see
    https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.set_index.html.

    :param windows: The target windows into which `df` is resampled. Must have two column index levels and columns
    `(time_col, "start")` (optional; contains start times of each window), `(time_col, "stop")` (optional; contains end
    times of each window), `(entity_col, "")` (optional; contains entity identifiers) and `(window_group_col, "")`
    (optional; contains information for creating groups of mutually disjoint windows). Start- and end times may be NA,
    but such windows are deemed invalid and by definition do not contain any observations.
    At least one of the two endpoint-columns must be given; if one is missing it is assumed to represent +/- inf.
    `windows` can be a Dask DataFrame as well. In that case, however, `entity_col` must not be None and entities should
    already be on the row index, with known divisions. Otherwise, the row index is set to `entity_col`, which can be
    very costly both in terms of time and memory. Especially if `windows` is known to be sorted wrt. entities already,
    the calling function should better take care of this; see
    https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.set_index.html.

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
        * "prod": Product of observed non-NA values
        * "skew": Skewness of observed non-NA values
        * "mad": Mean absolute deviation of observed non-NA values
        * "sem": Standard error of the mean of observed non-NA values
        * "size": Number of observations, including NA values
        * "count": Number of non-NA observations
        * "nunique": Number of unique observed non-NA values
        * "mode": Mode of observed non-NA values, i.e., most frequent value; ties are broken randomly but reproducibly
        * "mode_count": Number of occurrences of mode
        * "pxx": Percentile of observed non-NA values; `xx` is an arbitrary float in the interval [0, 100]
        * "rxx": `xx`-th observed value (possibly NA), starting from 0; negative indices count from the end
        * "txx": Time of `xx`-th observed value; negative indices count from the end
        * callable: Function that takes as input a DataFrame `in` and returns a new DataFrame `out`.
            `in` has two columns `time_col` and `value_col` (in that order). Its row index specifies which entries
            belong to the same observation window: entries with the same row index value belong to the same window,
            entries with different row index values belong to distinct windows. Observation times are guaranteed to be
            non-NA, values may be NA. Note, however, that `in` is _not_ necessarily sorted wrt. its row index and/or
            observation times! Also note that the entities the observations in `in` stem from (if `entity_col` is
            specified) are not known to the function.
            `out` should have one row per row index value of `in` (with the same row index value), and an arbitrary
            number of columns with arbitrary names and dtypes. Columns should be consistent in every invocation of the
            function.
            The reason why the function is not applied to each row-index-value group individually is that some
            aggregations can be implemented efficiently using sorting rather than grouping.
            The function should be stateless and must not modify `in` in place.

            Example 1: A simple aggregation which calculates the fraction of values between 0 and 1 in every window
            could be passed as

                lambda x: x[value_col].between(0, 1).groupby(level=0).mean().to_frame('frac_between_0_1')

            Example 2: A more sophisticated aggregation which fits a linear regression to the observations in every
            window and returns the slope of the resulting regression line could be defined as

                def slope(x):
                    tmp = pd.DataFrame(
                        index=x.index,
                        data={time_col: x[time_col].dt.total_seconds(), value_col: x[value_col]}
                    )
                    return tmp[tmp[value_col].notna()].groupby(level=0).apply(
                        lambda g: scipy.stats.linregress(g[time_col], y=g[value_col]).slope
                    ).to_frame('slope')

    :param entity_col: Name of the column in `df` and `windows` containing entity identifiers. If None, all entries
    are assumed to belong to the same entity. Note that entity identifiers may also be on the row index.

    :param time_col: Name of the column in `df` containing observation times, and also name of column(s) in `windows`
    containing start- and end times of the windows. Note that despite its name the data type of the column is
    arbitrary, as long as its values can be compared wrt. `<` and `<=`.

    :param attribute_col: Name of the column in `df` containing attribute identifiers. If None, all entries are assumed
    to belong to the same attribute; in that case `agg` may only contain one single item.

    :param value_col: Name of the column in `df` containing the observed values.

    :param include_start: Whether start times of observation windows are part of the windows.

    :param include_stop: Whether end times of observation windows are part of the windows.

    :param optimize: Whether to optimize runtime or memory requirements. If set to "time", the function returns faster
    but requires more memory; if set to "memory", the runtime is longer but memory consumption is reduced to a minimum.
    If "time", global variable `MAX_ROWS` can be used to adjust the time-memory tradeoff: increasing it increases
    memory consumption while reducing runtime.
    Note that this parameter is only relevant for computing non-rank-like aggregations, since rank-like aggregations
    ("rxx", "txx") can be efficiently computed anyway.

    :return: Resampled data. Like `windows`, but with one additional column for each requested aggregation.
    Order of columns is arbitrary, order of rows is exactly as in `windows` -- unless `windows` is a Dask DataFrame, in
    which case the order of rows may differ.
    The output is a (lazy) Dask DataFrame if `windows` is a Dask DataFrame, and a Pandas DataFrame otherwise,
    regardless of what `df` is.
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
    assert (time_col, 'start') not in windows.columns or windows[(time_col, 'start')].dtype == df[time_col].dtype
    assert (time_col, 'stop') not in windows.columns or windows[(time_col, 'stop')].dtype == df[time_col].dtype
    assert entity_col is None or (entity_col, '') in windows.columns or entity_col == windows.index.name

    if agg:
        assert attribute_col is not None or len(agg) == 1
    else:
        # no aggregations specified => nothing to do
        return windows.copy()       # works if `windows` is a Dask DataFrame, too

    # get schema (columns and data types) of output
    init_values = []
    columns = []
    val_val = _get_default_value(df[value_col].dtype)
    time_val = _get_default_value(df[time_col].dtype)
    zero_val = np.array(0, dtype=np.float64)
    dummy_df = None

    # partition aggregations
    standard_agg = {}       # maps attributes to lists of standard aggregation functions
    custom_agg = {}         # maps attributes to lists of custom aggregation functions
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
            if isinstance(f, str):
                if len(f) > 1:
                    if f[0] == 'p':
                        try:
                            x = float(f[1:])
                            if 0. <= x <= 100.:
                                quant.append((0.01 * x, f))
                                init_values.append(val_val)
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
                            init_values.append(val_val if f[0] == 'r' else time_val)
                            columns.append((attr, f[0] + str(x)))
                            continue
                        except ValueError:
                            pass
                if f in ('count', 'size', 'nunique', 'mode_count'):
                    init_values.append(zero_val)
                else:
                    init_values.append(val_val)
                columns.append((attr, f))
                if f not in ('mode', 'mode_count'):
                    standard_agg.setdefault(attr, []).append(f)
            else:
                if dummy_df is None:
                    dummy_df = pd.DataFrame(
                        index=[0, 0, 0],
                        data={time_col: _get_values(df[time_col].dtype, n=3),
                              value_col: _get_values(df[value_col].dtype, n=3)}
                    )
                dummy_res = f(dummy_df)
                if isinstance(dummy_res, pd.Series):
                    init_values.append(_get_default_value(dummy_res.dtype))
                    columns.append((attr, dummy_res.name or f.__name__))
                else:
                    assert isinstance(dummy_res, pd.DataFrame), type(dummy_res)
                    init_values += [_get_default_value(dummy_res[_c].dtype) for _c in dummy_res.columns]
                    columns += [(attr, _c) for _c in dummy_res.columns]
                custom_agg.setdefault(attr, []).append(f)
        if quant:
            quantile_agg[attr] = quant
        if nn_rank:
            nn_rank_agg[attr] = nn_rank
        if neg_rank:
            neg_rank_agg[attr] = neg_rank

    # restrict `df` to relevant entries; works if `df` is a Dask DataFrame, too
    mask = ~df[time_col].isna()     # `.notna()` does not work with Dask DataFrames
    if attribute_col is not None:
        mask &= df[attribute_col].isin(agg.keys())
    df = df[mask]

    if not (isinstance(df, pd.DataFrame) and isinstance(windows, pd.DataFrame)):
        # Dask DataFrame
        assert isinstance(df, pd.DataFrame) or str(type(df)) == "<class 'dask.dataframe.core.DataFrame'>"
        assert isinstance(windows, pd.DataFrame) or str(type(windows)) == "<class 'dask.dataframe.core.DataFrame'>"
        if entity_col is None:
            raise NotImplementedError('When passing Dask DataFrames, an entity column must be specified.')
        elif len(columns) > len(set(columns)):
            raise ValueError('When passing Dask DataFrames, no duplicate aggregations may be specified.')

        if isinstance(df, pd.DataFrame):
            if entity_col != df.index.name:
                df = df.set_index(entity_col)
        else:
            if entity_col == df.index.name:
                if not df.known_divisions or df.npartitions + 1 != len(df.divisions):
                    # could happen if `df` is result of `groupby()` or similar operation
                    raise NotImplementedError('When passing a Dask DataFrame with entities on the row index,'
                                              ' its divisions must be known and equal to 1 + number of partitions.')
            else:
                if df[entity_col].dtype.kind not in 'uif':
                    raise ValueError('When passing Dask DataFrames, the entity column must have a numeric data type.')
                # this operation can be _very_ slow and memory intensive; it may even crash if `df` is too large
                df = df.set_index(entity_col)
        # `df` is either a Pandas DataFrame or a Dask DataFrame, but in either case has entities on the row index

        orig_index_name = None
        change_windows_inplace = False
        if isinstance(windows, pd.DataFrame):
            if entity_col != windows.index.name:
                if windows.index.nlevels > 1:
                    raise ValueError('When passing Dask DataFrames, `windows` must have 1 row index level.')
                orig_index_name = windows.index.name
                windows = windows.reset_index(drop=False)       # don't change `windows` in place
                orig_index_name = (orig_index_name, windows.columns[0])
                windows.index = windows[(entity_col, '')].values
                windows.index.name = entity_col
                change_windows_inplace = True

            # restrict `df` further
            df = df[df.index.to_series().isin(windows.index)]       # doesn't work without `.to_series()`
        else:
            if entity_col != windows.index.name:
                if getattr(windows.index, 'nlevels', 1) > 1:
                    raise ValueError('When passing Dask DataFrames, `windows` must have 1 row index level.')
                elif windows[(entity_col, '')].dtype.kind not in 'uif':
                    raise ValueError('When passing a Dask DataFrame, the entity column must have a numeric data type.')
                orig_index_name = windows.index.name
                windows = windows.reset_index(drop=False)
                orig_index_name = (orig_index_name, windows.columns[0])
                i = list(windows.columns).index((entity_col, ''))
                orig_columns = windows.columns
                windows.columns = pd.RangeIndex(len(windows.columns))
                # this operation can be costly
                windows = windows.set_index(i, drop=False)      # does not work with MultiIndex columns => use `i`
                windows.columns = orig_columns
                windows.index.name = entity_col
            elif not windows.known_divisions or windows.npartitions + 1 != len(windows.divisions):
                # could happen if `windows` is result of `groupby()` or similar operation
                raise NotImplementedError('When passing a Dask DataFrame with entities on the row index,'
                                          ' its divisions must be known and equal to 1 + number of partitions.')
        # `windows` is either a Pandas DataFrame or a Dask DataFrame, but in either case has entities on the row index
        # `orig_index_name` is either None or a tuple `(name, column_name)`

        if any(c in columns for c in windows.columns):
            raise ValueError('When passing Dask DataFrames, `windows` must not contain newly generated column names.')

        meta = {c: windows[c].dtype for c in windows.columns if orig_index_name is None or c != orig_index_name[1]}
        meta.update({c: getattr(d, 'dtype', 'object') for c, d in zip(columns, init_values)})

        if isinstance(windows, pd.DataFrame):
            if not change_windows_inplace:
                windows = windows.copy()
            windows[('__orig_order__', '')] = np.arange(len(windows))
            out = df.map_partitions(
                _resample_eav_pandas,
                windows,
                standard_agg=standard_agg,
                mode_agg=mode_agg,
                quantile_agg=quantile_agg,
                custom_agg=custom_agg,
                nn_rank_agg=nn_rank_agg,
                neg_rank_agg=neg_rank_agg,
                entity_col=entity_col,
                time_col=time_col,
                attribute_col=attribute_col,
                value_col=value_col,
                include_start=include_start,
                include_stop=include_stop,
                optimize=optimize,
                orig_index_name=orig_index_name,
                columns=columns,
                init_values=init_values,
                swap_df_windows=True,
                # Dask kwargs
                align_dataframes=True,
                enforce_metadata=False,
                meta=meta
            ).compute()
            out.sort_values([('__orig_order__', '')], inplace=True)
            out.drop([('__orig_order__', '')], axis=1, inplace=True)
        else:
            out = windows.map_partitions(
                _resample_eav_pandas,
                df,
                standard_agg=standard_agg,
                mode_agg=mode_agg,
                quantile_agg=quantile_agg,
                custom_agg=custom_agg,
                nn_rank_agg=nn_rank_agg,
                neg_rank_agg=neg_rank_agg,
                entity_col=entity_col,
                time_col=time_col,
                attribute_col=attribute_col,
                value_col=value_col,
                include_start=include_start,
                include_stop=include_stop,
                optimize=optimize,
                orig_index_name=orig_index_name,
                columns=columns,
                init_values=init_values,
                swap_df_windows=False,
                # Dask kwargs
                align_dataframes=True,
                meta=meta
            )
    else:
        if entity_col is not None:
            # set `entity_col` to row index of `df`
            if entity_col != df.index.name:
                df = df.set_index(entity_col)

            # restrict `df` further
            df = df[df.index.isin(windows.index if entity_col == windows.index.name else windows[(entity_col, '')])]

        out = _resample_eav_pandas(
            windows,
            df,
            standard_agg=standard_agg,
            mode_agg=mode_agg,
            quantile_agg=quantile_agg,
            custom_agg=custom_agg,
            nn_rank_agg=nn_rank_agg,
            neg_rank_agg=neg_rank_agg,
            entity_col=entity_col,
            time_col=time_col,
            attribute_col=attribute_col,
            value_col=value_col,
            include_start=include_start,
            include_stop=include_stop,
            optimize=optimize,
            columns=columns,
            init_values=init_values
        )

    return out


def resample_interval(
        df: Union[pd.DataFrame, 'dask.dataframe.DataFrame'], # noqa F821
        windows: Union[pd.DataFrame, 'dask.dataframe.DataFrame'], # noqa F821
        attributes: list = None, entity_col=None, start_col=None, stop_col=None, attribute_col=None,
        value_col=None, time_col=None, epsilon=1e-7
) -> Union[pd.DataFrame, 'dask.dataframe.DataFrame']: # noqa F821
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
        * If `v` is NA, `W_I` is set to 0.
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
    bool). Observation times and observed values may be NA, but such entries are ignored entirely.
    Although both `start_col` and `stop_col` are optional, at least one must be present. Missing start- and end
    columns are interpreted as -/+ inf.
    All intervals are closed, i.e., start- and end times are included. This is especially relevant for entries whose
    start time equals their end time.
    `df` can be a Dask DataFrame as well. In that case, however, `entity_col` must not be None and entities should
    already be on the row index, with known divisions. Otherwise, the row index is set to `entity_col`, which can be
    very costly both in terms of time and memory. Especially if `df` is known to be sorted wrt. entities already, the
    calling function should better take care of this; see
    https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.set_index.html.

    :param windows: The target windows into which `df` is resampled. Must have either one or two columns index level(s).
    If it has one column index level, must have columns `start_col` (optional; contains start times of each window),
    `stop_col` (optional; contains end times of each window) and `entity_col` (optional; contains entity identifiers).
    If it has two column index levels, the columns must be `(time_col, "start")`, `(time_col, "stop")` and
    `(entity_col, "")`. Start- and end times may be NA, but such windows are deemed invalid and by definition do not
    overlap with any observation intervals.
    At least one of the two endpoint-columns must be present; if one is missing it is assumed to represent -/+ inf.
    All time windows are closed, i.e., start- and end times are included.
    `windows` can be a Dask DataFrame as well. In that case, however, `entity_col` must not be None and entities should
    already be on the row index, with known divisions. Otherwise, the row index is set to `entity_col`, which can be
    very costly both in terms of time and memory. Especially if `windows` is known to be sorted wrt. entities already,
    the calling function should better take care of this; see
    https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.set_index.html.

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

    :return: Resampled data. Like `windows`, but with one additional column for each attribute, and same number of
    column index levels.
    Order of columns is arbitrary, order of rows is exactly as in `windows` -- unless `windows` is a Dask DataFrame, in
    which case the order of rows may differ.
    The output is a (lazy) Dask DataFrame if `windows` is a Dask DataFrame, and a Pandas DataFrame otherwise,
    regardless of what `df` is.
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
    assert start_col != stop_col
    assert start_col in df.columns or stop_col in df.columns
    assert attribute_col is None or attribute_col in df.columns
    if start_col in df.columns:
        time_dtype = df[start_col].dtype
        if stop_col in df.columns:
            assert df[stop_col].dtype == time_dtype
    else:
        assert stop_col in df.columns
        time_dtype = df[stop_col].dtype

    # restrict `df` to relevant entries
    mask = ~df[value_col].isna()        # `.notna()` does not work with Dask DataFrames
    if start_col in df.columns:
        mask &= ~df[start_col].isna()
        if stop_col in df.columns:
            mask &= ~df[stop_col].isna() & (df[start_col] <= df[stop_col])
    else:
        mask &= ~df[stop_col].isna()
    if attribute_col in df.columns:
        if attributes is None:
            # `df` must be a Pandas DataFrame
            attributes = df[attribute_col].unique()
        else:
            mask &= df[attribute_col].isin(attributes)
    else:
        if attributes is None:
            attributes = ['sum']
        else:
            assert len(attributes) == 1
    df = df[mask]
    # `attributes` is now an iterable with attribute identifiers

    if windows.columns.nlevels == 2:
        window_start_col = (time_col, 'start')
        window_stop_col = (time_col, 'stop')
        window_entity_col = (entity_col, '')
        window_order_col = ('__orig_order__', '')
        columns = [(a, '') for a in attributes]
    else:
        assert windows.columns.nlevels == 1
        window_start_col = start_col
        window_stop_col = stop_col
        window_entity_col = entity_col
        window_order_col = '__orig_order__'
        columns = list(attributes)
    assert window_start_col in windows.columns or window_stop_col in windows.columns
    assert window_start_col not in windows.columns or windows[window_start_col].dtype == time_dtype
    assert window_stop_col not in windows.columns or windows[window_stop_col].dtype == time_dtype

    if not (isinstance(df, pd.DataFrame) and isinstance(windows, pd.DataFrame)):
        # Dask DataFrame
        assert isinstance(df, pd.DataFrame) or str(type(df)) == "<class 'dask.dataframe.core.DataFrame'>"
        assert isinstance(windows, pd.DataFrame) or str(type(windows)) == "<class 'dask.dataframe.core.DataFrame'>"
        if entity_col is None:
            raise NotImplementedError('When passing Dask DataFrames, an entity column must be specified.')

        if isinstance(df, pd.DataFrame):
            if entity_col != df.index.name:
                df = df.set_index(entity_col)
        else:
            if attribute_col in df.columns and attributes is None:
                raise ValueError('When passing a Dask DataFrame with a column containing attribute identifiers,'
                                 ' the list of requested attributes must be specified explicitly.')
            if entity_col == df.index.name:
                if not df.known_divisions or df.npartitions + 1 != len(df.divisions):
                    # could happen if `df` is result of `groupby()` or similar operation
                    raise NotImplementedError('When passing a Dask DataFrame with entities on the row index,'
                                              ' its divisions must be known and equal to 1 + number of partitions.')
            else:
                if df[entity_col].dtype.kind not in 'uif':
                    raise ValueError('When passing Dask DataFrames, the entity column must have a numeric data type.')
                # this operation can be _very_ slow and memory intensive; it may even crash if `df` is too large
                df = df.set_index(entity_col)
        # `df` is either a Pandas DataFrame or a Dask DataFrame, but in either case has entities on the row index

        orig_index_name = None
        change_windows_inplace = False
        if isinstance(windows, pd.DataFrame):
            if entity_col != windows.index.name:
                if windows.index.nlevels > 1:
                    raise ValueError('When passing Dask DataFrames, `windows` must have 1 row index level.')
                orig_index_name = windows.index.name
                windows = windows.reset_index(drop=False)  # don't change `windows` in place
                orig_index_name = (orig_index_name, windows.columns[0])
                windows.index = windows[window_entity_col].values
                windows.index.name = entity_col
                change_windows_inplace = True

            # restrict `df` further
            df = df[df.index.to_series().isin(windows.index)]   # doesn't work without `.to_series()`
        else:
            if entity_col != windows.index.name:
                if getattr(windows.index, 'nlevels', 1) > 1:
                    raise ValueError('When passing Dask DataFrames, `windows` must have 1 row index level.')
                elif windows[window_entity_col].dtype.kind not in 'uif':
                    raise ValueError('When passing a Dask DataFrame, the entity column must have a numeric data type.')
                orig_index_name = windows.index.name
                windows = windows.reset_index(drop=False)
                orig_index_name = (orig_index_name, windows.columns[0])
                i = list(windows.columns).index(window_entity_col)
                orig_columns = windows.columns
                windows.columns = pd.RangeIndex(len(windows.columns))
                # this operation can be costly
                windows = windows.set_index(i, drop=False)      # does not work with MultiIndex columns => use `i`
                windows.columns = orig_columns
                windows.index.name = entity_col
            elif not windows.known_divisions or windows.npartitions + 1 != len(windows.divisions):
                # could happen if `windows` is result of `groupby()` or similar operation
                raise NotImplementedError('When passing a Dask DataFrame with entities on the row index,'
                                          ' its divisions must be known and equal to 1 + number of partitions.')
        # `windows` is either a Pandas DataFrame or a Dask DataFrame, but in either case has entities on the row index
        # `orig_index_name` is either None or a tuple `(name, column_name)`

        if any(c in columns for c in windows.columns):
            raise ValueError('When passing Dask DataFrames, `windows` must not contain newly generated column names.')

        meta = {c: windows[c].dtype for c in windows.columns if orig_index_name is None or c != orig_index_name[1]}
        meta.update({c: 'float' for c in columns})

        if isinstance(windows, pd.DataFrame):
            if not change_windows_inplace:
                windows = windows.copy()
            windows[window_order_col] = np.arange(len(windows))
            out = df.map_partitions(
                _resample_interval_pandas,
                windows,
                entity_col=entity_col,
                time_col=time_col,
                attribute_col=attribute_col,
                value_col=value_col,
                start_col=start_col,
                stop_col=stop_col,
                epsilon=epsilon,
                orig_index_name=orig_index_name,
                attributes=attributes,
                swap_df_windows=True,
                # Dask kwargs
                align_dataframes=True,
                enforce_metadata=False,
                meta=meta
            ).compute()
            out.sort_values([window_order_col], inplace=True)
            out.drop([window_order_col], axis=1, inplace=True)
        else:
            out = windows.map_partitions(
                _resample_interval_pandas,
                df,
                entity_col=entity_col,
                time_col=time_col,
                attribute_col=attribute_col,
                value_col=value_col,
                start_col=start_col,
                stop_col=stop_col,
                epsilon=epsilon,
                orig_index_name=orig_index_name,
                attributes=attributes,
                swap_df_windows=False,
                # Dask kwargs
                align_dataframes=True,
                meta=meta
            )
    else:
        if entity_col is not None:
            # set `entity_col` to row index of `df`
            if entity_col != df.index.name:
                assert entity_col in df.columns
                df = df.set_index(entity_col)

            # restrict `df` further
            df = df[df.index.isin(windows.index if entity_col == windows.index.name else windows[window_entity_col])]

        out = _resample_interval_pandas(
            windows,
            df,
            entity_col=entity_col,
            time_col=time_col,
            attribute_col=attribute_col,
            value_col=value_col,
            start_col=start_col,
            stop_col=stop_col,
            epsilon=epsilon,
            attributes=attributes
        )

    return out


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


def inner_or_cross_join(left: pd.DataFrame, right: pd.DataFrame, on=None) -> pd.DataFrame:
    """
    Return the inner join or cross join of two DataFrames, depending on whether to column to join on actually occurs
    in the DataFrames.
    :param left: The first DataFrame.
    :param right: The second DataFrame.
    :param on: The column to join on, or None.
    :return: If `on` is not None and occurs in `left`, return the inner join of `left` (column `on`) and
    `right` (row index).
    Otherwise, return the cross join of `left` and `right`; in that case , the index of the result corresponds to the
    (replicated) index of `left` and the index of `right` is completely ignored.

    Functionally, the cross join operation is equivalent to joining `left` and `right` on a constantly-valued column:
    >>> left["join_column"] = 0
    >>> right.index = 0
    >>> left.join(right, on="join_column").drop("join_column", axis=1)
    """
    assert not any(c in right.columns for c in left.columns), 'Columns of `left` and `right` are not disjoint.'
    if on is None or on not in left.columns:
        out = pd.DataFrame(index=np.tile(left.index, len(right)),
                           data={c: np.tile(left[c].values, len(right)) for c in left.columns})
        for c in right.columns:
            out[c] = np.repeat(right[c].values, len(left))
        return out
    else:
        return left.join(right, on=on, how='inner')


def _resample_eav_pandas(windows: pd.DataFrame, df: pd.DataFrame, standard_agg=None, mode_agg=None, quantile_agg=None,
                         custom_agg=None, nn_rank_agg=None, neg_rank_agg=None, entity_col=None, time_col=None,
                         attribute_col=None, value_col=None, include_start: bool = True, include_stop: bool = False,
                         optimize: str = 'time', orig_index_name=None, columns=None, init_values=None,
                         swap_df_windows: bool = False) -> pd.DataFrame:
    # `entity_col` is either None or the row index of `df`, and it may or may not be a column / the index of `windows`
    # `df` has been restricted to relevant entries
    # observation times in `df` are not NA
    # data types of `df` and `windows` have been checked
    # both `df` and `windows` may be empty
    # start- and end times in `windows` may be NA

    if swap_df_windows:
        # if called from `map_partitions()`, `windows` and `df` may be swapped
        df, windows = windows, df

    # initialize `out` => faster than successive concatenation of partial results
    out = pd.DataFrame(index=windows.index, data=dict(enumerate(init_values)))
    out.columns = pd.MultiIndex.from_tuples(columns)
    out = pd.concat([windows, out], axis=1, sort=False, join='inner')
    # order of rows in `out` never changes during execution

    if len(df) == 0 or len(out) == 0:
        # no observations or no windows => nothing to do
        if orig_index_name is not None:
            out.set_index(orig_index_name[1], inplace=True, drop=True)
            out.index.name = orig_index_name[0]
        return out

    to_drop = []
    if entity_col is not None and (entity_col, '') not in out.columns:
        to_drop.append((entity_col, ''))
        out[(entity_col, '')] = out.index

    # compute aggregations for which observations and windows do not need to be merged
    if neg_rank_agg:
        if (time_col, 'stop') not in out.columns:
            _resample_eav_ranks_2(df, out, neg_rank_agg, entity_col, time_col, attribute_col, value_col,
                                  include_start)
            neg_rank_done = True
        else:
            neg_rank_done = False
    else:
        neg_rank_done = True
    if nn_rank_agg:
        if (time_col, 'start') not in out.columns:
            _resample_eav_ranks_2(df, out, nn_rank_agg, entity_col, time_col, attribute_col, value_col,
                                  include_stop)
            nn_rank_done = True
        else:
            nn_rank_done = False
    else:
        nn_rank_done = True

    # return if all requested aggregations have been computed already
    if neg_rank_done and nn_rank_done and not (standard_agg or mode_agg or quantile_agg or custom_agg):
        if orig_index_name is not None:
            out.set_index(orig_index_name[1], inplace=True, drop=True)
            out.index.name = orig_index_name[0]
        out.drop(to_drop, axis=1, inplace=True, errors='ignore')
        return out

    # merge windows with observations
    df0 = df[[time_col]].copy()
    df0.reset_index(inplace=True, drop=entity_col is None)
    df0['__observation_idx__'] = np.arange(len(df0))
    df0['__window_idx__'] = -1.
    df1 = pd.DataFrame(data=dict(__window_idx__=np.arange(len(out), dtype=np.float64), __observation_idx__=-1))
    merged = None

    if neg_rank_agg and (time_col, 'start') in out.columns and (time_col, 'stop') in out.columns:
        # special case: sort by "stop" first to handle negative ranks
        df1[time_col] = out[(time_col, 'stop')].values
        sort_col = [time_col, '__window_idx__' if include_stop else '__observation_idx__']
        if entity_col is not None:
            df1[entity_col] = out[(entity_col, '')].values
            sort_col = [entity_col] + sort_col
        merged = pd.concat([df0, df1], axis=0, sort=False, ignore_index=True)
        merged['__is_observation__'] = merged['__observation_idx__'] >= 0
        merged.sort_values(sort_col, inplace=True, na_position='first')
        _resample_eav_ranks(df, out, merged, neg_rank_agg, entity_col, time_col, attribute_col,
                            value_col, include_start)
        neg_rank_done = True
        if not (nn_rank_agg or standard_agg or mode_agg or quantile_agg or custom_agg):
            if orig_index_name is not None:
                out.set_index(orig_index_name[1], inplace=True, drop=True)
                out.index.name = orig_index_name[0]
            out.drop(to_drop, axis=1, inplace=True, errors='ignore')
            return out

    # now proper sorting wrt. "start" if possible, otherwise "stop"
    if merged is None or (time_col, 'start') in out.columns:
        if (time_col, 'start') in out.columns:
            df1[time_col] = out[(time_col, 'start')].values
            sort_col = [time_col, '__observation_idx__' if include_start else '__window_idx__']
            fill_method = 'ffill'
            na_position = 'last'
        else:
            df1[time_col] = out[(time_col, 'stop')].values
            sort_col = [time_col, '__window_idx__' if include_stop else '__observation_idx__']
            fill_method = 'bfill'
            na_position = 'first'
        if entity_col is not None:
            df1[entity_col] = out[(entity_col, '')].values
            sort_col = [entity_col] + sort_col
        merged = pd.concat([df0, df1], axis=0, sort=False, ignore_index=True)
        merged['__is_observation__'] = merged['__observation_idx__'] >= 0
        merged.sort_values(sort_col, inplace=True, na_position=na_position)
    del df0, df1

    # compute remaining rank aggregations
    if neg_rank_agg and not neg_rank_done:
        _resample_eav_ranks(df, out, merged, neg_rank_agg, entity_col, time_col, attribute_col, value_col,
                            include_start)
    if nn_rank_agg and not nn_rank_done:
        _resample_eav_ranks(df, out, merged, nn_rank_agg, entity_col, time_col, attribute_col, value_col,
                            include_stop)

    if standard_agg or mode_agg or quantile_agg or custom_agg:
        # set `windows` to a simple DataFrame with columns `entity_col`, "__start__" and "__stop__" (if present)
        # row index is connected to `out` via `.iloc[]` and to `merged` via "__windows_idx__"
        if (entity_col, '') in out.columns:
            take_cols = [(entity_col, '')]
            new_cols = [entity_col]
            sort_col = [entity_col]
        else:
            take_cols = []
            new_cols = []
            sort_col = []
        if (time_col, 'start') in out.columns:
            take_cols.append((time_col, 'start'))
            new_cols.append('__start__')
            sort_col.append('__start__')
        else:
            sort_col.append('__stop__')
        if (time_col, 'stop') in out.columns:
            take_cols.append((time_col, 'stop'))
            new_cols.append('__stop__')
        windows = out[take_cols].copy()
        windows.columns = new_cols
        windows.reset_index(drop=True, inplace=True)
        windows.sort_values(sort_col, inplace=True)

        # restrict to non-empty windows, and discard windows with NA endpoints
        if '__start__' in windows.columns:
            if '__stop__' in windows.columns:
                if include_start and include_stop:
                    windows = windows[windows['__start__'] <= windows['__stop__']]
                else:
                    windows = windows[windows['__start__'] < windows['__stop__']]
            else:
                windows = windows[windows['__start__'].notna()]
                if not include_start:
                    try:
                        windows = windows[~np.isposinf(windows['__start__'])]
                    except TypeError:
                        pass
        else:
            windows = windows[windows['__stop__'].notna()]
            if not include_stop:
                try:
                    windows = windows[~np.isneginf(windows['__stop__'])]
                except TypeError:
                    pass
        # `windows` is a DataFrame view containing only non-empty windows, in canonical order
        # from here on, order of rows does not change anymore, but subsets may be taken

        # analyze windows
        window_pattern = _analyze_windows(windows, entity_col, '__start__', '__stop__',
                                          include_start and include_stop)

        # select strategy for each entity: full join (fast but memory intensive) or group (slow but memory efficient)
        if entity_col is None:
            if len(windows) == 1 or (window_pattern['overlapping'] and optimize == 'time'):
                    join_mask = np.ones((len(windows),), dtype='bool')
            else:
                join_mask = np.zeros((len(windows),), dtype='bool')
        else:
            join_mask = window_pattern['n'] == 1
            if optimize == 'time':
                join_mask |= window_pattern['overlapping']
            join_mask = windows[entity_col].isin(window_pattern[join_mask].index)

        if not join_mask.all():
            merged.loc[merged['__window_idx__'] < 0., '__window_idx__'] = np.nan
            not_join_mask = ~join_mask

            # restrict `merged` to relevant entries
            if attribute_col is not None:
                mask = \
                    df[attribute_col].isin(list(standard_agg) + list(quantile_agg) + list(mode_agg) + list(custom_agg))
                if not mask.all():
                    merged = \
                        merged[
                            merged['__observation_idx__'].isin(np.where(mask)[0]) | ~merged['__is_observation__']]

            # assign windows to groups s.t. all windows in same group are disjoint
            group = _group_windows(windows[not_join_mask], window_pattern, entity_col, '__start__', '__stop__',
                                   include_start and include_stop)

            # Maybe regular windows can be treated more efficiently. But be careful:
            # - there may be gaps between consecutive windows,
            # - windows may have 0 length if `include_start` and `include_stop` are True,
            # - the offset may be 0, too, if all windows have the same start time (and hence are identical),
            # - the assignment of observations to windows depends on `include_start` and `include_stop`.

            # iterate over groups of disjoint windows
            for g in group.unique():
                group_mask = group == g
                group_merged = merged[merged['__window_idx__'].isin(windows[not_join_mask][group_mask].index) |
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
                if '__start__' in windows.columns and '__stop__' in windows.columns:
                    # otherwise, if only "__stop__" occurs in `windows`, `merged` has been sorted wrt. "stop" already
                    # => nothing to do
                    group_merged = \
                        group_merged.join(windows.loc[not_join_mask, '__stop__'], on='__window_idx__', how='left')
                    if include_stop:
                        mask &= group_merged[time_col] <= group_merged['__stop__']
                    else:
                        mask &= group_merged[time_col] < group_merged['__stop__']
                group_merged = group_merged[mask]
                # `group_merged` has columns "__observation_idx__" and "__window_idx__", mapping former to latter;
                # all entries are valid, i.e., >= 0; not all windows may appear

                df0 = df.iloc[group_merged['__observation_idx__'].values].copy()
                df0.index = group_merged['__window_idx__'].values

                # resample `df0`, using windows as new entities
                _resample_eav_no_windows(df0, out, standard_agg, mode_agg, quantile_agg, custom_agg,
                                         attribute_col, time_col, value_col)

        if join_mask.any():
            # restrict `df` to relevant entries
            # Note that this must happen _after_ the above 'grouping' code path, because otherwise
            # "__observation_idx__" does not match the rows of `df` anymore!
            if attribute_col is not None:
                df = df[df[attribute_col].isin(list(standard_agg) + list(quantile_agg) + list(mode_agg) +
                                               list(custom_agg))]

            max_rows = max(len(df), MAX_ROWS)  # maximum number of rows of DataFrame to process at once
            n_to_join = join_mask.sum()
            if entity_col is None:
                if n_to_join * len(df) <= max_rows:
                    partition = np.zeros((n_to_join,), dtype=np.int8)
                else:
                    partition = np.arange(n_to_join) % (max_rows // len(df))
            else:
                # `n_rows` contains the number of rows of `windows[join_mask].join(df, on=entity_col, how='inner')`
                # for each entity
                n_rows = df.index.value_counts().to_frame(name=entity_col + '_') \
                    .join(windows.loc[join_mask, entity_col].value_counts().to_frame(name=entity_col), how='inner') \
                    .prod(axis=1)
                if n_rows.sum() <= max_rows:
                    partition = np.zeros((n_to_join,), dtype=np.int8)
                else:
                    partition = partition_series(n_rows, max_rows) \
                        .reindex(windows.loc[join_mask, entity_col], fill_value=0).values

            start_comp = '__ge__' if include_start else '__gt__'
            stop_comp = '__le__' if include_stop else '__lt__'
            for g in range(partition.max() + 1):
                df0 = inner_or_cross_join(windows[join_mask][partition == g], df, on=entity_col)
                if '__start__' in df0.columns:
                    mask = getattr(df0[time_col], start_comp)(df0['__start__'])
                else:
                    # in this case "__stop__" needs to occur in `df0` => `mask` is set to proper Series below
                    mask = True
                if '__stop__' in df0.columns:
                    mask = getattr(df0[time_col], stop_comp)(df0['__stop__']) & mask

                # resample `df0[mask]`, using windows as new entities
                _resample_eav_no_windows(df0[mask], out, standard_agg, mode_agg, quantile_agg, custom_agg,
                                         attribute_col, time_col, value_col)

    if orig_index_name is not None:
        out.set_index(orig_index_name[1], inplace=True, drop=True)
        out.index.name = orig_index_name[0]
    out.drop(to_drop, axis=1, inplace=True, errors='ignore')
    return out


def _resample_interval_pandas(windows: pd.DataFrame, df: pd.DataFrame, entity_col=None, time_col=None,
                              attribute_col=None, value_col=None, start_col=None, stop_col=None, epsilon=1e-7,
                              orig_index_name=None, attributes=None, swap_df_windows: bool = False) -> pd.DataFrame:
    # `entity_col` is either None or the row index of `df`, and it may or may not be a column / the index of `windows`
    # `df` has been restricted to relevant entries
    # start- and end times in `df` are not NA
    # data types of `df` and `windows` have been checked
    # both `df` and `windows` may be empty
    # `windows` may have 1 or 2 column index levels
    # start- and end times in `windows` may be NA

    if swap_df_windows:
        # if called from `map_partitions()`, `windows` and `df` may be swapped
        df, windows = windows, df

    window_start_col = 'window_' + ('start' if start_col is None else start_col)
    window_stop_col = 'window_' + ('stop' if stop_col is None else stop_col)
    if window_start_col == window_stop_col:
        window_stop_col = window_stop_col + '_'
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

    if orig_index_name is not None:
        windows_orig.set_index(orig_index_name[1], inplace=True, drop=True)
        windows_orig.index.name = orig_index_name[0]

    if entity_col is None or entity_col in windows.columns:
        windows.reset_index(drop=True, inplace=True)
    else:
        windows.reset_index(inplace=True)

    if window_start_col in windows.columns:
        mask = windows[window_start_col].notna()
    else:
        mask = True
    if window_stop_col in windows.columns:
        mask = windows[window_stop_col].notna() & mask
    windows = windows[mask]

    if df.empty or windows.empty:
        out = pd.DataFrame(index=windows.index, columns=attributes, data=0, dtype=np.float32)
    elif (start_col in df.columns or window_start_col in windows.columns) \
            and (stop_col in df.columns or window_stop_col in windows.columns):
        max_rows = max(len(df), MAX_ROWS)  # maximum number of rows of DataFrame to process at once
        if entity_col is None:
            if len(windows) * len(df) <= max_rows:
                partition = np.zeros((len(windows),), dtype=np.int8)
            else:
                partition = np.arange(len(windows)) % (max_rows // len(df))
        else:
            # `n_rows` contains the number of rows of `windows.join(df, on=entity_col, how='inner')` for each entity
            n_rows = df.index.value_counts().to_frame(name=entity_col + '_') \
                .join(windows[entity_col].value_counts(), how='inner').prod(axis=1)
            if n_rows.sum() <= max_rows:
                partition = np.zeros((len(windows),), dtype=np.int8)
            else:
                partition = partition_series(n_rows, max_rows).reindex(windows[entity_col], fill_value=0).values

        df_cols = [c for c in (start_col, stop_col, value_col, attribute_col) if c in df.columns]

        # the following could easily be parallelized
        dfs = [
            _resample_interval_aux(inner_or_cross_join(windows[partition == g], df[df_cols], on=entity_col),
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
                df = df.set_index(attribute_col, append=True)
            s = df.groupby(level=list(range(df.index.nlevels)))[value_col].sum()
            if s.index.nlevels == 1:
                s = s.to_frame(attributes[0])
            else:
                s = s.unstack(level=-1, fill_value=0)
            out = s.reindex(windows[entity_col], fill_value=0)
            out.index = windows.index

    # `out` is a DataFrame with one column per attribute

    # make sure that all requested attributes appear in `out`
    out = out.reindex(attributes, axis=1, fill_value=0)

    # make sure that all rows appear in `out` in the right order
    if mask.all():
        out.sort_index(inplace=True)    # to be on the safe side
    else:
        out = out.reindex(np.arange(len(windows_orig)), axis=0, fill_value=0)

    if windows_orig.columns.nlevels == 2:
        out.columns = pd.MultiIndex.from_product([out.columns, ['']])
    out.index = windows_orig.index
    return pd.concat([windows_orig, out], axis=1, sort=False)


def _get_default_value(dtype):
    if dtype.name == 'category':
        # Using a singleton Categorical to initialize a DataFrame with `n > 1` rows does not work in pandas <= 0.24.2,
        # but in pandas >= 1.4.1.
        # Using a singleton Categorical to initialize a Series with `n > 1` rows does not work in pandas <= 1.4.1.
        return pd.Categorical.from_codes([-1], dtype=dtype)
    elif dtype.kind in 'mMf':
        return np.array(None, dtype=dtype)
    elif dtype.kind in 'iu':
        return np.array(None, dtype=np.float64)
    elif dtype.kind == 'b':
        return np.array(None, dtype=np.float32)
    return None


def _get_values(dtype, n: int = 3):
    if dtype.name == 'category':
        return pd.Categorical.from_codes([0] * n, dtype=dtype)
    elif dtype.kind == 'm':
        return pd.to_timedelta(np.arange(n, dtype=np.int64), unit='m')
    elif dtype.kind == 'M':
        return pd.Timestamp.now() + pd.to_timedelta(np.arange(n, dtype=np.int64), unit='m')
    elif dtype.kind in 'iuf':
        return np.arange(n, dtype=dtype)
    elif dtype.kind == 'b':
        return np.zeros(n, dtype=dtype)
    return [None] * n


def _group_windows(windows: pd.DataFrame, window_pattern: Union[pd.DataFrame, dict], entity_col, start_col, stop_col,
                   include_both_endpoints: bool) -> pd.Series:
    # assumes `windows` is sorted wrt. `entity_col` (if present) and either `start_col` (if present) or `stop_col`
    # assumes `windows` has one column index level
    # `windows` is left unchanged (can be view)
    # assumes `window_pattern` is dict iff `entity_col` is None
    # assumes `window_pattern` has entities on row index, if DataFrame
    # output Series has same row index as `windows`

    if start_col in windows.columns and stop_col in windows.columns:
        out = pd.Series(index=windows.index, data=0, dtype=np.int32)

        overlap_mask = False
        pre_group = windows_grouped = None
        if entity_col is None:
            if not window_pattern['overlapping']:
                # no overlapping windows => nothing to do
                pass
            elif window_pattern['regular']:
                out.values[:] = np.arange(len(windows), dtype=out.dtype) % window_pattern['n_shifts_regular']
            else:
                overlap_mask = np.ones((len(windows),), dtype='bool')      # don't set to True
                pre_group = _pregroup_windows(windows, entity_col, start_col, stop_col, include_both_endpoints)
                windows_grouped = windows.groupby(pre_group.values).agg({start_col: 'min', stop_col: 'max'})
                entity_col = '__entity__'
                windows_grouped[entity_col] = 0
        else:
            overlapping = window_pattern[window_pattern['overlapping'] & ~window_pattern['regular']].index
            regular_patt_mask = window_pattern['overlapping'] & window_pattern['regular']
            if len(overlapping) > 0:
                overlap_mask = windows[entity_col].isin(overlapping)
                pre_group = _pregroup_windows(windows[overlap_mask], entity_col, start_col, stop_col,
                                              include_both_endpoints)
                windows_grouped = windows[overlap_mask].groupby(pre_group.values)\
                    .agg({entity_col: 'first', start_col: 'min', stop_col: 'max'})
            if regular_patt_mask.any():
                regular_mask = windows[entity_col].isin(window_pattern[regular_patt_mask].index)
                s = pd.Series(index=windows.loc[regular_mask, entity_col].values, data=1).groupby(level=0).cumsum() - 1
                assert (s.index == windows.loc[regular_mask, entity_col].values).all()
                out.values[regular_mask] = (s % window_pattern['n_shifts_regular'].reindex(s.index)).values

        if np.any(overlap_mask):   # works if `overlap_mask` is bool
            # if offending entities exist, they must be processed manually => SLOW!
            # `windows_grouped` has `pre_group` on the row index and three columns `entity_col`, `start_col`
            # and `stop_col`
            # `len(pre_group)` equals `overlap_mask.sum()`

            grp_idx = 0
            windows_grouped['__grp__'] = 0
            windows_grouped['__i__'] = np.arange(len(windows_grouped))
            mask0 = np.ones((len(windows_grouped),), dtype=bool)
            comp = '__gt__' if include_both_endpoints else '__ge__'
            while mask0.any():
                indices = []
                sub = windows_grouped[mask0]
                mask1 = np.ones((mask0.sum(),), dtype=bool)
                while mask1.any():
                    cur = sub[mask1].groupby(entity_col)[[stop_col, '__i__']].first()
                    indices.append(cur['__i__'])
                    cur.columns = [stop_col, '__j__']
                    tmp = sub[[entity_col, start_col, '__i__']][mask1].join(cur, on=entity_col, how='left')
                    mask1 &= (tmp['__i__'] > tmp['__j__']) & (getattr(tmp[start_col], comp)(tmp[stop_col]))
                indices = np.concatenate(indices)
                windows_grouped['__grp__'].values[indices] = grp_idx
                grp_idx += 1
                mask0[mask0] &= ~sub['__i__'].isin(indices)
            # `windows_grouped` now has an additional column "__grp__"

            out.values[overlap_mask] = windows_grouped['__grp__'].reindex(pre_group.values, fill_value=0).values

        return out
    else:
        # replace `target_col` if it is already present
        if entity_col is None:
            return pd.Series(index=windows.index, data=np.arange(len(windows)))
        else:
            s = pd.Series(index=windows[entity_col].values, data=1).groupby(level=0).cumsum() - 1
            assert len(s) == len(windows)
            assert (s.index == windows[entity_col]).all()
            s.index = windows.index
            return s


def _check_disjoint(df: pd.DataFrame, entity_col, start_col, stop_col, include_both_endpoints: bool,
                    return_mask: bool = False) -> Union[bool, np.ndarray]:
    # assumes `df` is sorted wrt. `entity_col` and `start_col`
    # returns either mask or array with entities with overlapping windows

    if entity_col is None:
        mask = np.ones((len(df),), dtype='bool')
    else:
        mask = np.roll(df[entity_col].values, -1) == df[entity_col].values
    mask[-1:] = False       # works for empty array, too
    # `mask` now indicates whether next entry belongs to same entity

    next_start = np.roll(df[start_col].values, -1)
    if include_both_endpoints:
        mask &= (df[stop_col] >= next_start)
    else:
        mask &= (df[stop_col] > next_start)
    # `mask` now indicates whether next entry belongs to same entity and starts before current entry ends

    if return_mask:
        return mask
    elif entity_col is None:
        return mask.any()
    else:
        return df.loc[mask, entity_col].unique()


def _pregroup_windows(df: pd.DataFrame, entity_col, start_col, stop_col, include_both_endpoints: bool) -> pd.Series:
    # assumes `df` is sorted wrt. `entity_col` and `start_col`

    prev_stop = np.roll(df[stop_col].values, 1)
    if include_both_endpoints:
        mask = (df[start_col] <= prev_stop).values
    else:
        mask = (df[start_col] < prev_stop).values
    # `mask` indicates whether previous entry stops after current entry starts

    if entity_col is not None:
        mask |= np.roll(df[entity_col].values, 1) != df[entity_col].values
    mask[:1] = False        # works for empty array, too
    # `mask` now indicates whether previous entry stops after current entry starts,
    # or previous entry belongs to different entity

    return pd.Series(index=df.index, data=np.cumsum(mask))


def _analyze_windows(windows: pd.DataFrame, entity_col, start_col, stop_col, include_both_endpoints: bool) \
        -> Union[pd.DataFrame, dict]:
    # assumes `windows` is sorted wrt. `entity_col` (if present) and `start_col`
    # column index may have 1 or 2 levels
    # `windows` is left unchanged (may be view)

    if entity_col is None:
        if start_col in windows.columns and stop_col in windows.columns:
            duration = windows[stop_col] - windows[start_col]
            min_duration = duration.min()
            max_duration = duration.max()
            offset = np.roll(windows[start_col].values, -1) - windows[start_col]
            min_offset = offset.min()
            max_offset = offset.max()
            regular = (min_duration == max_duration) & (min_offset == max_offset)
            try:
                regular = regular and np.isfinite(max_duration)
                regular = regular and np.isfinite(max_offset)
            except:  # noqa
                pass
            if regular:
                try:
                    n_shifts = int(np.ceil(max_duration / max_offset))      # `offset` could be 0
                    if not include_both_endpoints and n_shifts * max_offset <= max_duration:
                        n_shifts += 1
                except (ZeroDivisionError, ValueError):
                    n_shifts = len(windows)
            else:
                n_shifts = 0
            # `n_shifts` indicates how often regular windows must be shifted to obtain a non-overlapping subsequence
            # if windows are non-overlapping anyway, `n_shifts` is 1

            overlapping = len(windows) > 1 and _check_disjoint(windows, entity_col, start_col, stop_col,
                                                               include_both_endpoints, return_mask=False)
            return dict(n=len(windows), first_start=windows[start_col].iloc[0], min_duration=min_duration,
                        max_duration=max_duration, min_offset=min_offset, max_offset=max_offset, regular=regular,
                        overlapping=overlapping, n_shifts_regular=n_shifts)
        else:
            return dict(n=len(windows), regular=False, overlapping=len(windows) > 1)
    else:
        if start_col in windows.columns and stop_col in windows.columns:
            duration = pd.Series(index=windows[entity_col], data=(windows[stop_col] - windows[start_col]).values)
            duration = duration.groupby(level=0).agg(['min', 'max'])
            duration.columns = ['min_duration', 'max_duration']
            mask = np.roll(windows[entity_col].values, -1) == windows[entity_col].values
            mask[-1] = False
            if mask.any():
                offset = pd.Series(index=windows[entity_col],
                                   data=np.roll(windows[start_col].values, -1) - windows[start_col].values)
                offset = offset[mask].groupby(level=0).agg(['min', 'max'])
                offset.columns = ['min_offset', 'max_offset']
                offset = offset.reindex(duration.index, fill_value=duration['max_duration'].iloc[0])
            else:
                # only one window per entity => we can safely copy durations
                offset = duration.copy()
                offset.columns = ['min_offset', 'max_offset']
            out = pd.concat([duration, offset], axis=1, sort=False)
            out['first_start'] = windows.groupby([entity_col])[[start_col]].min()[start_col]    # multi-index
            out['n'] = windows.groupby([entity_col]).size()
            regular = (out['min_duration'] == out['max_duration']) & (out['min_offset'] == out['max_offset'])
            try:
                regular &= np.isfinite(out['max_duration'])
                regular &= np.isfinite(out['max_offset'])
            except:     # noqa
                pass

            out['n_shifts_regular'] = 0
            if regular.any():
                n_shifts = np.ceil(out.loc[regular, 'max_duration'] / out.loc[regular, 'max_offset'])
                finite_mask = np.isfinite(n_shifts)
                n_shifts[~finite_mask] = out.loc[regular, 'n'][~finite_mask]
                n_shifts = n_shifts.astype(np.int32)
                if not include_both_endpoints:
                    n_shifts[
                        finite_mask & (n_shifts * out.loc[regular, 'max_offset'] <= out.loc[regular, 'max_duration'])
                    ] += 1
                out.loc[regular, 'n_shifts_regular'] = n_shifts

            out['regular'] = regular
            overlapping = _check_disjoint(windows, entity_col, start_col, stop_col,
                                          include_both_endpoints, return_mask=False)
            out['overlapping'] = out.index.isin(overlapping)
            return out
        else:
            out = windows.groupby([entity_col]).size().to_frame('n')
            out['regular'] = False
            out['overlapping'] = out['n'] > 1
            return out


def _resample_eav_no_windows(df: pd.DataFrame, out: pd.DataFrame, standard_agg: dict, mode_agg: dict,
                             quantile_agg: dict, custom_agg: dict, attribute_col, time_col, value_col) -> None:
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
        if not aux.empty:
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
        if not aux.empty:
            for i, (_, a) in enumerate(quantiles):
                out[(attr, a)].values[aux.index.values] = aux.iloc[:, i].values

    for attr, aggs in custom_agg.items():
        df0 = df.loc[attr_mask if attribute_col is None else (df[attribute_col] == attr), [time_col, value_col]]
        for func in aggs:
            aux = func(df0)
            if not aux.empty:
                if isinstance(aux, pd.Series):
                    out[(attr, aux.name or func.__name__)].values[aux.index.values] = aux.values
                else:
                    for c in aux.columns:
                        out[(attr, c)].values[aux.index.values] = aux[c].values


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
        comp = 'ge' if include_endpoint else 'gt'       # handles NA times correctly
        merged = merged.iloc[::-1]      # revert `merged`, s.t. we can pretend to extract _last_ observations
    else:
        # only negative ranks => `(time_col, "stop")` was used to sort `windows` in `merged`
        other_endpoint = (time_col, 'start')
        comp = 'le' if include_endpoint else 'lt'       # handles NA times correctly
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
            mask = ~getattr(start, comp)(aux[time_col])     # handles NA times correctly
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
    # assume that `entity_col` is None or on row index of `df`
    # assumes that windows in `out` are defined by single endpoint

    signs = list({r >= 0 for ranks in agg.values() for r in ranks})
    assert len(signs) == 1
    if signs[0]:
        other_endpoint = (time_col, 'stop')
        comp = 'le' if include_endpoint else 'lt'       # correctly handles NA times
    else:
        other_endpoint = (time_col, 'start')
        comp = 'ge' if include_endpoint else 'gt'       # correctly handles NA times
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
