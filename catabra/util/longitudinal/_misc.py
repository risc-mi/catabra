#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

from typing import Union

import numpy as np
import pandas as pd


def group_temporal(df: pd.DataFrame, group_by=None, time_col=None, start_col=None, stop_col=None, distance=None,
                   inclusive: bool = True) -> pd.Series:
    """
    Group intervals wrt. their temporal distance to each other. Intervals can also be isolated points, i.e.,
    single-point intervals of the form `[x, x]`.

    Parameters
    ----------
    df: DataFrame
        DataFrame with intervals.
    group_by: optional
        Additional column(s) to group `df` by, optional. If given, the computed grouping refines the given one, in the
        sense that any two intervals belonging to the same computed group are guaranteed to belong to the same given
        group, too. Can be the name of a single column or a list of column names and/or row index levels. Strings are
        interpreted as column names or row index names, integers are interpreted as row index levels.
    time_col: str, optional
        Name of the column in `df` containing both start- and end times of single-point intervals. If given, both
        `start_col` and `stop_col` must be None.
    start_col: str, optional
        Name of the column in `df` containing start times of intervals. If given, `time_col` must be None.
    stop_col: str, optional
        Name of the column in `df` containing end times of intervals. If given, `time_col` must be None. Note that the
        function tacitly assumes that no interval ends before it starts, although this is not checked. If this
        assumption is violated, the returned results may not be correct.
    distance: optional
        Maximum allowed distance between two intervals for being put into the same group. Should be non-negative.
        The distance between two intervals is the single-linkage distance, i.e., the minimum distance between any two
        points in the respective intervals. This means, for example, that the distance between overlapping intervals is
        always 0.
    inclusive: bool, default=False
        Whether `distance` is inclusive.

    Notes
    -----
    The returned grouping is the reflexive-transitive closure of the proximity relation induced by `distance`.
    Formally: Let :math:`R` be the binary relation on the set of intervals in `df` such that :math:`R(I_1, I_2)` holds
    iff the distance between :math:`I_1` and :math:`I_2` is less than (or equal to) `distance` (and additionally
    :math:`I_1` and :math:`I_2` belong to the same groups specified by `group_by`). :math:`R` is obviously symmetric,
    so its reflexive-transitive closure :math:`R^*` is an equivalence relation on the set of intervals in `df`. The
    returned grouping corresponds precisely to this equivalence relation, in the sense that there is one group per
    equivalence class and vice versa.
    Note that if two intervals belong to the same group, their distance may still be larger than `distance`.

    Returns
    -------
    Series
        Series with the same row index as `df`, in the same order, whose values are group indices.
    """

    assert distance is not None

    if time_col is None:
        assert start_col in df.columns
        if stop_col is None:
            stop_col = start_col
        else:
            assert stop_col in df.columns
    else:
        assert time_col in df.columns
        assert start_col is None
        assert stop_col is None
        start_col = time_col
        stop_col = time_col

    if group_by is None:
        group_by = []
    else:
        group_by = _parse_column_specs(df, group_by)

        if not all(isinstance(g, (str, tuple)) for g in group_by):
            # construct new DataFrame
            df_new = pd.DataFrame(
                data={
                    f'g{i}': df[g] if isinstance(g, (str, tuple)) else
                    (df.index.get_level_values(g) if isinstance(g, int) else g)
                    for i, g in enumerate(group_by)
                }
            )
            group_by = [f'g{i}' for i in range(len(group_by))]
            df_new['start'] = df[start_col].values
            if stop_col == start_col:
                stop_col = 'start'
            else:
                df_new['stop'] = df[stop_col].values
                stop_col = 'stop'
            start_col = 'start'
            df_new.index = df.index
            df = df_new

    sorting_col = '__sorting__'
    if df.columns.nlevels > 1:
        sorting_col = tuple([sorting_col] + [''] * (df.columns.nlevels - 1))
    assert sorting_col not in df.columns

    df[sorting_col] = np.arange(len(df))
    df_sorted = df.sort_values(group_by + [start_col])
    out = pd.Series(index=df_sorted[sorting_col].values, data=0, dtype=np.int64)
    df.drop([sorting_col], axis=1, inplace=True, errors='ignore')

    start = df_sorted[start_col].values
    if start_col == stop_col:
        stop = start
    elif group_by:
        stop = df_sorted.groupby(group_by)[stop_col].cummax()
        assert (stop.index == df_sorted.index).all()
        stop = stop.values
    else:
        stop = df_sorted[stop_col].cummax().values
    # `stop` is the per-group cumulative maximum of all previous interval ends.
    # This ensures that some kind of triangle inequality holds: if all interval endpoints are modified according to
    # this procedure and `I_1` and `I_2` are consecutive intervals, then there does not exist another interval `I_3`
    # with `dist(I_1, I_3) < dist(I_1, I_2)` and `dist(I_2, I_3) < dist(I_1, I_2)`. This property is crucial for
    # ensuring that the grouping indeed corresponds to the reflexive-transitive closure of the proximity relation.
    # If the complete linkage distance were used, no ordering of `df` could establish the above property.
    # Counterexample: I_1 = [0, 10], I_2 = [2, 9], I_3 = [3, 8]. No matter how `df` is sorted, `I_1` and `I_2` would
    # always end up next to each other, but `dist(I_1, I_3) = 7 < 9 = dist(I_1, I_2)` and
    # `dist(I_2, I_3) = 6 < 9 = dist(I_1, I_2)`.

    dist: np.ndarray = start - np.roll(stop, 1)
    if isinstance(distance, pd.Timedelta):
        distance = distance.to_timedelta64()    # cannot be compared to `dist` otherwise
    if inclusive:
        same_group_as_prev = dist <= distance
    else:
        same_group_as_prev = dist < distance

    if len(same_group_as_prev):
        for g in group_by:
            same_group_as_prev &= df_sorted[g].values == np.roll(df_sorted[g].values, 1)
        same_group_as_prev[0] = True
    out.values[:] = (~same_group_as_prev).cumsum()
    out.sort_index(inplace=True)
    out.index = df.index
    return out


def prev_next_values(df: pd.DataFrame, sort_by=None, group_by=None, columns=None, first_indicator_name=None,
                     last_indicator_name=None, keep_sorted: bool = False, inplace: bool = False) -> pd.DataFrame:
    """
    Find the previous/next values of some columns in DataFrame `df`, for every entry. Additionally, entries can be
    grouped and previous/next values only searched within each group.

    Parameters
    ----------
    df: DataFrame
        The DataFrame.
    sort_by: list | str, optional
        The column(s) to sort by. Can be the name of a single column or a list of column names and/or row index levels.
        Strings are interpreted as column names or row index names, integers are interpreted as row index levels.
        ATTENTION! N/A values in columns to sort by are not ignored; rather, they are treated in the same way as Pandas
        treats such values in `DataFrame.sort_values()`, i.e., they are put at the end.
    group_by: list | str, optional
        Column(s) to group `df` by, optional. Same values as `sort_by`.
    columns: dict
        A dict mapping column names to dicts of the form

        ::

            {
                "prev_name": <prev_name>,
                "prev_fill": <prev_fill>,
                "next_name": <next_name>,
                "next_fill": <next_fill>
            }

        `prev_name` and `next_name` are the names of the columns in the result, containing the previous/next values.
        If any of them is None, the corresponding previous/next values are not computed for that column.
        `prev_fill` and `next_fill` specify which values to assign to the first/last entry in every group, which does
        not have any previous/next values.
        Note that column names not present in `df` are tacitly skipped.
    first_indicator_name: str, optional
        Name of the column in the result containing boolean indicators whether the corresponding entries come first in
        their respective groups. If None, no such column is added.
    last_indicator_name: str, optional
        Name of the column in the result containing boolean indicators whether the corresponding entries come last in
        their respective groups. If None, no such column is added.
    keep_sorted: bool, default=False
        Whether to keep the result sorted wrt. `group_by` and `sort_by`. If False, the order of rows of the result is
        identical to that of `df`.
    inplace: bool, default=False
        If `True`, the new columns are added to `df`.

    Returns
    -------
    DataFrame
        The modified DataFrame if `inplace` is True, a DataFrame with the requested previous/next values otherwise.
    """

    if columns is None:
        columns = {}
    elif isinstance(columns, (list, np.ndarray)):
        columns = {k: dict(prev_name=f'{k}_prev', next_name=f'{k}_next') for k in columns if k in df.columns}
    elif isinstance(columns, dict):
        columns = {k: v for k, v in columns.items() if k in df.columns and ('prev_name' in v or 'next_name' in v)}
    elif columns in df.columns:
        columns = {columns: dict(prev_name=f'{columns}_prev', next_name=f'{columns}_next')}
    else:
        columns = {}

    if not (columns or first_indicator_name or last_indicator_name):
        return df if inplace else pd.DataFrame(index=df.index)

    assert sort_by is not None
    sort_by = _parse_column_specs(df, sort_by)
    assert len(sort_by) > 0

    if group_by is None:
        group_by = []
    else:
        group_by = _parse_column_specs(df, group_by)
        if any(s in group_by for s in sort_by):
            raise ValueError('sort_by and group_by must be disjoint.')

    sorting_col = '__sorting__'

    prev_mask = np.zeros(len(df), dtype=bool)       # True iff previous element belongs to different group
    if len(df) == 0:
        sorting = np.zeros(0, dtype=np.int32)
        df_sorted = df
    else:
        if all(isinstance(g, (str, tuple)) for g in group_by) and all(isinstance(s, (str, tuple)) for s in sort_by):
            if len(df) > 0:     # otherwise row index would be renamed to None
                if df.columns.nlevels > 1:
                    sorting_col_df = tuple([sorting_col] + [''] * (df.columns.nlevels - 1))
                else:
                    sorting_col_df = sorting_col
                assert sorting_col_df not in df.columns
                df[sorting_col_df] = np.arange(len(df))
            else:
                sorting_col_df = None
            if inplace and keep_sorted:
                df.sort_values(group_by + sort_by, inplace=True)
                df_sorted = df
            else:
                df_sorted = df.sort_values(group_by + sort_by)
            if len(df) > 0:
                sorting = df_sorted[sorting_col_df].values
                df.drop([sorting_col_df], axis=1, inplace=True)
            else:
                sorting = np.zeros(0, dtype=np.int32)
            df_aux = df_sorted
        else:
            # construct new DataFrame
            df_aux = pd.DataFrame(
                data={
                    k: df[c] if isinstance(c, (str, tuple))
                    else (df.index.get_level_values(c) if isinstance(c, int) else c)
                    for k, c in [('g' + str(i), g) for i, g in enumerate(group_by)] +
                                [('s' + str(i), s) for i, s in enumerate(sort_by)]
                }
            )
            df_aux[sorting_col] = np.arange(len(df))
            group_by = [f'g{i}' for i in range(len(group_by))]
            sort_by = [f's{i}' for i in range(len(sort_by))]
            df_aux.sort_values(group_by + sort_by, inplace=True)
            sorting = df_aux[sorting_col].values
            if inplace and keep_sorted:
                if df.columns.nlevels > 1:
                    sorting_col_df = tuple([sorting_col] + [''] * (df.columns.nlevels - 1))
                else:
                    sorting_col_df = sorting_col
                assert sorting_col_df not in df.columns
                df[sorting_col_df] = np.argsort(sorting)
                df.sort_values([sorting_col_df], inplace=True)
                df.drop([sorting_col_df], axis=1, inplace=True)
                # `df` is now sorted as `df_aux`
                df_sorted = df
            else:
                df_sorted = df.iloc[sorting]

        for g in group_by:
            prev_mask |= (df_aux[g].values != np.roll(df_aux[g].values, 1))
        prev_mask[0] = True

        if df_aux is not df_sorted:
            del df_aux

    next_mask = np.roll(prev_mask, -1)      # True iff next element belongs to different group
    new_columns = []
    for k, v in columns.items():
        col = v.get('prev_name')
        if col is not None:
            if len(df_sorted) == 0:
                s = pd.Series(index=df_sorted.index, data=df_sorted[k], name=col)
            else:
                s = pd.Series(index=df_sorted.index, data=np.roll(df_sorted[k], 1), name=col)
                s[prev_mask] = v.get('prev_fill', pd.Timedelta(None) if s.dtype.kind == 'm' else None)
            new_columns.append(s)

        col = v.get('next_name')
        if col is not None:
            if len(df_sorted) == 0:
                s = pd.Series(index=df_sorted.index, data=df_sorted[k], name=col)
            else:
                s = pd.Series(index=df_sorted.index, data=np.roll(df_sorted[k], -1), name=col)
                s[next_mask] = v.get('next_fill', pd.Timedelta(None) if s.dtype.kind == 'm' else None)
            new_columns.append(s)
    if first_indicator_name is not None:
        new_columns.append(pd.Series(index=df_sorted.index, data=prev_mask, name=first_indicator_name))
    if last_indicator_name is not None:
        new_columns.append(pd.Series(index=df_sorted.index, data=next_mask, name=last_indicator_name))

    if new_columns:
        if inplace:
            if keep_sorted:
                for s in new_columns:
                    df[s.name] = s
            else:
                sorting_inv = np.argsort(sorting)
                for s in new_columns:
                    df[s.name] = s.iloc[sorting_inv]
            out = df
        else:
            out = pd.concat(new_columns, axis=1, sort=False)
            if len(out) > 0 and not keep_sorted:
                if out.columns.nlevels > 1:
                    sorting_col_out = tuple([sorting_col] + [''] * (out.columns.nlevels - 1))
                else:
                    sorting_col_out = sorting_col
                assert sorting_col_out not in out.columns
                out[sorting_col_out] = sorting
                out.sort_values([sorting_col_out], inplace=True)
                out.drop([sorting_col_out], axis=1, inplace=True)
    elif inplace:
        out = df
    elif keep_sorted:
        out = pd.DataFrame(index=df_sorted.index)
    else:
        out = pd.DataFrame(index=df.index)

    return out


def _parse_column_specs(df: Union[pd.DataFrame, 'dask.dataframe.DataFrame'], spec) -> list: # noqa F821
    if isinstance(spec, (tuple, str, int, np.ndarray, pd.Series)):
        spec = [spec]
    out = []
    for s in spec:
        if isinstance(s, int):
            if s < 0:
                s += df.index.nlevels
            assert 0 <= s < df.index.nlevels
        elif isinstance(s, str):
            if s not in df.columns:
                s = list(df.index.names).index(s)
        elif isinstance(s, tuple):
            assert s in df.columns
        elif isinstance(df, pd.DataFrame):
            assert len(s) == len(df)
        out.append(s)

    return out
