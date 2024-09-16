#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.


import pandas as pd
from catabra_pandas import (
    group_intervals,
    make_windows,
    prev_next_values,
    resample_eav,
    resample_interval,
)


def group_temporal(df: pd.DataFrame, group_by=None, time_col=None, start_col=None, stop_col=None, distance=None,
                   inclusive: bool = True) -> pd.Series:
    # legacy alias for backward compatibility
    # note that both function- and argument names changed
    return group_intervals(df, group_by=group_by, point_col=time_col, start_col=start_col, stop_col=stop_col,
                           distance=distance, inclusive=inclusive)


__all__ = ['resample_eav', 'resample_interval', 'make_windows', 'group_temporal', 'prev_next_values']
