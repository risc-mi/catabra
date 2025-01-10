#  Copyright (c) 2022-2025. RISC Software GmbH.
#  All rights reserved.

from catabra_lib.preprocessing import DTypeTransformer, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


def make_standard_transformer() -> DTypeTransformer:
    """
    Construct a transformer that scales numerical and time-like columns to the range [0, 1], one-hot encodes
    categorical columns, and imputes missing numerical values with -1 (after scaling).

    Returns
    -------
    DTypeTransformer
        Instance of class `DTypeTransformer`.
    """
    return DTypeTransformer(
        num=make_pipeline(
            MinMaxScaler(fit_bool=False),
            SimpleImputer(strategy='constant', fill_value=-1),
            'passthrough'
        ),
        cat=OneHotEncoder(drop_na=True),
        obj='drop',
        bool='num',     # cast to float by setting False to 0 and True to 1
        timedelta='num',
        datetime='num',
        timedelta_resolution='s',
        datetime_resolution='s'
    )
