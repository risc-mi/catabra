#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from catabra.util.preprocessing import MinMaxScaler, NumCatTransformer, OneHotEncoder


def make_standard_transformer() -> NumCatTransformer:
    """
    Construct a transformer that scales numerical and time-like columns to the range [0, 1], one-hot encodes
    categorical columns, and imputes missing numerical values with -1 (after scaling).
    :return: Instance of class `NumCatTransformer`.
    """
    return NumCatTransformer(
        num_transformer=make_pipeline(
            MinMaxScaler(fit_bool=False),
            SimpleImputer(strategy='constant', fill_value=-1),
            'passthrough'
        ),
        cat_transformer=OneHotEncoder(drop_na=True),
        obj='drop',
        bool='num',     # cast to float by setting False to 0 and True to 1
        timedelta='[s]',
        timestamp='[s]'
    )
