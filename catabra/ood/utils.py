from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

from ..util.preprocessing import NumCatTransformer, OneHotEncoder


def make_standard_transformer() -> NumCatTransformer:
    """
    Construct a transformer that scales numerical and time-like columns to the range [0, 1], one-hot encodes
    categorical columns, and imputes missing numerical values with -1 (after scaling).
    :return: Instance of class `NumCatTransformer`.
    """
    return NumCatTransformer(
        num_transformer=make_pipeline(
            MinMaxScaler(),
            SimpleImputer(strategy='constant', fill_value=-1),
            'passthrough'
        ),
        cat_transformer=OneHotEncoder(drop_na=True),
        obj='passthrough',
        bool='passthrough',
        timedelta='[s]',
        timestamp='[s]'
    )
