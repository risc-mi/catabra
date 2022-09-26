from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


class StandardTransformer(BaseEstimator, TransformerMixin):
    """
    Scales numeric columns to the range [0,1]
    In case of categorical columns they are one-hot encoded and scaled to range [0, feature_weights[col] / # unique values]
    Missing values are replaced by -1
    """

    def __init__(self):
        super().__init__()
        self._cat_features = []
        self._num_features = []
        self._scaler = MinMaxScaler()

    def fit(self, X: pd.DataFrame):
        self._columns = X.columns
        self._num_features = X.select_dtypes('number').columns
        self._cat_features = X.select_dtypes('category').columns

        self._scaler.fit(X[self._num_features])

    def transform(self, X: Union[np.ndarray, pd.DataFrame]):
        X_trans = X.copy(deep=True)
        if not isinstance(X, pd.DataFrame):
            X_trans = pd.DataFrame(X, columns=self._columns)

        X_trans[self._num_features] = self._scaler.transform(X_trans[self._num_features])
        if len(self._cat_features) > 0:
            dummies = pd.get_dummies(X[self._cat_features])
            X_trans.drop(self._cat_features, axis=1, inplace=True)
            X_trans[dummies.columns] = dummies
        X_trans = X_trans.fillna(-1)
        return X_trans
