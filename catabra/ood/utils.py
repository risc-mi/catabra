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
        self._num_features = X.select_dtypes('number').columns
        self._cat_features = X.select_dtypes('category').columns

        self._scaler.fit(X[self._num_features])

    def transform(self, X: pd.DataFrame):
        X[self._num_features] = self._scaler.transform(X[self._num_features])
        if len(self._cat_features) > 0:
            dummies = pd.get_dummies(X[self._cat_features])
            X.drop(self._cat_features, axis=1, inplace=True)
            X[dummies.columns] = dummies
        X = X.fillna(-1)
        return X

    @staticmethod
    def _scale_cat_column(self, col: pd.DataFrame, weight=1):
        dummies = pd.get_dummies(col)
        scaler = MinMaxScaler(feature_range=(0, 1. / dummies.shape[1]))
        return scaler.fit_transform(dummies) * weight