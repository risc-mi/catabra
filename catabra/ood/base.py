import abc
from abc import abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from catabra.ood.pyod import PyODDetector
from catabra.util import logging


class OODDetector(BaseEstimator, ClassifierMixin, abc.ABC):
    """
    Base class for out-of-distribution detection
    """

    @classmethod
    def create(cls, name: str, source: str = 'internal', **kwargs) -> 'OODDetector':
        """
        factory method for creating OODDetector subclasses or PyOD classes from strings
        @param name: if source is 'internal' name of OODDetector class;
                     if source is 'pyod' name of pyod detector in snake_case
        @param source: whether to use internal class (from CaTaBra) or classes from pyod. ['internal, 'pyod']
        @param kwargs: keyword arguments for the detector class
        """
        if source == 'internal':
            ood = next((sc for sc in cls.__subclasses__() if name.lower() == sc.__name__.lower()), None)
            if ood is None:
                raise ValueError(name + ' is not a valid OODDetector.')
            ood = ood(**kwargs)
        elif source == 'pyod':
            ood = PyODDetector(name, kwargs=kwargs)
        else:
            # TODO allow customs sources
            raise ValueError(source + 'is not a valid OOD source.')

        return ood

    def __init__(self, subset: float = 1, verbose=False):
        super().__init__()
        self._subset = subset
        self._X: pd.DataFrame = None
        self._verbose = verbose
        if verbose:
            logging.log('Initialized out-of-distribution detector of type ' + self.__class__.__name__)

    @abstractmethod
    def _transform(self, X: pd.DataFrame):
        pass

    @abstractmethod
    def _fit_transformer(self, X: pd.DataFrame):
        pass

    @abstractmethod
    def _fit_transformed(self, X: pd.DataFrame, y: pd.Series):
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series=None):
        if self._subset < 1:
            X = np.random.choice(X, np.round(X.shape[0] * self._subset))
        if self._verbose:
            logging.log('Fitting out-of-distribution detector...')
        self._fit_transformer(X)
        X = self._transform(X)
        self._fit_transformed(X, y)
        if self._verbose:
            logging.log('Out-of-distribution detector fitted.')

    @abstractmethod
    def _predict_transformed(self, X):
        pass

    def predict(self, X):
        X = self._transform(X)
        return self._predict_transformed(X)

    @abstractmethod
    def _predict_proba_transformed(self, X):
        pass

    def predict_proba(self, X):
        X = self._transform(X)
        logging.log('Predicting out-of-distribution samples.')
        return self._predict_proba_transformed(X)



