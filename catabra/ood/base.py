import abc
import importlib
import inspect
from abc import abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from ..util import logging


class OODDetector(BaseEstimator, ClassifierMixin, abc.ABC):
    """
    Base class for out-of-distribution detection
    """

    @classmethod
    def create(cls, name: str, source: str = 'internal', kwargs=None) -> 'OODDetector':
        """
        factory method for creating OODDetector subclasses or PyOD classes from strings
        :param name: if source is 'internal' name of OODDetector module in snake_case;
                     if source is 'pyod' name of pyod detector module in snake_case;
                     if source is 'external' full path to the OODDetector (module1.module2.CustomOOD)
        :param source: whether to use internal class (from CaTaBra) or classes from pyod. ['internal, 'pyod']
        :param kwargs: keyword arguments for the detector class
        """
        if source == 'internal':
            module = importlib.import_module('catabra.ood.internal.' + name)
            module_classes = inspect.getmembers(module, inspect.isclass)
            ood_class = next(
                class_ for class_name, class_ in module_classes if class_name.lower() == name.replace('_', '')
            )
            ood = ood_class(**kwargs) if kwargs is not None and len(kwargs) > 0 else ood_class()
            if ood is None:
                raise ValueError(name + ' is not a valid OODDetector.')

        elif source == 'pyod':
            from catabra.ood.pyod import PyODDetector
            ood = PyODDetector(name, kwargs=kwargs)

        elif source == 'external':
            path_split = name.split('.')
            module_name = '.'.join(path_split[:-1])
            class_name = path_split[-1]
            module = importlib.import_module('.'.join(module_name))
            module_classes = inspect.getmembers(module, inspect.isclass)
            ood_class = next(class_ for cn, class_ in module_classes if cn == class_name)
            ood = ood_class(**kwargs) if kwargs is not None and len(kwargs) > 0 else ood_class()

        else:
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
        X_fit = X.copy(deep=True)
        if self._subset < 1:
            X_fit = np.random.choice(X, np.round(X.shape[0] * self._subset))
        if self._verbose:
            logging.log('Fitting out-of-distribution detector...')
        self._fit_transformer(X)
        X_fit = self._transform(X_fit.copy(deep=True))
        self._fit_transformed(X_fit, y)
        if self._verbose:
            logging.log('Out-of-distribution detector fitted.')

    @abstractmethod
    def _predict_transformed(self, X):
        pass

    def predict(self, X):
        X_trans = self._transform(X)
        return self._predict_transformed(X_trans)

    @abstractmethod
    def _predict_proba_transformed(self, X):
        pass

    def predict_proba(self, X):
        X_trans = self._transform(X)
        logging.log('Predicting out-of-distribution samples.')
        return self._predict_proba_transformed(X_trans)




