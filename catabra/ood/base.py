#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import abc
import importlib
import inspect
from abc import abstractmethod
from enum import Enum

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
        if kwargs is None:
            kwargs = {}

        if source == 'internal':
            module = importlib.import_module('catabra.ood.internal.' + name)
            module_classes = inspect.getmembers(module, inspect.isclass)
            ood_class = next(
                class_ for class_name, class_ in module_classes if class_name.lower() == name.replace('_', '')
            )
            ood = ood_class(**kwargs)
            if ood is None:
                raise ValueError(name + ' is not a valid OODDetector.')

        elif source == 'pyod':
            from catabra.ood.pyod import PyODDetector
            ood = PyODDetector(name, **kwargs)

        elif source == 'external':
            path_split = name.split('.')
            module_name = '.'.join(path_split[:-1])
            class_name = path_split[-1]
            module = importlib.import_module(module_name)
            module_classes = inspect.getmembers(module, inspect.isclass)
            ood_class = next(class_ for cn, class_ in module_classes if cn == class_name)
            ood = ood_class(**kwargs)

        else:
            raise ValueError(source + ' is not a valid OOD source.')

        return ood

    def __init__(self, subset: float = 1, random_state=None, verbose=False):
        super().__init__()
        self._subset = subset
        self._X: pd.DataFrame = None
        self._verbose = verbose,
        self._random_state = np.random.randint(1000) if random_state is None else random_state
        if verbose:
            logging.log('Initialized out-of-distribution detector of type ' + self.__class__.__name__)

    @property
    def subset(self):
        return self._subset

    @property
    def random_state(self):
        return self._random_state

    @property
    def verbose(self):
        return self._verbose

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
            np.random.RandomState(self._random_state)
            indices = np.random.choice(X.shape[0], np.round(X.shape[0] * self._subset).astype(int))
            X_fit = X_fit.iloc[indices,:]
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
        """
        Get o.o.d. probabilities of the given samples. Note that despite its name, this function does not necessarily
        return probabilities between 0 and 1, but in any case larger values correspond to an increased likelihood of
        being o.o.d.
        :param X: The data to analyze.
        :return: O.o.d. probabilities.
        """
        X_trans = self._transform(X)
        logging.log('Predicting out-of-distribution samples.')
        return self._predict_proba_transformed(X_trans)

# ----------------------------------------------------------------------------------------------------------------------
# Types of OOD detectors

# Subclasses defining return type of
# TODO: add pytest for returning expected shape

class SamplewiseOODDetector(OODDetector, abc.ABC):

    def _predict_transformed(self, X: pd.DataFrame) -> pd.Series:
        pass

    def _predict_proba_transformed(self, X: pd.DataFrame) -> pd.Series:
        pass


class FeaturewiseOODDetector(OODDetector, abc.ABC):

    def _predict_transformed(self, X: pd.DataFrame) -> pd.Series:
        pass

    def _predict_proba_transformed(self, X: pd.DataFrame) -> pd.Series:
        pass


class OverallOODDetector(OODDetector, abc.ABC):

    def _predict_transformed(self, X: pd.DataFrame) -> int:
        pass

    def _predict_proba_transformed(self, X: pd.DataFrame) -> float:
        pass


