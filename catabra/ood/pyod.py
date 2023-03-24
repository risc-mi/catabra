#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import importlib
import inspect

import pandas as pd

try:
    import pyod  # noqa F401
except ImportError:
    raise ImportError(
        'Package pyod is required for out-of-distribution detection with class PyODDetector. You can install it either'
        ' through pip (`pip install pyod`) or conda (`conda install -c conda-forge pyod`).'
        ' Visit https://github.com/yzhao062/pyod for details.'
    )

from catabra.ood.base import SamplewiseOODDetector
from catabra.ood.utils import make_standard_transformer


class PyODDetector(SamplewiseOODDetector):
    """
    class to transform pyod ood class into a OODDetector
    Requires sbo to be installed.
    """

    @property
    def pyod_detector(self):
        return self._pyod_detector

    def __init__(self, name: str, subset: float = 1, transformer=make_standard_transformer, verbose=False, **kwargs):
        """
        :param: name: name of the module the detector class is in. Given in snake_case format.
        :param: subset: proportion of features to use  [0,1]
        :param transformer: transformer to apply to data before fitting the detector
        :param verbose:  whether to log the detection steps
        :param kwargs: keyword arguments for the specific pyod detector
        """
        super().__init__(subset=subset, verbose=verbose)
        # class paths are given in the form: pyod.detector_name.DetectorName
        module = importlib.import_module('pyod.models.' + name)
        module_classes = inspect.getmembers(module, inspect.isclass)
        pyod_class = next(
            class_ for class_name, class_ in module_classes if class_name.lower() == name.replace('_', '')
        )
        self._pyod_detector = pyod_class(**kwargs) if len(kwargs) > 0 else pyod_class()
        self._transformer = transformer()

    def _transform(self, X: pd.DataFrame):
        return self._transformer.transform(X)

    def _fit_transformer(self, X: pd.DataFrame, verbose=True):
        self._transformer.fit(X)

    def _fit_transformed(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        self._pyod_detector.fit(X)

    def _predict_transformed(self, X):
        return self._pyod_detector.predict(X)

    def _predict_proba_transformed(self, X):
        proba = self._pyod_detector.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] > 1:
            proba = proba[:, -1]    # only probability of positive=OOD class
        return proba
