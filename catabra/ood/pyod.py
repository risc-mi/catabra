import importlib
import inspect

import pandas as pd

from .utils import StandardTransformer
from .base import OODDetector


class PyODDetector(OODDetector):
    """
    class to transform pyod ood class into a OODDetector
    Requires sbo to be installed.
    """

    @property
    def pyod_detector(self):
        return self._pyod_detector

    def __init__(self, name: str, subset: float = 1, transformer=StandardTransformer, verbose=False, **kwargs):
        """
        :param: name: name of the module the detector class is in. Given in snake_case format.
        :param: subset: proportion of features to use  [0,1]
        :transformer: transformer to apply to data before fitting the detector
        :verbose:  whether to log the detection steps
        :kwargs: keyword arguments for the specific pyod detector
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
        return self._pyod_detector.predict_proba(X)
