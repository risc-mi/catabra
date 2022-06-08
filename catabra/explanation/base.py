from typing import Optional, Callable
from pathlib import Path
import importlib
import numpy as np
import pandas as pd


class DataExplainer:

    def explain(self, x, y) -> np.ndarray:
        """
        Explain data locally by transforming it into an array of explanations.
        :param x: Features, array-like of shape `(n_samples, n_features)`.
        :param y: Labels, array-like of shape `(n_samples, n_labels)` or `(n_samples,)`.
        :return: Explanations, array of shape `(*dims, n_samples, n_features)`.
        """
        raise NotImplementedError()

    def explain_global(self, x, y) -> np.ndarray:
        """
        Explain data globally.
        :param x: Features, array-like of shape `(n_samples, n_features)`.
        :param y: Labels, array-like of shape `(n_samples, n_labels)` or `(n_samples,)`.
        :return: Explanations, array of shape `(*dims, n_features)`.
        """
        raise NotImplementedError()


class ModelExplainer:

    @classmethod
    def global_behavior(cls) -> dict:
        """
        Description of the behavior of method `explain_global()`, especially w.r.t. parameter `x`. Dict with keys
        * "accepts_x": True if `x` can be provided.
        * "requires_x": True if `x` must be provided. If False but "accepts_x" is True, the global behavior differs
            depending on whether `x` is provided. "requires_x" can only be True if "accepts_x" is True as well.
        * "mean_of_local": True if global explanations are the mean of the individual local explanations, if `x` is
            provided. If True, it might be better to call method `explain()` instead of `explain_global()`.
        :return: Dict, as described above.
        """
        raise NotImplementedError()

    def explain(self, x, jobs: int = 1, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Explain data locally by transforming it into an array of explanations.
        :param x: Features, array-like of shape `(n_samples, n_features)`. `n_features` must be the same as in the data
        this explainer instance was fitted on.
        :param jobs: The number of jobs to use, or -1 if all available processors shall be used.
        :param batch_size: Batch size, i.e., number of samples processed in parallel.
        :return: Explanations, array of shape `(*dims, n_samples, n_features)`.
        """
        raise NotImplementedError()

    def explain_global(self, x=None, jobs: int = 1, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Explain the model globally.
        :param x: Data used for explaining the model, optional, array-like of shape `(n_samples, n_features)`.
        Since global explanations are computed, some explainers might not require this parameter.
        :param jobs: The number of jobs to use, or -1 if all available processors shall be used.
        :param batch_size: Batch size, i.e., number of samples processed in parallel.
        :return: Explanations, array of shape `(*dims, n_features)`, where `n_features` are as in the data this
        explainer instance was fitted on.
        """
        raise NotImplementedError()


class TransformationExplainer:

    _factories = []

    @staticmethod
    def register_factory(name: str, func, errors: str = 'raise'):
        i = [j for j, (n, _) in enumerate(TransformationExplainer._factories) if name == n]
        if i:
            if errors == 'raise':
                raise ValueError(f'Transformation explainer factory with name "{name}" already exists.')
            elif errors == 'update':
                TransformationExplainer._factories[i[0]] = (name, func)
            elif errors == 'replace':
                del TransformationExplainer._factories[i[0]]
                TransformationExplainer._factories.insert(0, (name, func))
        else:
            TransformationExplainer._factories.insert(0, (name, func))

    @staticmethod
    def make(obj, params=None) -> Optional['TransformationExplainer']:
        if all(hasattr(obj, attr)
               for attr in ('fit', 'transform', 'fit_forward', 'forward', 'backward', 'backward_global')):
            return obj

        for _, func in TransformationExplainer._factories:
            out = func(obj, params=params)
            if out is not obj:
                return out

        raise RuntimeError(f'Object of type {type(obj)} cannot be converted into a transformation explainer.')

    def __init__(self, transformer=None, params=None):
        if params is not None:
            assert params.get('class_name', self.__class__.__name__) == self.__class__.__name__
        self._transformer = transformer

    @property
    def transformer(self):
        return self._transformer

    @property
    def params_(self) -> dict:
        """
        Get all params obtained from fitting the explainer to data in method `fit_forward()`, and which can be passed
        to `__init__()`.
        """
        return dict(class_name=self.__class__.__name__)

    def fit(self, x, y=None):
        # only to implement the standard sklearn API, which makes it possible to combine individual explainers in
        # pipelines and similar compound transformations
        raise RuntimeError(f'Method fit() of class {self.__class__.__name__} cannot be called.')

    def transform(self, x):
        return self._transformer.transform(x)

    def fit_forward(self, x, y):
        """
        Fit this explainer to training data, and transform the data by applying the underlying transformation.
        `forward()` is implicitly called on `x` as well, meaning that invoking `backward()` immediately afterwards is
        possible and refers to the given samples `x`.
        :param x: Features, array-like of shape `(n_samples, n_features_in)`.
        :param y: Labels, array-like of shape `(n_samples, n_labels)` or `(n_samples,)`.
        :return: The transformed features, array-like of shape `(n_samples, n_features_out)`.
        """
        raise NotImplementedError()

    def forward(self, x):
        """
        Transform `x` by applying the underlying transformation, and record all intermediate values needed for
        back-propagating explanations generated by downstream `DataExplainer` or `ModelExplainer` instances.
        :param x: Data to transform (and later explain), array-like of shape `(n_samples, n_features_in)`.
        `n_features_in` must be the same as in the data this explainer instance was fitted on.
        :return: Transformed data, array-like of shape `(n_samples, n_features_out)`.
        """
        raise NotImplementedError()

    def backward(self, s: np.ndarray) -> np.ndarray:
        """
        Back-propagate local explanations from output to input.
        :param s: Explanations (feature importance scores) generated downstream for the last `x` method `forward()` was
        applied to. Array of shape `(*dims, n_samples, n_features_out)`, where `n_samples` must be as in the last
        invocation of `forward()`.
        :return: Explanations, array of shape `(*dims, n_samples, n_features_in)`.

        **Note**: In contrast to method `forward()`, this method expects plain Numpy arrays as input and returns plain
        Numpy arrays.
        """
        raise NotImplementedError()

    def backward_global(self, s: np.ndarray) -> np.ndarray:
        """
        Back-propagate global explanations from output to input.
        :param s: Global explanations (feature importance scores) generated downstream by an instance of
        `DataExplainer` or `ModelExplainer`. Array of shape `(*dims, n_features_out)`.
        :return: Explanations, array of shape `(*dims, n_features_in)`.

        **Note**: In contrast to method `forward()`, this method expects plain Numpy arrays as input and returns plain
        Numpy arrays.
        """
        raise NotImplementedError()


class IdentityTransformationExplainer(TransformationExplainer):

    def __init__(self, transformer=None, params=None):
        super(IdentityTransformationExplainer, self).__init__(transformer=transformer, params=params)
        self._transform_func = getattr(self._transformer, 'transform', None)

    def transform(self, x):
        return x if self._transform_func is None else self._transform_func(x)

    def fit_forward(self, x, y):
        return self.forward(x)

    def forward(self, x):
        return self.transform(x)

    def backward(self, s):
        return s

    def backward_global(self, s):
        return s


class EnsembleExplainer:
    __registered = {}

    @staticmethod
    def register(name: str, factory: Callable[..., 'EnsembleExplainer']):
        """
        Register a new ensemble explainer factory.
        :param name: The name of the ensemble explainer.
        :param factory: The factory, a function mapping argument-dicts to instances of class `EnsembleExplainer` (or
        subclasses thereof).
        """
        EnsembleExplainer.__registered[name] = factory

    @staticmethod
    def get(name: str, **kwargs) -> Optional['EnsembleExplainer']:
        factory = EnsembleExplainer.__registered.get(name)
        return factory if factory is None else factory(**kwargs)

    @classmethod
    def name(cls) -> str:
        raise NotImplementedError()

    @classmethod
    def global_behavior(cls) -> dict:
        """
        Description of the behavior of method `explain_global()`, especially w.r.t. parameter `x`. Dict with keys
        * "accepts_x": True if `x` can be provided.
        * "requires_x": True if `x` must be provided. If False but "accepts_x" is True, the global behavior differs
            depending on whether `x` is provided. "requires_x" can only be True if "accepts_x" is True as well.
        * "mean_of_local": True if global explanations are the mean of the individual local explanations, if `x` is
            provided. If True, it might be better to call method `explain()` instead of `explain_global()`, since the
            computational effort is identical.
        :return: Dict, as described above.
        """
        raise NotImplementedError()

    def __init__(self, ensemble: 'FittedEnsemble' = None, x: Optional[pd.DataFrame] = None, y=None, params=None):
        """
        Initialize an EnsembleExplainer for explaining the given ensemble, or constituents of it.
        :param ensemble: The ensemble to explain, an instance of FittedEnsemble.
        :param x: Training data, which is required by some explanation methods (e.g., SHAP).
        :param y: Labels of `x`, optional.
        :param params: Params obtained from a previous instantiation of an ensemble explainer of this type on
        `ensemble`. If given, neither `x` nor `y` may be provided.
        """
        if not (params is None or (x is None and y is None)):
            raise ValueError('If params is given, x and y must be None.')

    @property
    def params_(self) -> dict:
        """
        Get all params necessary for instantiating this EnsembleExplainer via parameter `params`.
        """
        raise NotImplementedError()

    def explain(self, x: pd.DataFrame, jobs: int = 1, batch_size: Optional[int] = None, model_id=None,
                show_progress: bool = False) -> dict:
        raise NotImplementedError()

    def explain_global(self, x: Optional[pd.DataFrame] = None, jobs: int = 1, batch_size: Optional[int] = None,
                       model_id=None, show_progress: bool = False) -> dict:
        raise NotImplementedError()


# load explanation backends
for _d in Path(__file__).parent.iterdir():
    if _d.is_dir() and (_d / '__init__.py').exists():
        importlib.import_module('.' + _d.stem, package=__package__)


# Paradigm for explaining a pipeline `model` of a FittedEnsemble
#
# setup:
# >>> preprocessing_explainer = TransformationExplainer.make(transformation=model.preprocessing)
# >>> x_train = preprocessing_explainer.fit_forward(x_train, y_train)
# >>> estimator_explainer = ModelExplainer(model.estimator, x_train, ...)
#
# local explanations for `x_test`:
# >>> x_test_pp = preprocessing_estimator.forward(x_test)
# >>> explanation = estimator_explainer.explain(x_test_pp)
# >>> explanation = preprocessing_explainer.backward(explanation)
#
# global explanations:
# >>> explanation = estimator_explainer.explain_global()
# >>> explanation = preprocessing_explainer.backward_global(explanation)


# Paradigm for explaining data `(x, y)` after applying some preprocessing steps `preprocessing`:
# >>> preprocessing_explainer = TransformationExplainer.make(transformation=preprocessing)
# >>> data_explainer = DataExplainer()
# >>> x_pp = preprocessing_explainer.fit_forward(x, y)`
# >>> explanation = data_explainer.explain(x_pp, y)                     # or `explain_global(x_pp, y)`
# >>> explanation = preprocessing_explainer.backward(explanation)       # or `backward_global(explanation)`
