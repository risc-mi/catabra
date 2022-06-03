import numpy as np


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

    def __init__(self, model=None):
        self._model = model

    @property
    def model(self):
        return self._model

    def fit(self, x, y) -> 'ModelExplainer':
        """
        Fit this explainer to training data.
        :param x: Features, array-like of shape `(n_samples, n_features)`. Does not need to be the data the underlying
        prediction model was trained on!
        :param y: Labels, array-like of shape `(n_samples, n_labels)` or `(n_samples,)`.
        :return: This `ModelExplainer` instance.
        """
        return self

    def explain(self, x) -> np.ndarray:
        """
        Explain data locally by transforming it into an array of explanations.
        :param x: Features, array-like of shape `(n_samples, n_features)`. `n_features` must be the same as in the data
        this explainer instance was fitted on.
        :return: Explanations, array of shape `(*dims, n_samples, n_features)`.
        """
        raise NotImplementedError()

    def explain_global(self) -> np.ndarray:
        """
        Explain the model globally.
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
    def make(obj):
        if all(hasattr(obj, attr)
               for attr in ('fit', 'transform', 'fit_forward', 'forward', 'backward', 'backward_global')):
            return obj

        for _, func in TransformationExplainer._factories:
            out = func(obj)
            if out is not obj:
                return out

        raise RuntimeError(f'Object of type {type(obj)} cannot be converted into a transformation explainer.')

    def __init__(self, transformer=None):
        self._transformer = transformer

    @property
    def transformer(self):
        return self._transformer

    @property
    def params_(self) -> dict:
        """
        Get all params obtained from fitting the explainer to data in method `fit_forward()`, and which can be passed
        to method `set_params()`.
        """
        raise NotImplementedError()

    def set_params(self, **params):
        """
        Set pre-computed params of this explainer, which are normally obtained through fitting it to data in method
        `fit_forward()`.
        """
        # Implementation note: this method is intentionally _not_ implemented as the setter of property `params_`,
        # because statements like `explainer.params_ = ...` would look odd.
        raise NotImplementedError()

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

    def __init__(self, transformer=None):
        super(IdentityTransformationExplainer, self).__init__(transformer=transformer)
        self._transform_func = getattr(self._transformer, 'transform', None)

    @property
    def params_(self) -> dict:
        return {}

    def set_params(self, **params):
        pass

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


# Paradigm for explaining a pipeline `model` of a FittedEnsemble
#
# setup:
# >>> preprocessing_explainer = TransformationExplainer(transformation=model['preprocessing'])
# >>> estimator_explainer = ModelExplainer(model=model['estimator'])
# >>> x_train = preprocessing_explainer.fit_forward(x_train, y_train)
# >>> estimator_explainer.fit(x_train, y_train)
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
# >>> preprocessing_explainer = TransformationExplainer(transformation=preprocessing)
# >>> data_explainer = DataExplainer()
# >>> x_pp = preprocessing_explainer.fit_forward(x, y)`
# >>> explanation = data_explainer.explain(x_pp, y)                     # or `explain_global(x_pp, y)`
# >>> explanation = preprocessing_explainer.backward(explanation)       # or `backward_global(explanation)`
