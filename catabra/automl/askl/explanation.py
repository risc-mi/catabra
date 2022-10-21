import numpy as np

from ...explanation.sklearn_explainer import TransformationExplainer, IdentityTransformationExplainer, \
    OneHotEncoderExplainer, SelectPercentileExplainer, GenericUnivariateSelectExplainer


def askl_explainer_factory(obj, params=None):
    if obj.__class__.__module__.startswith('autosklearn.'):
        if obj.__class__.__name__ in ('OrdinalEncoding', 'CategoryShift', 'MinorityCoalescer', 'Densifier'):
            return IdentityTransformationExplainer(transformer=obj, params=params)
        elif obj.__class__.__name__ == 'SelectPercentileClassification':
            return SelectPercentileClassificationExplainer(obj, params=params)
        elif obj.__class__.__name__ == 'SelectClassificationRates':
            return SelectClassificationRatesExplainer(obj, params=params)
        elif obj.__class__.__name__ == 'SparseOneHotEncoder':
            return SparseOneHotEncoderExplainer(obj, params=params)
        elif obj.__class__.__name__ == 'TfidfEncoder':
            return TfidfEncoderExplainer(obj, params=params)
        # TODO: Nystroem.
        #    Nystroem works as follows:
        #    * Pick random `n_components` samples from the training data, without replacement. Call them `components_`.
        #    * For each sample to be transformed, compute similarity/distance to each of the `components_`.
        #      The output is a feature array of length `n_components`, regardless of the original number of features.
        #    * Multiply with a normalization matrix of shape `(n_components, n_components)`.
        #      This last step can perhaps be ignored during back-propagation.

    return obj


class SelectPercentileClassificationExplainer(SelectPercentileExplainer):

    def __init__(self, transformer, params=None):
        super(SelectPercentileClassificationExplainer, self).__init__(transformer.preprocessor, params=params)
        self._transformer_askl = transformer

    def transform(self, x):
        # original autosklearn object must be used for transforming data
        return self._transformer_askl.transform(x)


class SelectClassificationRatesExplainer(GenericUnivariateSelectExplainer):

    def __init__(self, transformer, params=None):
        super(SelectClassificationRatesExplainer, self).__init__(transformer.preprocessor, params=params)
        self._transformer_askl = transformer

    def transform(self, x):
        # original autosklearn object must be used for transforming data
        return self._transformer_askl.transform(x)


class SparseOneHotEncoderExplainer(OneHotEncoderExplainer):

    def __init__(self, transformer=None, params=None):
        TransformationExplainer.__init__(self, transformer=transformer, params=params)
        self._n_features_out = []
        cumsum = 0
        for n in self._transformer.n_values_:
            mask = (self._transformer.active_features_ >= cumsum) & (self._transformer.active_features_ < cumsum + n)
            self._n_features_out.append(mask.sum())
            cumsum += n


class TfidfEncoderExplainer(TransformationExplainer):

    def __init__(self, transformer, params=None):
        super(TfidfEncoderExplainer, self).__init__(transformer, params=params)
        self.n_features_in_ = None if params is None else params.get('n_features_in')

    @property
    def params_(self) -> dict:
        out = super(TfidfEncoderExplainer, self).params_
        out['n_features_in'] = self.n_features_in_
        return out

    def fit_forward(self, x, y):
        self.n_features_in_ = x.shape[1]
        return self.forward(x)

    def forward(self, x):
        return self.transform(x)

    def backward(self, s: np.ndarray) -> np.ndarray:
        if self._transformer.per_column:
            n_features_out = [len(pp.vocabulary_) for pp in self._transformer.preprocessor.values()]
            assert s.shape[-1] == sum(n_features_out)
            out = np.zeros((*s.shape[:-1], self.n_features_in_), dtype=s.dtype)

            a = 0
            for i, n in enumerate(n_features_out):
                out[..., i] = s[..., a:a + n].sum(axis=-1)
                a += n
        else:
            assert s.shape[-1] == len(self._transformer.preprocessor.vocabulary_)
            # redistribute importance uniformly across features
            out = s.sum(axis=-1, keepdims=True).repeat(self.n_features_in_, axis=-1) / self.n_features_in_

        return out

    def backward_global(self, s: np.ndarray) -> np.ndarray:
        return self.backward(s)
