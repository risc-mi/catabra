from ...explanation.sklearn_explainer import TransformationExplainer, IdentityTransformationExplainer, \
    OneHotEncoderExplainer, SelectPercentileExplainer, GenericUnivariateSelectExplainer


def askl_explainer_factory(obj):
    if obj.__class__.__module__.startswith('autosklearn.'):
        if obj.__class__.__name__ in ('OrdinalEncoding', 'CategoryShift', 'MinorityCoalescer', 'Densifier'):
            return IdentityTransformationExplainer(transformer=obj)
        elif obj.__class__.__name__ == 'SelectPercentileClassification':
            return SelectPercentileClassificationExplainer(obj)
        elif obj.__class__.__name__ == 'SelectClassificationRates':
            return SelectClassificationRatesExplainer(obj)
        elif obj.__class__.__name__ == 'SparseOneHotEncoder':
            return SparseOneHotEncoderExplainer(obj)
        # TODO: Nystroem.
        #    Nystroem works as follows:
        #    * Pick random `n_components` samples from the training data, without replacement. Call them `components_`.
        #    * For each sample to be transformed, compute similarity/distance to each of the `components_`.
        #      The output is a feature array of length `n_components`, regardless of the original number of features.
        #    * Multiply with a normalization matrix of shape `(n_components, n_components)`.
        #      This last step can perhaps be ignored during back-propagation.

    return obj


class SelectPercentileClassificationExplainer(SelectPercentileExplainer):

    def __init__(self, transformer):
        super(SelectPercentileClassificationExplainer, self).__init__(transformer.preprocessor)
        self._transformer_askl = transformer

    def transform(self, x):
        # original autosklearn object must be used for transforming data
        return self._transformer_askl.transform(x)


class SelectClassificationRatesExplainer(GenericUnivariateSelectExplainer):

    def __init__(self, transformer):
        super(SelectClassificationRatesExplainer, self).__init__(transformer.preprocessor)
        self._transformer_askl = transformer

    def transform(self, x):
        # original autosklearn object must be used for transforming data
        return self._transformer_askl.transform(x)


class SparseOneHotEncoderExplainer(OneHotEncoderExplainer):

    def fit_forward(self, x, y):
        self._n_features_out = []
        cumsum = 0
        for n in self._transformer.n_values_:
            mask = (self._transformer.active_features_ >= cumsum) & (self._transformer.active_features_ < cumsum + n)
            self._n_features_out.append(mask.sum())
            cumsum += n
        return self.forward(x)
