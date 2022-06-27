from typing import Optional
import numpy as np
import pandas as pd
import sklearn.impute
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.cluster
import sklearn.compose
import sklearn.decomposition
import sklearn.feature_selection
import sklearn.kernel_approximation
from sklearn.utils import _safe_indexing
from sklearn.utils.fixes import delayed
from sklearn.utils._mask import _get_mask
from joblib import Parallel
import sys

from catabra.explanation.base import TransformationExplainer, IdentityTransformationExplainer


def sklearn_explainer_factory(obj, params=None):
    if obj is None:
        obj = []
    if isinstance(obj, sklearn.pipeline.Pipeline):
        # last step in a pipeline MUST be an estimator or 'passthrough', which we can ignore here
        obj = [t[1] for t in obj.steps[:-1]]

    if isinstance(obj, list):
        return PipelineExplainer(obj, params=params)
    elif isinstance(obj, (sklearn.preprocessing.RobustScaler, sklearn.preprocessing.StandardScaler,
                          sklearn.preprocessing.QuantileTransformer, sklearn.preprocessing.PowerTransformer,
                          sklearn.preprocessing.MinMaxScaler, sklearn.preprocessing.Normalizer,
                          sklearn.preprocessing.OrdinalEncoder)):
        return IdentityTransformationExplainer(transformer=obj, params=params)
    elif isinstance(obj, str):
        if obj == 'passthrough':
            return IdentityTransformationExplainer(transformer=obj, params=params)
        else:
            return obj

    cls = getattr(sys.modules[__name__], obj.__class__.__name__ + 'Explainer', None)
    if cls is None:
        return obj
    else:
        return cls(obj, params=params)


_BACKWARD_INPUT_SHAPE = 'S has {} features, but {} returns output with {} features.'


class PipelineExplainer(TransformationExplainer):

    def __init__(self, steps: list, params=None):
        super(PipelineExplainer, self).__init__(params=params)
        if params is None:
            self._explainers = [TransformationExplainer.make(t) for t in steps]
        else:
            step_params = params.get('steps', ())
            assert len(step_params) == len(steps)
            self._explainers = [TransformationExplainer.make(t, params=p) for t, p in zip(steps, step_params)]

    @property
    def params_(self) -> dict:
        out = super(PipelineExplainer, self).params_
        out['steps'] = [e.params_ for e in self._explainers]
        return out

    def fit(self, x, y=None):
        raise RuntimeError(f'Method fit() of class {self.__class__.__name__} cannot be called.')

    def transform(self, x):
        for e in self._explainers:
            x = e.transform(x)
        return x

    def fit_forward(self, x, y):
        for e in self._explainers:
            x = e.fit_forward(x, y)
        return x

    def forward(self, x):
        for e in self._explainers:
            x = e.forward(x)
        return x

    def backward(self, s):
        for e in self._explainers[::-1]:
            s = e.backward(s)
        return s

    def backward_global(self, s):
        for e in self._explainers[::-1]:
            s = e.backward_global(s)
        return s


class ColumnTransformerExplainer(TransformationExplainer):

    def __init__(self, transformer: sklearn.compose.ColumnTransformer, params=None):
        super(ColumnTransformerExplainer, self).__init__(transformer=transformer, params=params)
        if params is None:
            self._explainers = [t if t == 'drop' else TransformationExplainer.make(t)
                                for _, t, _ in self._transformer.transformers_]
            # list of pairs `(columns, n_out)`, where `columns` is a list of column-indices and `n_out` is the number
            # of output features originating from `columns`
            self.mapping_: Optional[dict] = None
        else:
            transformer_params = params.get('transformers', ())
            assert len(transformer_params) == len(self._transformer.transformers_)
            self._explainers = [t if t == 'drop' else TransformationExplainer.make(t, params=p)
                                for (_, t, _), p in zip(self._transformer.transformers_, transformer_params)]
            self.mapping_ = params.get('mapping')
        self._feature_names_in = getattr(self._transformer, '_feature_names_in', None)

    @property
    def params_(self) -> dict:
        out = super(ColumnTransformerExplainer, self).params_
        out.update(transformers=[{} if e == 'drop' else e.params_ for e in self._explainers], mapping=self.mapping_)
        return out

    def fit_forward(self, x, y):
        self._validate_input(x, fitting=True)
        if hasattr(x, 'columns'):
            x_copy = pd.DataFrame(data={c: i for i, c in enumerate(x.columns)}, index=[0])
        else:
            x_copy = np.arange(x.shape[1]).reshape(1, -1)
        self.mapping_ = [_safe_indexing(x_copy, column, axis=1) for (_, _, column, _) in self._iter_explainers()]
        self.mapping_ = [list(m.values[0] if hasattr(m, 'values') else m[0]) for m in self.mapping_]

        xs = self._fit_forward(x, y, _fit_forward_one)
        self.mapping_ = [(m, xt.shape[1]) for m, xt in zip(self.mapping_, xs)]

        if not xs:
            return np.zeros((x.shape[0], 0))
        return self._transformer._hstack(xs)    # noqa

    def forward(self, x):
        self._validate_input(x)
        xs = self._fit_forward(x, None, _forward_one)

        if not xs:
            return np.zeros((x.shape[0], 0))
        return self._transformer._hstack(xs)    # noqa

    def backward(self, s: np.ndarray) -> np.ndarray:
        self._validate_backward_input(s, min_rank=2)
        return self._backward(s, _backward_one)

    def backward_global(self, s: np.ndarray) -> np.ndarray:
        self._validate_backward_input(s, min_rank=1)
        return self._backward(s, _backward_global_one)

    def _validate_input(self, x, fitting: bool = False):
        if hasattr(x, 'columns'):
            x_feature_names = np.asarray(x.columns)
        else:
            x_feature_names = None

        if x.shape[1] != self._transformer.n_features_in_:
            raise ValueError(f'X has {x.shape[1]} features, but {self.__class__.__name__}'
                             f' is expecting {self._transformer.n_features_in_} features as input.')
        if self._feature_names_in is None and x_feature_names is not None:
            if fitting:
                self._feature_names_in = x_feature_names
        elif (self._feature_names_in is not None and x_feature_names is not None
                and np.any(self._feature_names_in != x_feature_names)):
            raise RuntimeError('Given feature/column names do not match the ones for the data given during fit.')

    def _validate_backward_input(self, s, min_rank: int = 2):
        assert self.mapping_ is not None
        if len(s.shape) < min_rank:
            raise ValueError(f'S must be at least a {min_rank}D array.')
        elif s.shape[-1] != sum(n for _, n in self.mapping_):
            raise ValueError(_BACKWARD_INPUT_SHAPE.format(s.shape[-1], self.__class__.__name__,
                                                          sum(n for _, n in self.mapping_)))

    def _iter_explainers(self):
        get_weight = (self._transformer.transformer_weights or {}).get

        for (name, _, column), explainer in zip(self._transformer.transformers_, self._explainers):
            if explainer == 'drop' or ColumnTransformerExplainer._is_empty_column_selection(column):
                continue

            yield name, explainer, column, get_weight(name)

    def _fit_forward(self, x, y, func):
        try:
            return Parallel(n_jobs=self._transformer.n_jobs)(
                delayed(func)(
                    explainer=explainer,
                    x=_safe_indexing(x, column, axis=1),
                    y=y,
                    weight=weight)
                for name, explainer, column, weight in self._iter_explainers())
        except ValueError as e:
            if 'Expected 2D array, got 1D array instead' in str(e):
                raise ValueError('1D data passed to a transformer that expects 2D data. Try to specify the column'
                                 ' selection as a list of one item instead of a scalar.') from e
            else:
                raise

    def _backward(self, s: np.ndarray, func) -> np.ndarray:
        indices = []
        i = 0
        for _, n in self.mapping_:
            indices.append((i, i + n))
            i += n

        out = np.zeros(s.shape[:-1] + (self._transformer.n_features_in_,), dtype=s.dtype)

        xs = Parallel(n_jobs=self._transformer.n_jobs)(
            delayed(func)(
                explainer=explainer,
                s=s[..., i:j],
                weight=None)        # `weight` is set to None to make back-propagation of importance conservative
            for (i, j), (name, explainer, _, _) in zip(indices, self._iter_explainers())
        )

        for (columns, _), x in zip(self.mapping_, xs):
            assert len(columns) == x.shape[-1]
            out[..., columns] += x

        return out

    @staticmethod
    def _is_empty_column_selection(column):
        """
        Return True if the column selection is empty (empty list or all-False
        boolean array).

        """
        if hasattr(column, 'dtype') and np.issubdtype(column.dtype, np.bool_):
            return not column.any()
        elif hasattr(column, '__len__'):
            return (len(column) == 0 or
                    all(isinstance(col, bool) for col in column)
                    and not any(column))
        else:
            return False


class OneHotEncoderExplainer(TransformationExplainer):

    def __init__(self, transformer: sklearn.preprocessing.OneHotEncoder, params=None):
        super(OneHotEncoderExplainer, self).__init__(transformer=transformer, params=params)
        if self._transformer.drop_idx_ is not None:
            self._n_features_out = []
            for i, cats in enumerate(self._transformer.categories_):
                n_cats = len(cats)
                if self._transformer.drop_idx_[i] is None:
                    self._n_features_out.append(n_cats)
                else:
                    self._n_features_out.append(n_cats - 1)
        else:
            self._n_features_out = [len(cats) for cats in self._transformer.categories_]

    def fit_forward(self, x, y):
        return self.forward(x)

    def forward(self, x):
        return self.transform(x)

    def backward(self, s: np.ndarray) -> np.ndarray:
        if s.shape[-1] != sum(self._n_features_out):
            raise ValueError(_BACKWARD_INPUT_SHAPE.format(s.shape[-1], self.__class__.__name__,
                                                          sum(self._n_features_out)))

        out = np.zeros(s.shape[:-1] + (len(self._n_features_out),), dtype=s.dtype)
        j = 0
        for i, n in enumerate(self._n_features_out):
            out[..., i] = s[..., j:j + n].sum(axis=-1)
            j += n

        return out

    def backward_global(self, s: np.ndarray) -> np.ndarray:
        return self.backward(s)


class SimpleImputerExplainer(TransformationExplainer):

    def __init__(self, transformer: sklearn.impute.SimpleImputer, params=None):
        super(SimpleImputerExplainer, self).__init__(transformer=transformer, params=params)
        if self._transformer.strategy == 'constant':
            self._features = list(range(self._transformer.n_features_in_))
        else:
            valid_mask = ~_get_mask(self._transformer.statistics_, np.nan)
            self._features = np.where(valid_mask)[0]
        if self._transformer.indicator_ is None:
            self._indicator_features = []
        else:
            self._indicator_features = list(self._transformer.indicator_.features_)

    def fit_forward(self, x, y):
        return self.forward(x)

    def forward(self, x):
        return self.transform(x)

    def backward(self, s: np.ndarray) -> np.ndarray:
        if s.shape[-1] != len(self._features) + len(self._indicator_features):
            raise ValueError(_BACKWARD_INPUT_SHAPE.format(s.shape[-1], self.__class__.__name__,
                                                          len(self._features) + len(self._indicator_features)))
        if len(self._features) == self._transformer.n_features_in_ \
                and all(i == j for i, j in enumerate(self._features)):
            out = s[..., :self._transformer.n_features_in_]
        else:
            out = np.zeros(s.shape[:-1] + (self._transformer.n_features_in_,), dtype=s.dtype)
            for i, f in enumerate(self._features):
                out[..., f] = s[..., i]

        if self._indicator_features:
            out += _BaseFilterExplainer.static_backward(
                s[..., -len(self._indicator_features):],
                self._indicator_features,
                self._transformer.n_features_in_,
                cls_name='_MissingIndicatorExplainer'
            )

        return out

    def backward_global(self, s: np.ndarray) -> np.ndarray:
        return self.backward(s)


class _BaseFilterExplainer(TransformationExplainer):

    def __init__(self, transformer, params=None):
        super(_BaseFilterExplainer, self).__init__(transformer=transformer, params=params)
        mask = self._transformer._get_support_mask()  # noqa
        self._features = np.where(mask)[0]
        self.n_features_in_ = None if params is None else params.get('n_features_in')

    @property
    def params_(self) -> dict:
        out = super(_BaseFilterExplainer, self).params_
        out['n_features_in'] = self.n_features_in_
        return out

    def fit_forward(self, x, y):
        self.n_features_in_ = x.shape[1]        # not all subclasses have an `n_features_in_` attribute
        return self.forward(x)

    def forward(self, x):
        return self.transform(x)

    def backward(self, s: np.ndarray) -> np.ndarray:
        return self.static_backward(s, self._features, self.n_features_in_, self.__class__.__name__)

    def backward_global(self, s: np.ndarray) -> np.ndarray:
        return self.backward(s)

    @classmethod
    def static_backward(cls, s: np.ndarray, feat: list, n_feat_in: int, cls_name: Optional[str] = None) -> np.ndarray:
        if s.shape[-1] != len(feat):
            if cls_name is None:
                cls_name = cls.__name__
            raise ValueError(_BACKWARD_INPUT_SHAPE.format(s.shape[-1], cls_name, len(feat)))
        if len(feat) == n_feat_in and all(i == j for i, j in enumerate(feat)):
            return s

        out = np.zeros(s.shape[:-1] + (n_feat_in,), dtype=s.dtype)
        for i, f in enumerate(feat):
            out[..., f] = s[..., i]
        return out


class MissingIndicatorExplainer(TransformationExplainer):

    def __init__(self, transformer: sklearn.impute.MissingIndicator, params=None):
        super(MissingIndicatorExplainer, self).__init__(transformer=transformer, params=params)

    def fit_forward(self, x, y):
        return self.forward(x)

    def forward(self, x):
        return self.transform(x)

    def backward(self, s: np.ndarray) -> np.ndarray:
        # _BaseFilterExplainer can be employed here as well
        return _BaseFilterExplainer.static_backward(s, list(self._transformer.features_),
                                                    self._transformer.n_features_in_)

    def backward_global(self, s: np.ndarray) -> np.ndarray:
        return self.backward(s)


class VarianceThresholdExplainer(_BaseFilterExplainer):
    pass


class SelectPercentileExplainer(_BaseFilterExplainer):
    pass


class SelectKBestExplainer(_BaseFilterExplainer):
    pass


class SelectFprExplainer(_BaseFilterExplainer):
    pass


class SelectFdrExplainer(_BaseFilterExplainer):
    pass


class SelectFweExplainer(_BaseFilterExplainer):
    pass


class GenericUnivariateSelectExplainer(_BaseFilterExplainer):
    pass


class SelectFromModelExplainer(_BaseFilterExplainer):
    pass


class SequentialFeatureSelectorExplainer(_BaseFilterExplainer):
    pass


class RFEExplainer(_BaseFilterExplainer):
    pass


class RFECV(_BaseFilterExplainer):
    pass


class FeatureAgglomerationExplainer(TransformationExplainer):

    def __init__(self, transformer: sklearn.cluster.FeatureAgglomeration, params=None):
        super(FeatureAgglomerationExplainer, self).__init__(transformer=transformer, params=params)
        self._clusters = np.unique(self._transformer.labels_)
        self._weights = None

    def fit_forward(self, x, y):
        return self.forward(x)

    def forward(self, x):
        if self._transformer.pooling_func in (np.min, np.max):
            self._weights = np.zeros(x.shape, dtype=np.float32)
            for lbl in self._clusters:
                mask = np.asarray(self._transformer.labels_ == lbl)     # (n_features,)
                mask = (mask[np.newaxis] & (x == self._transformer.pooling_func(x[:, mask], axis=1)[..., np.newaxis]))\
                    .astype(self._weights.dtype)                        # (n_samples, n_features)
                mask /= np.maximum(mask.sum(axis=1, keepdims=True), 1e-8)
                self._weights += mask
        return self.transform(x)

    def backward(self, s: np.ndarray) -> np.ndarray:
        if self._transformer.pooling_func in (np.min, np.max):
            if len(s.shape) < 2:
                raise ValueError('S must be at least a 2D array.')
            elif s.shape[-1] != len(self._clusters):
                raise ValueError(
                    _BACKWARD_INPUT_SHAPE.format(s.shape[-1], self.__class__.__name__, len(self._clusters))
                )
            elif self._weights is None:
                raise ValueError('Method forward() must be called before calling method backward().')
            elif len(self._weights) != s.shape[-2]:
                raise ValueError(f'Method forward() was called on {len(self._weights)} samples,'
                                 f' but method backward() is called on {s.shape[-2]} samples.')

            out = np.zeros(s.shape[:-1] + (len(self._transformer.labels_),), dtype=s.dtype)
            for i, lbl in enumerate(self._clusters):
                mask = self._transformer.labels_ == lbl
                out[..., mask] = s[..., [i]]      # normalization happens below
            out *= self._weights.reshape(tuple([1] * (len(s.shape) - 2)) + self._weights.shape)

            return out
        else:
            return self.backward_global(s)

    def backward_global(self, s: np.ndarray) -> np.ndarray:
        if len(s.shape) < 1:
            raise ValueError('S must be at least a 1D array.')
        elif s.shape[-1] != len(self._clusters):
            raise ValueError(_BACKWARD_INPUT_SHAPE.format(s.shape[-1], self.__class__.__name__, len(self._clusters)))

        out = np.zeros(s.shape[:-1] + (len(self._transformer.labels_),), dtype=s.dtype)
        for i, lbl in enumerate(self._clusters):
            mask = self._transformer.labels_ == lbl
            out[..., mask] = s[..., [i]] / mask.sum()

        return out


class _LinearTransformationExplainer(TransformationExplainer):
    """
    Linear transformations are tricky. Assume that x_1, ..., x_m are the input features and that an new feature z is
    constructed as a linear combination of the x_i:

          z = sum_{i=1}^m alpha_i * x_i + b

    Then there are at least three possibilities how to back-propagate the importance s of z:

    1. Global back-propagation. Do not take actual feature values into account, only consider coefficients:

          t_i = alpha_i / (sum_{j=1)^m |alpha_j|) * s

          This is a very simple approach -- maybe too simple?

    2. LRP-0-like. Multiply coefficients by actual feature values:

          t_i = alpha_i * x_i / (sum_{j=1}^m |alpha_j * x_j|) * s

          This closely resembles the LRP-0 rule of layer-wise relevance propagation, the only differences being the
          absolute values in the sum in the denominator (to avoid division by 0), and the fact that the bias b is
          ignored. This approach is probably better suited to cases where 0 represents "missingness"/"de-activation",
          as in ReLU networks; that's in fact what LRP-0 was originally proposed for.

    3. SHAP-like:

          t_i = alpha_i * (x_i - mu_i) / (sum_{j=1}^m |alpha_j * (x_j - mu_j)|) * s

          where mu_i := E(X_i) is the expected feature value.
          Removing the absolute values in the denominator yields a rule that fits perfectly with linear SHAP (and
          maybe SHAP in general?). However, the choice of the expected value is sort of arbitrary and merely owes
          to the fact that SHAP sets features to their expected value for "switching them off". Other methods might
          set them to 0 (think of LRP-0!) or any other sentinel value instead.

    In every case, the denominator ensures that total importance is roughly preserved -- "roughly" because of the
    absolute values that avoid division by 0. Other means for avoiding division by 0 are conceivable as well, of course.
    The fact that the bias b is entirely ignored owes to the general paradigm of back-propagating importance, that
    transformations that act on each feature separately have no impact on feature importance.

    Given that no solution appears to be perfect, we go with the simplest one (1.) for the time being. This decision
    might have to be revisited in the future, though.
    """

    def __init__(self, transformer, matrix: np.ndarray, params=None):
        # `matrix` must be array of shape `(n_features_in, n_features_out)`, such that `transform()` approximately
        # returns `np.dot(x, matrix)`
        super(_LinearTransformationExplainer, self).__init__(transformer=transformer, params=params)
        assert matrix.ndim == 2
        self._matrix = matrix / np.maximum(np.abs(matrix).sum(axis=0, keepdims=True), 1e-7)

    def fit_forward(self, x, y):
        return self.forward(x)

    def forward(self, x):
        return self.transform(x)

    def backward(self, s: np.ndarray) -> np.ndarray:
        return self.backward_global(s)

    def backward_global(self, s: np.ndarray) -> np.ndarray:
        if len(s.shape) < 1:
            raise ValueError('S must be at least a 1D array.')
        elif s.shape[-1] != self._matrix.shape[1]:
            raise ValueError(_BACKWARD_INPUT_SHAPE.format(s.shape[-1], self.__class__.__name__, self._matrix.shape[1]))
        return np.dot(s, self._matrix.T)


class PCAExplainer(_LinearTransformationExplainer):

    def __init__(self, transformer: sklearn.decomposition.PCA, params=None):
        super(PCAExplainer, self).__init__(transformer, transformer.components_.T, params=params)


class FastICAExplainer(_LinearTransformationExplainer):

    def __init__(self, transformer: sklearn.decomposition.FastICA, params=None):
        super(FastICAExplainer, self).__init__(transformer, transformer.components_.T, params=params)


class TruncatedSVDExplainer(_LinearTransformationExplainer):

    def __init__(self, transformer: sklearn.decomposition.TruncatedSVD, params=None):
        super(TruncatedSVDExplainer, self).__init__(transformer, transformer.components_.T, params=params)


class RBFSamplerExplainer(_LinearTransformationExplainer):

    def __init__(self, transformer: sklearn.kernel_approximation.RBFSampler, params=None):
        # RBFSampler does not implement a linear transformation, but instead returns
        #   X_hat = cos(X @ A + offset) * c
        # Adding a component-wise offset, taking the cosine and multiplying by a constant factor have no influence on
        # feature importance, and can hence be ignored.
        super(RBFSamplerExplainer, self).__init__(transformer, transformer.random_weights_, params=params)


def _fit_forward_one(explainer, x, y, weight):
    res = explainer.fit_forward(x, y)
    if weight is None:
        return res
    return res * weight


def _forward_one(explainer, x, y, weight):
    res = explainer.forward(x)
    if weight is None:
        return res
    return res * weight


def _backward_one(explainer, s, weight):
    res = explainer.backward(s)
    if weight is None:
        return res
    return res * weight


def _backward_global_one(explainer, s, weight):
    res = explainer.backward_global(s)
    if weight is None:
        return res
    return res * weight
