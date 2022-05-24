from .base import DataExplainer, ModelExplainer, TransformationExplainer, IdentityTransformationExplainer
from . import sklearn_explainer

try:
    # in version 0.24.2 this is still experimental, and shap does not find it without explicitly enabling it
    from sklearn.experimental import enable_hist_gradient_boosting
except ImportError:
    pass


TransformationExplainer.register_factory('sklearn', sklearn_explainer.sklearn_explainer_factory, errors='ignore')


__all__ = ['DataExplainer', 'ModelExplainer', 'TransformationExplainer', 'IdentityTransformationExplainer',
           'sklearn_explainer']
