#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

from .base import TransformationExplainer, IdentityTransformationExplainer, EnsembleExplainer
from . import sklearn_explainer
from .main import explain, CaTabRaExplanation, explain_split, plot_beeswarms, plot_bars, average_local_explanations

try:
    # in version 0.24.2 this is still experimental, and shap does not find it without explicitly enabling it
    from sklearn.experimental import enable_hist_gradient_boosting
except ImportError:
    pass


TransformationExplainer.register_factory('sklearn', sklearn_explainer.sklearn_explainer_factory, errors='ignore')


__all__ = ['TransformationExplainer', 'IdentityTransformationExplainer', 'EnsembleExplainer', 'sklearn_explainer',
           'explain', 'CaTabRaExplanation', 'explain_split', 'plot_beeswarms', 'plot_bars',
           'average_local_explanations']
