#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

from catabra.explanation import sklearn_explainer
from catabra.explanation.base import (
    EnsembleExplainer,
    IdentityTransformationExplainer,
    TransformationExplainer,
)
from catabra.explanation.main import (
    CaTabRaExplanation,
    average_local_explanations,
    explain,
    explain_split,
    plot_bars,
    plot_beeswarms,
)

try:
    # in version 0.24.2 this is still experimental, and shap does not find it without explicitly enabling it
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa F401
except ImportError:
    pass


TransformationExplainer.register_factory('sklearn', sklearn_explainer.sklearn_explainer_factory, errors='ignore')


__all__ = ['TransformationExplainer', 'IdentityTransformationExplainer', 'EnsembleExplainer', 'sklearn_explainer',
           'explain', 'CaTabRaExplanation', 'explain_split', 'plot_beeswarms', 'plot_bars',
           'average_local_explanations']
