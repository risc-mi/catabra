#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

from catabra.explanation.base import EnsembleExplainer


def shap_factory(**kwargs):
    from catabra.explanation._shap.backend import SHAPEnsembleExplainer
    return SHAPEnsembleExplainer(**kwargs)


EnsembleExplainer.register('shap', shap_factory)
