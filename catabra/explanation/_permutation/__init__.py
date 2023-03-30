#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

from ..base import EnsembleExplainer


def permutation_factory(**kwargs):
    from catabra.explanation._permutation.backend import PermutationEnsembleExplainer
    return PermutationEnsembleExplainer(**kwargs)


EnsembleExplainer.register('permutation', permutation_factory)
