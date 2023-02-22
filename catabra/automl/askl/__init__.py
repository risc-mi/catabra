#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

from ..base import AutoMLBackend


def askl_factory(**kwargs):
    from .backend import AutoSklearnBackend
    return AutoSklearnBackend(**kwargs)


AutoMLBackend.register('auto-sklearn', askl_factory)
