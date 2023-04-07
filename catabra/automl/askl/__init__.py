#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

from catabra.automl.base import AutoMLBackend


def askl_factory(**kwargs):
    from catabra.automl.askl.backend import AutoSklearnBackend
    return AutoSklearnBackend(**kwargs)


AutoMLBackend.register('auto-sklearn', askl_factory)
