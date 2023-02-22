#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

from ..base import TrainingMonitorBackend
from ...util import logging, plotting


def plotly_factory(**kwargs):
    try:
        from .backend import PlotlyBackend
        return PlotlyBackend(**kwargs)
    except Exception as e:  # noqa
        if isinstance(e, ImportError) and 'plotly' in e.msg:
            logging.warn(plotting.PLOTLY_WARNING)
        else:
            logging.warn('Loading the plotly training monitor backend failed with the following exception: ' + str(e))
        return None


TrainingMonitorBackend.register('plotly', plotly_factory)
