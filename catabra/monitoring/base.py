#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import importlib
import time
from pathlib import Path
from typing import Callable, Optional


class TrainingMonitorBackend:
    """
    Base class of training monitor backends, i.e., Simple abstraction layer for training monitor backends, which allows
    to include other backends besides plotly in the future.

    Parameters
    ----------
    text_pattern: str, optional
    folder: str, optional
    """

    __registered = {}

    @staticmethod
    def register(name: str, backend: Callable[..., 'TrainingMonitorBackend']):
        """
        Register a new training monitor backend.

        Parameters
        ----------
        name: str
            The name of the backend.
        backend: Callable
            The backend, a function mapping argument-dicts to instances of class `TrainingMonitorBackend`
        (or subclasses thereof).
        """
        TrainingMonitorBackend.__registered[name] = backend

    @staticmethod
    def get(name: str, **kwargs) -> Optional['TrainingMonitorBackend']:
        cls = TrainingMonitorBackend.__registered.get(name)
        return cls if cls is None else cls(**kwargs)

    @classmethod
    def name(cls) -> str:
        raise NotImplementedError()

    def __init__(self, text_pattern: Optional[str] = None, folder: Optional[str] = None):
        self._start_time = None
        self._text_pattern = text_pattern
        self._folder = folder
        self._event_step = {}

    @property
    def start_time(self) -> Optional[float]:
        return self._start_time

    @start_time.setter
    def start_time(self, value: float):
        self._start_time = value

    @property
    def text_pattern(self) -> Optional[str]:
        return self._text_pattern

    @property
    def folder(self) -> Optional[str]:
        return self._folder

    def launch(self):
        """
        Launch the training monitor. This method is usually called before training starts.
        """
        raise NotImplementedError()

    def shutdown(self):
        """
        Shut the training monitor down. This method is usually called after training ends.
        """
        raise NotImplementedError()

    def set_params(self, **params):
        """
        Set parameters associated with the current training run, to be logged with the training monitor.
        Note: Not every backend needs to support logging parameters in this way.
        """
        raise NotImplementedError()

    def get_info(self) -> Optional[str]:
        """
        Get information about this training monitor instance, e.g., how the user interface can be accessed.
        """
        raise NotImplementedError()

    def update(self, event=None, timestamp: Optional[float] = None, elapsed_time: Optional[float] = None,
               step: Optional[int] = None, text: Optional[str] = None, **metrics: float):
        """
        Update this training monitor instance by logging performance metrics of a new training iteration.

        Parameters
        ----------
        event: optional
            Name of the event that triggered this call.
        timestamp: float, optional
            Timestamp of the update. None defaults to the current timestamp.
        elapsed_time: float, optional
            Total elapsed time of the update since training started. None defaults to the difference between `timestamp`
            and the start time of this training monitor instance.
        step: int, optional
            Step (iteration) of the update. None defaults to the total number of steps of `event` recorded so far.
        text: str, optional
            Additional text recorded with the update, optional.
        **metrics: float
            Metrics to be logged, and their values.
        """
        if timestamp is None:
            timestamp = time.time()
        if self._start_time is None:
            self._start_time = timestamp
        if elapsed_time is None:
            elapsed_time = timestamp - self._start_time
        if step is None:
            step = self._event_step.get(event, 0) + 1
            self._event_step[event] = step
        else:
            self._event_step[event] = max(step, self._event_step.get(event, 0))
        if self._text_pattern is not None:
            new_text = self._text_pattern.format(event=event, timestamp=timestamp, elapsed_time=elapsed_time, step=step)
            if text is None or text == '':
                text = new_text
            else:
                text += '\n' + new_text
        self._update(event, timestamp, elapsed_time, step, text, **metrics)

    def __enter__(self):
        self.launch()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def _update(self, event, timestamp: float, elapsed_time: float, step: int, text: Optional[str], **metrics: float):
        raise NotImplementedError()


for _d in Path(__file__).parent.iterdir():
    if _d.is_dir() and (_d / '__init__.py').exists():
        importlib.import_module('catabra.monitoring.' + _d.stem, package=__package__)
