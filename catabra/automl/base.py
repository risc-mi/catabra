#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import importlib
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd

from catabra.automl.fitted_ensemble import FittedEnsemble


class AutoMLBackend:
    """
    Simple abstraction layer for AutoML backends, which allows to include other backends besides autosklearn
    in the future.
    """

    __registered = {}

    @staticmethod
    def register(name: str, backend: Callable[..., 'AutoMLBackend']):
        """
        Register a new AutoML backend.

        Parameters
        ----------
        name: str
            The name of the backend.
        backend: Callable[..., AutoMLBackend]
            The backend, a function mapping argument-dicts to instances of class `AutoMLBackend` (or subclasses
            thereof).
        """
        AutoMLBackend.__registered[name] = backend

    @staticmethod
    def get(name: str, **kwargs) -> Optional['AutoMLBackend']:
        cls = AutoMLBackend.__registered.get(name)
        return cls if cls is None else cls(**kwargs)

    def __init__(self, task: str = None, config: dict = None, tmp_folder: Union[Path, str, None] = None):
        if task not in ('regression', 'binary_classification', 'multiclass_classification',
                        'multilabel_classification'):
            raise ValueError(f'Unknown prediction task: {task}')
        self.task: str = task
        self.config: dict = config or {}
        # NEVER pickle Path objects!
        self._tmp_folder: Optional[str] = tmp_folder.as_posix() if isinstance(tmp_folder, Path) else tmp_folder
        self.model_ = None
        self.calibrator_ = None

    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def tmp_folder(self) -> Optional[Path]:
        return self._tmp_folder if self._tmp_folder is None else Path(self._tmp_folder)

    @property
    def model_ids_(self) -> list:
        """
        List of IDs for accessing individual (constituent) models of the final ensemble. Return [] if no such models
        exist or cannot be accessed
        """
        raise NotImplementedError()

    @property
    def calibrator_(self):
        return getattr(self, '_calibrator', None)

    @calibrator_.setter
    def calibrator_(self, value):
        if value is None:
            self._calibrator_method = None
        elif self.task not in ('binary_classification', 'multiclass_classification', 'multilabel_classification'):
            raise ValueError('Calibrator can only be used in classification tasks.')
        elif hasattr(value, 'transform'):
            self._calibrator_method = 'transform'
        elif hasattr(value, 'predict'):
            self._calibrator_method = 'predict'
        else:
            raise ValueError('Calibrator must implement method `transform()` or `predict()`.')
        self._calibrator = value

    def summary(self) -> dict:
        """
        Summary of trained model(s) and preprocessing pipeline(s). No scores (neither train, nor validation,
        nor test), only "architectural" information.
        """
        raise NotImplementedError()

    def training_history(self) -> pd.DataFrame:
        """
        Summary of the model training. Contains at least the final training- and validation scores, and, depending
        on the backend, also their temporal evolution.
        """
        raise NotImplementedError()

    def fitted_ensemble(self, ensemble_only: bool = True) -> FittedEnsemble:
        raise NotImplementedError()

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame, groups: Optional[np.ndarray] = None,
            sample_weights: Optional[np.ndarray] = None, time: Optional[int] = None, memory: Optional[int] = None,
            jobs: Optional[int] = None, dataset_name: Optional[str] = None, monitor=None) -> 'AutoMLBackend':
        """
        Fit models and/or an ensemble thereof to new data.

        Parameters
        ----------
        x_train: DataFrame
            Features DataFrame. Allowed column data types are numeric (float, int, bool) and categorical. May contain
            NaN values.
        y_train: DataFrame
            Targets DataFrame. Columns must be suitable for the specified prediction task, and must have float data
            type. May contain NaN values.
        groups: ndarray, optional
            Grouping information for internal (cross) validation. If given, must have shape `(n_samples,)`.
        sample_weights: ndarray, optional
            Optional sample weights. If given, must have shape `(n_samples,)`.
        time: int, optional
            The time budget, in minutes, or -1 if no budget is imposed. Overwrites the time budget specified in the
            config dict.
        memory: int, optional
            The memory budget, in MB. Overwrites the time budget specified in the config dict.
        jobs: int, optional
            The number of jobs to use, or -1 if all available processors shall be used. Overwrites the number of jobs
            specified in the config dict.
        dataset_name: str
            Name of the data set.
        monitor: TrainingMonitorBackend, optinoal
            Backend used for live monitoring of the training process.

        Returns
        -------
        AutoMLBackend
            This AutoML object.
        """
        raise NotImplementedError()

    def predict(self, x: pd.DataFrame, jobs: Optional[int] = None, batch_size: Optional[int] = None,
                model_id=None, calibrated: bool = 'auto') -> np.ndarray:
        """
        Apply trained models to given data.

        Parameters
        ----------

        x: DataFrame
            Features DataFrame. Must have the exact same format as the data this AutoML object was trained on.
        jobs: int, optional
            The number of jobs to use, or -1 if all available processors shall be used. Overwrites the number of jobs
            specified in the config dict.
        batch_size: int, optional
            Batch size, i.e., number of samples processed in parallel.
        model_id: str, optional
            The ID of the model to apply, as in `model_ids_`. If None, the whole ensemble is applied.
        calibrated:  bool | 'auto', default='auto'
            Whether to return calibrated predictions in case of classification tasks. If `"auto"`, calibrated
            predictions are returned iff `model_id` is None.

        Returns
        -------
        ndarray
            Array of predictions. In case of classification, these are class indicators rather than probabilities.
        """
        raise NotImplementedError()

    def predict_proba(self, x: pd.DataFrame, jobs: Optional[int] = None, batch_size: Optional[int] = None,
                      model_id=None, calibrated: bool = 'auto') -> np.ndarray:
        """
        Apply trained models to given data. In contrast to method `predict()`, this method returns class probabilities
        in case of classification tasks. Does not work for regression tasks.

        Parameters
        ----------
        x: DataFrame
            Features DataFrame. Must have the exact same format as the data this AutoML object was trained on.
        jobs: int, optional
            The number of jobs to use, or -1 if all available processors shall be used. Overwrites the number of jobs
            specified in the config dict.
        batch_size: int, optional
            Batch size, i.e., number of samples processed in parallel.
        model_id: str, optional
            The ID of the model to apply, as in `model_ids_`. If None, the whole ensemble is applied.
        calibrated: bool | 'auto'
            Whether to return calibrated predictions in case of classification tasks. If `"auto"`, calibrated
            predictions are returned iff `model_id` is `None`.

        Returns
        -------
        ndarray
            Array of class probabilities, of shape `(n_samples, n_classes)`.
        """
        raise NotImplementedError()

    def predict_all(self, x: pd.DataFrame, jobs: Optional[int] = None, batch_size: Optional[int] = None) \
            -> Dict[Any, np.ndarray]:
        """
        Apply all trained models to given data, including constituent models and the entire ensemble.

        Parameters
        ----------
        x: DataFrame
            Features DataFrame. Must have the exact same format as the data this AutoML object was trained on.
        jobs: int, optional
            The number of jobs to use, or -1 if all available processors shall be used. Overwrites the number of jobs
            specified in the config dict.
        batch_size: int,optional
            Batch size, i.e., number of samples processed in parallel.

        Returns
        -------
        Dict[Any, np.ndarray]
            Dict mapping model IDs to prediction-arrays. If existing, the key of the entire ensemble is
            `"__ensemble__"`.
        """
        raise NotImplementedError()

    def predict_proba_all(self, x: pd.DataFrame, jobs: Optional[int] = None, batch_size: Optional[int] = None) \
            -> Dict[Any, np.ndarray]:
        """
        Apply all trained models to given data, including constituent models and the entire ensemble. In contrast to
        method `predict_all()`, this method returns class probabilities in case of classification tasks. Does not work
        for regression tasks.

        Parameters
        ----------
        x: DataFrame
            Features DataFrame. Must have the exact same format as the data this AutoML object was trained on.
        jobs: int, optional
            The number of jobs to use, or -1 if all available processors shall be used. Overwrites the number of jobs
            specified in the config dict.
        batch_size: int, optional
            Batch size, i.e., number of samples processed in parallel.

        Returns
        -------
        Dict[Any, np.ndarray]
            Dict mapping model IDs to probability-arrays. If existing, the key of the entire ensemble is "__ensemble__".
        """
        raise NotImplementedError()

    def calibrate(self, y: np.ndarray) -> np.ndarray:
        if self.calibrator_ is not None:
            y = getattr(self._calibrator, self._calibrator_method)(y)
        return y

    def get_versions(self) -> dict:
        """
        Get the versions of all key packages and libraries this AutoML backend depends upon.

        Returns
        -------
        dict
            Dict whose keys are package names and whose values are version strings.
        """
        raise NotImplementedError()


for _d in Path(__file__).parent.iterdir():
    if _d.is_dir() and (_d / '__init__.py').exists():
        importlib.import_module('catabra.automl.' + _d.stem)
