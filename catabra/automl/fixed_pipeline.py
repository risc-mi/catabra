#  Copyright (c) 2022-2025. RISC Software GmbH.
#  All rights reserved.

import time as py_time  # otherwise shadowed by parameter of method `fit()`
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from catabra_lib import preprocessing
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline

from catabra.automl.base import AutoMLBackend
from catabra.automl.fitted_ensemble import FittedEnsemble, _model_predict
from catabra.util import logging


def _flatten_transformer_pipeline(obj) -> list:
    if obj in ('passthrough', None):
        return []
    elif isinstance(obj, Pipeline) and obj.steps[-1] in ('passthrough', None):
        return [t for _, s in obj.steps[:-1] for t in _flatten_transformer_pipeline(s)]
    else:
        return [obj]


class _FixedPipelineBackend(AutoMLBackend):

    def __init__(self, _name: str, _preprocessing=None, _estimator=None, **kwargs):
        super(_FixedPipelineBackend, self).__init__(**kwargs)
        self._name = _name
        self._preprocessing = _preprocessing
        self._estimator = _estimator

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_ids_(self) -> list:
        if self.estimator_ is None:
            raise ValueError('Fixed pipelines must be fit to training data before model_ids_ can be accessed.')
        return [0]

    def summary(self) -> dict:
        steps = self._get_preprocessing_steps()
        steps_summary = []
        for s in steps:
            try:
                txt = ' '.join(repr(s).replace('\n', ' ').split())
            except:  # noqa
                txt = s.__class__.__name__ + '(<unknown params>)'
            steps_summary.append(txt)
        out = dict(model_id=0, preprocessing=steps_summary)
        try:
            out['estimator'] = ' '.join(repr(self.estimator_).replace('\n', ' ').split())
        except:  # noqa
            out['estimator'] = self.estimator_.__class__.__name__ + '(<unknown params>)'

        return dict(
            automl=self.name,
            task=self.task,
            models=[out]
        )

    def training_history(self) -> pd.DataFrame:
        start = pd.Timestamp(self.fit_start_time_, unit='s', tz='utc').tz_convert(py_time.tzname[0]).tz_localize(None)
        end = pd.Timestamp(self.fit_end_time_, unit='s', tz='utc').tz_convert(py_time.tzname[0]).tz_localize(None)
        return pd.DataFrame(
            data={
                'model_id': 0,
                'timestamp': end,
                'total_elapsed_time': end - start,
                'type': self.estimator_.__class__.__name__
            },
            index=pd.RangeIndex(1)
        )

    def fitted_ensemble(self, ensemble_only: bool = True) -> FittedEnsemble:
        return FittedEnsemble(
            name=self.name,
            task=self.task,
            models={0: dict(preprocessing=self._get_preprocessing_steps(), estimator=self.estimator_)},
            calibrator=self.calibrator_
        )

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame, groups: Optional[np.ndarray] = None,
            sample_weights: Optional[np.ndarray] = None, time: Optional[int] = None, memory: Optional[int] = None,
            jobs: Optional[int] = None, dataset_name: Optional[str] = None, monitor=None) -> '_FixedPipelineBackend':
        self.fit_start_time_ = py_time.time()
        if time is not None:
            logging.warn('Time limits cannot be imposed on fixed pipelines.')
        if memory is not None:
            logging.warn('Memory limits cannot be imposed on fixed pipelines.')
        if jobs is None:
            jobs = self.config.get('jobs')
        if groups is not None:
            logging.warn('Fixed pipelines do not support sample grouping.')
        if monitor is not None:
            logging.warn('Fixed pipelines do not support monitoring the training process.')

        prefix = self.name + '_'
        specific = {k[len(prefix):]: v for k, v in self.config.items() if k.startswith(prefix)}
        preproc_config = {k[15:]: v for k, v in specific.items() if k.startswith('preprocessing__')}
        estimator_config = {k[11:]: v for k, v in specific.items() if k.startswith('estimator__')}

        if self._preprocessing is None:
            self.preprocessing_ = None
        else:
            try:
                self.preprocessing_ = clone(self._preprocessing)
            except:  # noqa
                self.preprocessing_ = self._preprocessing
            if preproc_config:
                # if this fails, we raise an exception
                self.preprocessing_.set_params(**preproc_config)
            if jobs is not None:
                try:
                    self.preprocessing_.set_params(jobs=jobs)
                except:  # noqa
                    try:
                        self.preprocessing_.set_params(n_jobs=jobs)
                    except:  # noqa
                        logging.warn(
                            f'Could not set number of jobs of {self.preprocessing_.__class__.__name__}'
                            f' preprocessing to {jobs}.'
                        )
        try:
            self.estimator_ = clone(self._estimator)
        except:  # noqa
            self.estimator_ = self._estimator
        if estimator_config:
            # if this fails, we raise an exception
            self.estimator_.set_params(**estimator_config)
        if jobs is not None:
            try:
                self.estimator_.set_params(jobs=jobs)
            except:  # noqa
                try:
                    self.estimator_.set_params(n_jobs=jobs)
                except:  # noqa
                    logging.warn(
                        f'Could not set number of jobs of {self.estimator_.__class__.__name__} estimator to {jobs}.'
                    )

        if self.task in ('binary_classification', 'multiclass_classification'):
            assert y_train.shape[1] == 1
            y_train = y_train.iloc[:, 0]

        if self.preprocessing_ is not None:
            x_train = self.preprocessing_.fit_transform(x_train, y=y_train)
        self.estimator_.fit(x_train, y_train, sample_weight=sample_weights)

        self.fit_end_time_ = py_time.time()

        return self

    def predict(self, x: pd.DataFrame, jobs: Optional[int] = None, batch_size: Optional[int] = None, model_id=None,
                calibrated: bool = 'auto') -> np.ndarray:
        if model_id not in (None, 0):
            raise ValueError(f'Unknown model_id {model_id} of fixed pipeline specified; only known ID is 0.')
        if model_id is None:
            calibrated = calibrated is not False
        else:
            calibrated = calibrated is True
        calibrated = calibrated and self.calibrator_ is not None

        if calibrated:
            y = self.predict_proba(x, jobs=jobs, batch_size=batch_size, model_id=model_id, calibrated=True)
            if self.task == 'multiclass_classification':
                y = np.argmax(y, axis=1)
            else:
                y = (y > 0.5).astype(np.int32)
        else:
            y = self._predict(x, batch_size, False)

        return y

    def predict_proba(self, x: pd.DataFrame, jobs: Optional[int] = None, batch_size: Optional[int] = None,
                      model_id=None, calibrated: bool = 'auto') -> np.ndarray:
        if model_id not in (None, 0):
            raise ValueError(f'Unknown model_id {model_id} of fixed pipeline specified; only known ID is 0.')
        if model_id is None:
            calibrated = calibrated is not False
        else:
            calibrated = calibrated is True
        y = self._predict(x, batch_size, True)
        if calibrated:
            y = self.calibrate(y)
        return y

    def predict_all(self, x: pd.DataFrame, jobs: Optional[int] = None, batch_size: Optional[int] = None) \
            -> Dict[Any, np.ndarray]:
        return {0: self.predict(x, jobs=jobs, batch_size=batch_size, calibrated=False)}

    def predict_proba_all(self, x: pd.DataFrame, jobs: Optional[int] = None, batch_size: Optional[int] = None) \
            -> Dict[Any, np.ndarray]:
        return {0: self.predict_proba(x, jobs=jobs, batch_size=batch_size, calibrated=False)}

    def get_versions(self) -> dict:
        return {}

    def _predict(self, x: pd.DataFrame, batch_size: Optional[int], proba: bool) -> np.ndarray:
        return _model_predict(
            dict(preprocessing=self._get_preprocessing_steps(), estimator=self.estimator_),
            x,
            batch_size=batch_size,
            proba=proba,
            copy=True
        )

    def _get_preprocessing_steps(self) -> list:
        return _flatten_transformer_pipeline(self.preprocessing_)


def register_backend(name: str, preprocessing=None, estimator=None):
    """
    Register a fixed pipeline as new AutoML backend. Tne term "AutoML" is strictly speaking not appropriate here, since
    hyperparameters are not automatically optimized.

    Parameters
    ----------
    name: str
        The name of the pipeline. It can be activated by setting config parameter `"automl"` to this name.
    preprocessing: optional
        Optional preprocessing steps. If given, must be an object implement scikit-learn's transformer API, in
        particular methods `fit_transform()` and `transform()`. It should also subclass `sklearn.base.BaseEstimator` for
        being able to get/set hyperparameters with `get_params()` and `set_params()`, respectively.
    estimator: optional
        The final estimator (classifier/regressor). Must be an object implement scikit-learn's estimator API, in
        particular methods `fit()`, `predict()` and, if used for classification, `predict_proba()`.  It should also
        subclass `sklearn.base.BaseEstimator` for being able to get/set hyperparameters with `get_params()` and
        `set_params()`, respectively.
    """
    if estimator is None:
        raise ValueError('The final estimator of a pipeline cannot be None.')
    elif not (hasattr(estimator, 'fit') and hasattr(estimator, 'predict')):
        raise TypeError(f"estimator must implement fit() and predict(); {estimator} (type {type(estimator)}) doesn't.")
    if preprocessing is not None \
            and not (hasattr(preprocessing, 'fit_transform') and hasattr(preprocessing, 'transform')):
        raise TypeError(
            "preprocessing must be a transformer and implement fit_transform() and transform();"
            f" {preprocessing} (type {type(preprocessing)}) doesn't."
        )

    def _fixed_pipeline_factory(**kwargs):
        assert all(k not in kwargs for k in ('_name', '_preprocessing', '_estimator'))
        return _FixedPipelineBackend(_name=name, _preprocessing=preprocessing, _estimator=estimator, **kwargs)

    AutoMLBackend.register(name, _fixed_pipeline_factory)


###############################################
#           Predefined Pipelines
###############################################


def standard_preprocessing() -> preprocessing.DTypeTransformer:
    """
    Construct a transformer that scales numerical and time-like columns to the range [0, 1], one-hot encodes
    categorical columns, and imputes missing numerical values with -1 (after scaling).

    Returns
    -------
    DTypeTransformer
        The transformer object. Parameters of its components can be retrieved and set via `get_params()` and
        `set_params()`, respectively, following the scikit-learn convention of composing names with the infix "__".
        Example: `set_params(num_transformer__simpleimputer__strategy="mean")`.
    """
    return preprocessing.DTypeTransformer(
        num=make_pipeline(
            preprocessing.MinMaxScaler(fit_bool=False),
            SimpleImputer(strategy='constant', fill_value=-1),
            'passthrough'
        ),
        cat=preprocessing.OneHotEncoder(drop_na=True),
        obj='drop',
        bool='num',     # cast to float by setting False to 0 and True to 1
        timedelta='num',
        datetime='num',
        timedelta_resolution='s',
        datetime_resolution='s'
    )
