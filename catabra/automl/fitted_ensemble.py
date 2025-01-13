#  Copyright (c) 2022-2025. RISC Software GmbH.
#  All rights reserved.

from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd
import sklearn
import sklearn.ensemble
from catabra_lib import metrics

from catabra.util import io


def _preprocess(x, pp: list):
    for step in pp:
        x = step.transform(x)
    return x


def _predict_proba(estimator, x, **kwargs):
    try:
        prediction = estimator.predict_proba(x, **kwargs)
    except:     # noqa
        # some estimators implement `decision_function()`, but not `predict_proba()`
        prediction = estimator.decision_function(x, **kwargs)

        # copied from autosklearn.pipeline.implementations.util.softmax()
        if prediction.ndim == 1:
            # positive values indicate positive class, negative values indicate negative class
            # => apply logit to get class probabilities
            np.clip(prediction, -20, 20, out=prediction)
            prediction = 1 / (1 + np.exp(-prediction))
            prediction = np.stack([1 - prediction, prediction], axis=-1)
        else:
            # Compute the Softmax like it is described here:
            # https://www.deeplearningbook.org/contents/numerical.html
            prediction -= np.max(prediction, axis=1, keepdims=True)
            prediction = np.exp(prediction)
            prediction /= np.sum(prediction, axis=1, keepdims=True)

    # multilabel output might be a list => convert to array
    # copied from autosklearn.pipeline.implementations.util.convert_multioutput_multiclass_to_multilabel()
    if isinstance(prediction, list):
        if prediction:
            assert all(isinstance(p, np.ndarray) and p.ndim == 2 for p in prediction)
            tmp = np.empty((prediction[0].shape[0], len(prediction)), dtype=prediction[0].dtype)
            for i, label_scores in enumerate(prediction):
                if label_scores.shape[1] == 1:
                    tmp[:, i] = label_scores
                elif label_scores.shape[1] == 2:
                    tmp[:, i] = label_scores[:, 1]
                elif label_scores.shape[1] > 2:
                    raise ValueError('Multioutput-Multiclass supported by'
                                     ' scikit-learn, but not by auto-sklearn!')
                else:
                    RuntimeError(f'Unknown predict_proba output={prediction}')

            prediction = tmp
        else:
            prediction = np.empty((len(x), 0), dtype=np.float32)

    np.clip(prediction, 0., 1., out=prediction)
    return prediction


def _predict_voting_regressor(estimator: sklearn.ensemble.VotingRegressor, x, **kwargs):
    # `VotingRegressor` is not meant for multi-output regression and hence averages on wrong axis
    # `VotingRegressor.transform()` returns array of shape `(n_samples, n_estimators)` in case of single target,
    # and `(n_targets, n_samples, n_estimators)` in case of multiple targets
    prediction = np.average(estimator.transform(x, **kwargs), axis=-1, weights=estimator._weights_not_none)
    if prediction.ndim == 2:
        prediction = prediction.T
    return prediction


def get_prediction_function(estimator, proba: bool = False):
    if isinstance(estimator, FittedModel):
        return estimator.predict_proba if proba else estimator.predict
    elif isinstance(estimator, sklearn.ensemble.VotingRegressor):
        return partial(_predict_voting_regressor, estimator)    # `partial` works well with multiprocessing
    elif proba:
        return partial(_predict_proba, estimator)               # `partial` works well with multiprocessing
    else:
        return estimator.predict


def _model_predict(model: Union[dict, 'FittedModel'], x, batch_size: Optional[int] = None, proba: bool = False,
                   copy: bool = False) -> np.ndarray:
    # copied and adapted from autosklearn.automl._model_predict()
    if copy:
        x = x.copy()
    if isinstance(model, dict):
        pp = model.get('preprocessing') or []
        if not isinstance(pp, (list, tuple)):
            pp = [pp]
        estimator = model['estimator']
    else:
        pp = model.preprocessing
        estimator = model.estimator
    x = _preprocess(x, pp)
    predict_func = get_prediction_function(estimator, proba=proba)
    if batch_size is not None and len(x) > batch_size and hasattr(estimator, 'batch_size'):
        return predict_func(x, batch_size=batch_size)
    else:
        return predict_func(x)


class FittedModel:
    """
    Canonical, uniform representation of fitted prediction models, independent of the method and backend used for
    fitting them. Ideally, the preprocessing transformations and estimator passed as arguments should be plain
    sklearn/XGBoost/TensorFlow/... objects, but this is no formal requirement.

    Note that the name "FittedModel" was chosen to be consistent with auto-sklearn's terminology. Strictly speaking,
    this class represents whole pipelines rather than just prediction models.


    Parameters
    ----------
    preprocessing:
        single preprocessing transformation or list of such transformations. Each step must implement the `transform()`
        method. `None` defaults to the empty list.
    estimator:
        Final estimator, must implement the `predict()` and, in case of classification, the `predict_proba()` method.

    Notes
    -----
    There is no clear definition of what constitutes a preprocessing step and what the final estimator.
    As a rule of thumb, try to set `estimator` to an atomic sklearn/XGBoost/TensorFlow/... model that cannot be
    decomposed into individual parts anymore, and put everything else into `preprocessing`.
    """

    def __init__(self, preprocessing=None, estimator=None):
        if estimator is None:
            raise ValueError('The estimator of a FittedModel cannot be None.')
        if preprocessing is None:
            preprocessing = []
        elif not isinstance(preprocessing, (list, tuple)):
            preprocessing = [preprocessing]
        self.preprocessing: list = preprocessing
        self.estimator = estimator

    def fit(self, x, y=None):
        # only to implement the standard sklearn API, which is required by functions like
        # `sklearn.metrics.check_scoring()`
        raise RuntimeError(f'Method fit() of class {self.__class__.__name__} cannot be called.')

    def transform(self, x: pd.DataFrame, batch_size: Optional[int] = None):
        """
        Transform given data by applying the preprocessing steps of this FittedModel object.

        Parameters
        ----------
        x: DataFrame
            Features DataFrame. Must have the exact same format as the data this FittedModel was trained on.
        batch_size: int, optional
            Batch size, i.e., number of samples processed in parallel.

        Returns
        -------
            Transformed data, array-like of shape `(n_samples, n_features)`.
        """
        if batch_size is None or len(x) <= batch_size:
            return _preprocess(x, self.preprocessing)
        else:
            out = [_preprocess(x0, self.preprocessing) for x0 in FittedModel._batch_iter(x, batch_size)]
            if isinstance(out[0], np.ndarray):
                return np.concatenate(out)
            elif isinstance(out[0], (pd.DataFrame, pd.Series)):
                return pd.concat(out, axis=0, sort=False)
            else:
                raise ValueError(f'Cannot concatenate preprocessed data of type {type(out[0])}')

    def predict(self, x: pd.DataFrame, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Apply this `FittedModel` object to given data.

        Parameters
        ----------
        x: DataFrame
            Features DataFrame. Must have the exact same format as the data this FittedModel was trained on.
        batch_size: int, optional
            Batch size, i.e., number of samples processed in parallel.

        Returns
        -------
        ndarray
            Array of predictions. In case of classification, these are class indicators rather than probabilities.
        """
        return _model_predict(self, x, batch_size=batch_size, proba=False, copy=False)

    def predict_proba(self, x: pd.DataFrame, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Apply this FittedModel object to given data. In contrast to method `predict()`, this method returns class
        probabilities in case of classification tasks. Does not work for regression tasks.

        Parameters
        ----------
        x: DataFrame
            Features DataFrame. Must have the exact same format as the data this FittedModel was trained on.
        batch_size: int, optional
            Batch size, i.e., number of samples processed in parallel.

        Returns
        -------
        ndarray
            Array of class probabilities, of shape `(n_samples, n_classes)`.
        """
        return _model_predict(self, x, batch_size=batch_size, proba=True, copy=False)

    def to_dict(self):
        return dict(preprocessing=self.preprocessing, estimator=self.estimator)

    def dump(self, fn: Union[Path, str], as_dict: bool = False):
        """
        Dump this FittedModel object to disk.

        Parameters
        ----------
        fn: Path | str
            File name.
        as_dict: bool, default=False
            Whether to convert this object into a dict before dumping it. Set to True for maximum portability.
        """
        io.dump(self.to_dict() if as_dict else self, fn)

    @classmethod
    def load(cls, fn: Union[Path, str]) -> 'FittedModel':
        """
        Load a dumped FittedModel from disk. The value of parameter `as_dict` upon dumping does not matter.

        Parameters
        ----------
        fn: Path | str
            File name.

        Returns
        -------
        FittedModel
            The loaded `FittedModel` object.
        """
        obj = io.load(fn)
        if isinstance(obj, dict):
            return cls(**obj)
        return obj

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(\n    preprocessing={self.preprocessing},\n    estimator={self.estimator})'

    @staticmethod
    def _batch_iter(x: pd.DataFrame, batch_size: int):
        assert batch_size > 0
        i = 0
        while i < len(x):
            yield x.iloc[i:i + batch_size]
            i += batch_size


class FittedEnsemble:
    """
    Canonical, uniform representation of fitted ensembles of prediction models, independent of the method and
    backend used for fitting them. Ideally, the constituent models and meta estimators passed as arguments should
    be plain sklearn/XGBoost/TensorFlow/... objects, but this is no formal requirement.

    Parameters
    ----------
    name: str, optional
        Optional, name of this ensemble (e.g., name of the AutoML-backend used for generating it).
    task: str, optional
        Prediction task, one of `"regression"`, `"binary_classification"`, `"multiclass_classification"` or
        `"multilabel_classification"`.
    models: dict, optional
        Constituent models of the ensemble, a non-empty dict. Each element must be a FittedModel or a dict of the form
        {
            "preprocessing": preprocessing,
            "estimator": estimator
        }
        which is used to initialize a `FittedModel` object. The keys of the dict serve as unique model-IDs.
    meta_input: list, optional
        List of model-IDs that serve as the input to the meta estimator (must be a subset of the keys of `models`).
        None defaults to all model-IDs, in the same order as in `models`.
    meta_estimator: optional
        Meta estimator that combines the outputs of the individual models into a final prediction. Must implement the
        `predict()` method and, in case of classification, the `predict_proba()` method. Alternatively, `estimator` can
        also be a list of weights with the same length as `meta_input`, in which case soft voting with the specified
        weights is employed. The weights are completely arbitrary, i.e., they can be negative and do not need to sum to
        1. None defaults to a soft voting estimator with uniform weights.
    calibrator:
        Calibrator, optional. If given, must be a fitted calibrator with a `transform()` or `predict()` method that
        takes the output of `meta_estimator` as input and returns an array of the same shape. May only be provided for
        classification tasks.
    """

    def __init__(self, name: Optional[str] = None, task: str = None, models: dict = None, meta_input: list = None,
                 meta_estimator=None, calibrator=None):
        if not models:
            raise ValueError('A FittedEnsemble cannot be instantiated without models.')
        if task not in ('regression', 'binary_classification', 'multiclass_classification',
                        'multilabel_classification'):
            raise ValueError(f'Unknown prediction task: {task}')
        if meta_input is None:
            meta_input = list(models)
        elif any(k not in models for k in meta_input):
            raise ValueError('Inputs of meta estimator must be models listed in `models`.')
        if meta_estimator is None:
            meta_estimator = [1 / len(meta_input)] * len(meta_input)
        elif isinstance(meta_estimator, (list, tuple)):
            if len(meta_estimator) != len(meta_input):
                raise ValueError(f'List of meta estimator weights ({len(meta_estimator)})'
                                 f' differs from number of inputs ({len(meta_input)})')
        self.name: Optional[str] = name
        self.task: str = task
        self.models_: dict = {k: FittedModel(**m) if isinstance(m, dict) else m for k, m in models.items()}
        self.meta_input_: list = meta_input
        self.meta_estimator_ = meta_estimator
        self.calibrator_ = calibrator

    @property
    def model_ids_(self) -> list:
        """
        List of IDs for accessing individual (constituent) models of the final ensemble.
        """
        return list(self.models_)

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

    def fit(self, x, y=None):
        # only to implement the standard sklearn API, which is required by functions like
        # `sklearn.metrics.check_scoring()`
        raise RuntimeError(f'Method fit() of class {self.__class__.__name__} cannot be called.')

    def predict(self, x: pd.DataFrame, jobs: int = 1, batch_size: Optional[int] = None, model_id=None,
                calibrated: bool = 'auto') -> np.ndarray:
        """
        Apply trained models to given data.

        Parameters
        ----------
        x: DataFrame
            Features DataFrame. Must have the exact same format as the data this FittedEnsemble was trained on.
        jobs: int, optional
            The number of jobs to use, or -1 if all available processors shall be used.
        batch_size: int, optional
            Batch size, i.e., number of samples processed in parallel.
        model_id: optional
            The ID of the model to apply, as in `model_ids_`. If None, the whole ensemble is applied.
        calibrated: bool | 'auto', default='auto'
            Whether to return calibrated predictions in case of classification tasks. If `"auto"`, calibrated
            predictions are returned iff `model_id` is `None`; note that only the entire ensemble is calibrated, not
            individual constituents.

        Returns
        -------
        ndarray
            Array of predictions. In case of classification, these are class indicators rather than probabilities.
        """
        if model_id is None:
            y = self._predict_all(x, jobs, batch_size, calibrated is not False, False)['__ensemble__']
        elif calibrated is True and self.calibrator_ is not None:
            # must be classification task
            y = self.calibrate(self.models_[model_id].predict_proba(x, batch_size=batch_size))
            if self.task == 'multiclass_classification':
                y = metrics.multiclass_proba_to_pred(y)
            else:
                y = (y >= 0.5).astype(np.int32)
        else:
            y = self.models_[model_id].predict(x, batch_size=batch_size)

        return y

    def predict_proba(self, x: pd.DataFrame, jobs: int = 1, batch_size: Optional[int] = None,
                      model_id=None, calibrated: bool = 'auto') -> np.ndarray:
        """
        Apply trained models to given data. In contrast to method `predict()`, this method returns class probabilities
        in case of classification tasks. Does not work for regression tasks.

        Parameters
        ----------
        x: DataFrame
            Features DataFrame. Must have the exact same format as the data this FittedEnsemble was trained on.
        jobs: int, optional
            The number of jobs to use, or -1 if all available processors shall be used.
        batch_size: int, optional
            Batch size, i.e., number of samples processed in parallel.
        model_id:
            The ID of the model to apply, as in `model_ids_`. If None, the whole ensemble is applied.
        calibrated: bool | 'auto', default='auto'
            Whether to return calibrated predictions in case of classification tasks. If "auto", calibrated predictions
            are returned iff `model_id` is None; note that only the entire ensemble is calibrated, not individual
            constituents.

        Returns
        -------
        ndarray
            Array of class probabilities, of shape `(n_samples, n_classes)`.
        """
        if model_id is None:
            y = self._predict_all(x, jobs, batch_size, calibrated is not False, True)['__ensemble__']
        else:
            y = self.models_[model_id].predict_proba(x, batch_size=batch_size)
            if calibrated is True:
                y = self.calibrate(y)
        return y

    def predict_all(self, x: pd.DataFrame, jobs: int = 1, batch_size: Optional[int] = None,
                    calibrated_ensemble: bool = False) -> Dict[Any, np.ndarray]:
        """
        Apply all trained models to given data, including constituent models and the entire ensemble. In classification
        tasks, class probabilities are returned for all constituent models and class indices only for the whole
        ensemble.

        Parameters
        ----------
        x: DataFrame
            Features DataFrame. Must have the exact same format as the data this FittedEnsemble was trained on.
        jobs: int, optional
            The number of jobs to use, or -1 if all available processors shall be used.
        batch_size: int, optional
            Batch size, i.e., number of samples processed in parallel.
        calibrated_ensemble: bool, default=False
            Whether to return calibrated predictions of the whole ensemble. Does not affect constituent models, whose
            predictions are always returned uncalibrated. This can lead to awkward situations for singleton ensembles
            when predictions of the sole model are returned both calibrated und uncalibrated.

        Returns
        -------
        Dict[Any, np.ndarray]
            Dict mapping model IDs to prediction-arrays. The key of the entire ensemble is "__ensemble__".
        """
        return self._predict_all(x, jobs, batch_size, calibrated_ensemble, False)

    def predict_proba_all(self, x: pd.DataFrame, jobs: int = 1, batch_size: Optional[int] = None,
                          calibrated_ensemble: bool = False) -> Dict[Any, np.ndarray]:
        """
        Apply all trained models to given data, including constituent models and the entire ensemble. In contrast to
        method `predict_all()`, this method returns class probabilities in case of classification tasks. Does not work
        for regression tasks.

        Parameters
        ----------
        x: DataFrame
            Features DataFrame. Must have the exact same format as the data this FittedEnsemble was trained on.
        jobs: int, optional
            The number of jobs to use, or -1 if all available processors shall be used.
        batch_size: int, optional
            Batch size, i.e., number of samples processed in parallel.
        calibrated_ensemble: bool, default=False
            Whether to return calibrated predictions of the whole ensemble. Does not affect constituent models, whose
            predictions are always returned uncalibrated. This can lead to awkward situations for singleton ensembles
            when predictions of the sole model are returned both calibrated und uncalibrated.

        Returns
        -------
        Dict[Any, np.ndarray]:
            Dict mapping model IDs to probability-arrays. The key of the entire ensemble is "__ensemble__".
        """
        return self._predict_all(x, jobs, batch_size, calibrated_ensemble, True)

    def calibrate(self, y: np.ndarray) -> np.ndarray:
        if self.calibrator_ is not None:
            y = getattr(self._calibrator, self._calibrator_method)(y)
        return y

    def to_dict(self) -> dict:
        """Convert this FittedEnsemble object into a dict."""
        return dict(
            name=self.name,
            task=self.task,
            models={k: m.to_dict() for k, m in self.models_.items()},
            meta_input=self.meta_input_,
            meta_estimator=self.meta_estimator_,
            calibrator=self.calibrator_
        )

    def dump(self, fn: Union[Path, str], as_dict: bool = False):
        """
        Dump this FittedEnsemble object to disk.

        Parameters
        ----------
        fn: Path | str
            File name.
        as_dict: bool, default=False
            Whether to convert this object into a dict before dumping it. Set to True for maximum portability.
        """
        io.dump(self.to_dict() if as_dict else self, fn)

    @classmethod
    def load(cls, fn: Union[Path, str]) -> 'FittedEnsemble':
        """
        Load a dumped FittedEnsemble from disk. The value of parameter `as_dict` upon dumping does not matter.

        Parameters
        ----------
        fn: Path | str
            File name.

        Returns
        -------
        FittedEnsemble
            The loaded FittedEnsemble object.
        """
        obj = io.load(fn)
        if isinstance(obj, dict):
            return cls(**obj)
        return obj

    def __repr__(self) -> str:
        dct = {k: v.estimator.__class__.__name__ for k, v in self.models_.items()}
        return f'{self.__class__.__name__}(models={dct}, task="{self.task}")'

    def _predict_all(self, x: pd.DataFrame, jobs: int, batch_size: Optional[int], calibrated_ensemble: bool,
                     proba: bool) -> Dict[Any, np.ndarray]:
        calibrated_ensemble = calibrated_ensemble and self.calibrator_ is not None

        # get uncalibrated predictions of all constituent models
        all_predictions = joblib.Parallel(n_jobs=jobs)(
            joblib.delayed(_model_predict)(
                self.models_[key],
                x,
                batch_size=batch_size,
                # always return class probabilities, because that's what the meta estimator expects
                proba=self.task != 'regression',
                copy=True
            )
            for key in self.meta_input_
        )
        out = {k: v for k, v in zip(self.meta_input_, all_predictions)}

        if isinstance(self.meta_estimator_, (list, tuple)):
            y = sum(w * p for w, p in zip(self.meta_estimator_, all_predictions))
            if calibrated_ensemble:
                y = self.calibrate(y)
            if proba:
                np.clip(y, 0., 1., out=y)
            elif self.task == 'multiclass_classification':
                y = metrics.multiclass_proba_to_pred(y)
            elif self.task != 'regression':
                y = (y >= 0.5).astype(np.int32)
        elif proba or calibrated_ensemble:
            # must be classification task
            y = self.meta_estimator_.predict_proba(all_predictions)
            if calibrated_ensemble:
                y = self.calibrate(y)
            if proba:
                np.clip(y, 0., 1., out=y)
            elif self.task == 'multiclass_classification':
                y = metrics.multiclass_proba_to_pred(y)
            else:
                y = (y >= 0.5).astype(np.int32)
        else:
            y = self.meta_estimator_.predict(all_predictions)

        out['__ensemble__'] = y

        return out
