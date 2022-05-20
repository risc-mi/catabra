from typing import Union, Optional, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import joblib

from ..util import io


def _preprocess(x, pp: list):
    for step in pp:
        x = step.transform(x)
    return x


def _model_predict(model: dict, x, batch_size: Optional[int] = None, proba: bool = False,
                   copy: bool = False) -> np.ndarray:
    # copied and adapted from autosklearn.automl._model_predict()
    if copy:
        x = x.copy()
    pp = model.get('preprocessing') or []
    if not isinstance(pp, (list, tuple)):
        pp = [pp]
    x = _preprocess(x, pp)
    estimator = model['estimator']
    if isinstance(estimator, sklearn.ensemble.VotingRegressor):
        # `VotingRegressor` is not meant for multi-output regression and hence averages on wrong axis
        # `VotingRegressor.transform()` returns array of shape `(n_samples, n_estimators)` in case of single target,
        # and `(n_targets, n_samples, n_estimators)` in case of multiple targets
        prediction = np.average(estimator.transform(x), axis=-1, weights=estimator._weights_not_none)       # noqa
        if prediction.ndim == 2:
            prediction = prediction.T
    else:
        predict_func = estimator.predict_proba if proba else estimator.predict
        if batch_size is not None and len(x) > batch_size and hasattr(estimator, 'batch_size'):
            prediction = predict_func(x, batch_size=batch_size)
        else:
            prediction = predict_func(x)
        if proba:
            # multilabel output might be a list => convert into array
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
                            RuntimeError(f'Unkown predict_proba output={prediction}')

                    prediction = tmp
                else:
                    prediction = np.empty((len(x), 0), dtype=np.float32)

            np.clip(prediction, 0., 1., out=prediction)

    return prediction


class FittedEnsemble:

    def __init__(self, name: Optional[str] = None, task: str = None, models: dict = None, voting_input: list = None,
                 voting_estimator=None):
        """
        Canonical, uniform representation of fitted ensembles of prediction models, independent of the method and
        backend used for fitting them. Ideally, the constituent models and voting estimators passed as arguments should
        be plain sklearn/XGBoost/TensorFlow/... objects, but this is no formal requirement.

        :param name: Optional, name of this ensemble (e.g., name of the AutoML-backend used for generating it).
        :param task: Prediction task, one of "regression", "binary_classification", "multiclass_classification" or
        "multilabel_classification".
        :param models: Constituent models of the ensemble, a non-empty dict. Each element must be a dict of the form

            {
                "preprocessing": preprocessing,
                "estimator": estimator
            }

        where `preprocessing` is an optional (list of) preprocessing step(s) and `estimator` is the final estimator.
        The preprocessing steps must implement the `transform()` method, and the final estimator must implement the
        `predict()` method and, in case of classification, the `predict_proba()` method.
        Strictly speaking, there is no clear definition of what constitutes a preprocessing step and what the final
        estimator. As a rule of thumb, try to set `estimator` to an atomic sklearn/XGBoost/TensorFlow/... model that
        cannot be decomposed into individual parts any more, and put everything else into `preprocessing`.
        The keys of the dict serve as unique model-IDs.

        :param voting_input: List of model-IDs that serve as the input to the voting estimator (must be a subset of the
        keys of `models`). None defaults to all model-IDs, in the same order as in `models`.
        :param voting_estimator: Voting estimator that combines the outputs of the individual models into a final
        prediction. Must implement the `predict()` method and, in case of classification, the `predict_proba()` method.
        Alternatively, `estimator` can also be a list of weights with the same length as `voting_input`, in which case
        soft voting with the specified weights is employed.
        None defaults to a soft voting estimator with uniform weights.
        """

        if not models:
            raise ValueError('A FittedEnsemble cannot be instantiated without models.')
        if task not in ('regression', 'binary_classification', 'multiclass_classification',
                        'multilabel_classification'):
            raise ValueError(f'Unknown prediction task: {task}')
        if voting_input is None:
            voting_input = list(models)
        elif any(k not in models for k in voting_input):
            raise ValueError('Inputs of voting estimator must be models listed in `models`.')
        if voting_estimator is None:
            voting_estimator = [1] * len(voting_input)
        elif isinstance(voting_estimator, (list, tuple)):
            if len(voting_estimator) != len(voting_input):
                raise ValueError(f'List of voting estimator weights ({len(voting_estimator)})'
                                 f' differs from number of inputs ({len(voting_input)})')
            elif sum(voting_estimator) <= 0:
                raise ValueError(
                    'Sum of voting estimator weights must be > 0, but is {:.4f}'.format(sum(voting_estimator))
                )
            elif any(w < 0 for w in voting_estimator):
                raise ValueError('Voting estimator weights must not be < 0')
        self.name: Optional[str] = name
        self.task: str = task
        self.models_: dict = models
        self.voting_input_: list = voting_input
        self.voting_estimator_ = voting_estimator

    @property
    def model_ids_(self) -> list:
        """List of IDs for accessing individual (constituent) models of the final ensemble."""
        return list(self.models_)

    def transform(self, x: pd.DataFrame, model_id, batch_size: Optional[int] = None):
        """
        Transform given data by applying the preprocessing steps of a single model.
        :param x: Features DataFrame. Must have the exact same format as the data this FittedEnsemble was trained on.
        :param model_id: The ID of the model whose preprocessing steps to apply, as in `model_ids_`.
        :param batch_size: Batch size, i.e., number of samples processed in parallel.
        :return: Transformed data, array-like of shape `(n_samples, n_features)`.
        """
        pp = self.models_[model_id].get('preprocessing') or []
        if not isinstance(pp, (list, tuple)):
            pp = [pp]
        if batch_size is None or len(x) <= batch_size:
            return _preprocess(x, pp)
        else:
            out = [_preprocess(x0, pp) for x0 in FittedEnsemble._batch_iter(x, batch_size)]
            if isinstance(out[0], np.ndarray):
                return np.concatenate(out)
            elif isinstance(out[0], (pd.DataFrame, pd.Series)):
                return pd.concat(out, axis=0, sort=False)
            else:
                raise ValueError(f'Cannot concatenate preprocessed data of type {type(out[0])}')

    def predict(self, x: pd.DataFrame, jobs: int = 1, batch_size: Optional[int] = None, model_id=None) -> np.ndarray:
        """
        Apply trained models to given data.
        :param x: Features DataFrame. Must have the exact same format as the data this FittedEnsemble was trained on.
        :param jobs: The number of jobs to use, or -1 if all available processors shall be used.
        :param batch_size: Batch size, i.e., number of samples processed in parallel.
        :param model_id: The ID of the model to apply, as in `model_ids_`. If None, the whole ensemble is applied.
        :return: Array of predictions. In case of classification, these are class indicators rather than probabilities.
        """
        if model_id is None:
            return self._predict_all(x, jobs, batch_size, False)['__ensemble__']
        else:
            return self._predict_single(x, model_id, batch_size, False)

    def predict_proba(self, x: pd.DataFrame, jobs: int = 1, batch_size: Optional[int] = None,
                      model_id=None) -> np.ndarray:
        """
        Apply trained models to given data. In contrast to method `predict()`, this method returns class probabilities
        in case of classification tasks. Does not work for regression tasks.
        :param x: Features DataFrame. Must have the exact same format as the data this FittedEnsemble was trained on.
        :param jobs: The number of jobs to use, or -1 if all available processors shall be used.
        :param batch_size: Batch size, i.e., number of samples processed in parallel.
        :param model_id: The ID of the model to apply, as in `model_ids_`. If None, the whole ensemble is applied.
        :return: Array of class probabilities, of shape `(n_samples, n_classes)`.
        """
        if model_id is None:
            return self._predict_all(x, jobs, batch_size, True)['__ensemble__']
        else:
            return self._predict_single(x, model_id, batch_size, True)

    def predict_all(self, x: pd.DataFrame, jobs: int = 1, batch_size: Optional[int] = None) -> Dict[Any, np.ndarray]:
        """
        Apply all trained models to given data, including constituent models and the entire ensemble.
        :param x: Features DataFrame. Must have the exact same format as the data this FittedEnsemble was trained on.
        :param jobs: The number of jobs to use, or -1 if all available processors shall be used.
        :param batch_size: Batch size, i.e., number of samples processed in parallel.
        :return: Dict mapping model IDs to prediction-arrays. The key of the entire ensemble is "__ensemble__".
        """
        return self._predict_all(x, jobs, batch_size, False)

    def predict_proba_all(self, x: pd.DataFrame, jobs: int = 1,
                          batch_size: Optional[int] = None) -> Dict[Any, np.ndarray]:
        """
        Apply all trained models to given data, including constituent models and the entire ensemble. In contrast to
        method `predict_all()`, this method returns class probabilities in case of classification tasks. Does not work
        for regression tasks.
        :param x: Features DataFrame. Must have the exact same format as the data this FittedEnsemble was trained on.
        :param jobs: The number of jobs to use, or -1 if all available processors shall be used.
        :param batch_size: Batch size, i.e., number of samples processed in parallel.
        :return: Dict mapping model IDs to probability-arrays. The key of the entire ensemble is "__ensemble__".
        """
        return self._predict_all(x, jobs, batch_size, True)

    def to_dict(self) -> dict:
        """Convert this FittedEnsemble object into a dict."""
        return dict(
            name=self.name,
            task=self.task,
            models=self.models_,
            voting_input=self.voting_input_,
            voting_estimator=self.voting_estimator_
        )

    def dump(self, fn: Union[Path, str], as_dict: bool = False):
        """
        Dump this FittedEnsemble object to disk.
        :param fn: File name.
        :param as_dict: Whether to convert this object into a dict before dumping it.
        Set to True for maximum portability.
        """
        io.dump(self.to_dict() if as_dict else self, fn)

    @classmethod
    def load(cls, fn: Union[Path, str]) -> 'FittedEnsemble':
        """
        Load a dumped FittedEnsemble from disk. The value of parameter `as_dict` upon dumping does not matter.
        :param fn: File name.
        :return: FittedEnsemble object.
        """
        obj = io.load(fn)
        if isinstance(obj, dict):
            return cls(**obj)
        return obj

    def _predict_single(self, x: pd.DataFrame, model_id, batch_size: Optional[int], proba: bool) -> np.ndarray:
        # get predictions of a single constituent model
        return _model_predict(self.models_[model_id], x, batch_size=batch_size, proba=proba, copy=False)

    def _predict_all(self, x: pd.DataFrame, jobs: int, batch_size: Optional[int], proba: bool) -> Dict[Any, np.ndarray]:
        # get predictions of all constituent models and entire ensemble
        all_predictions = joblib.Parallel(n_jobs=jobs)(
            joblib.delayed(_model_predict)(
                self.models_[key],
                x,
                batch_size=batch_size,
                # always return class probabilities, because that's what the voting estimator expects
                proba=self.task != 'regression',
                copy=True
            )
            for key in self.voting_input_
        )
        out = {k: v for k, v in zip(self.voting_input_, all_predictions)}
        if isinstance(self.voting_estimator_, (list, tuple)):
            pred = sum(w * p for w, p in zip(self.voting_estimator_, all_predictions)) / sum(self.voting_estimator_)
            if proba:
                np.clip(pred, 0., 1., out=pred)
            elif self.task == 'multilabel_classification':
                pred = pred >= 0.5
            else:
                pred = np.argmax(pred, axis=1)
        elif proba:
            pred = self.voting_estimator_.predict_proba(all_predictions)
            np.clip(pred, 0., 1., out=pred)
        else:
            pred = self.voting_estimator_.predict(all_predictions)
        out['__ensemble__'] = pred

        return out

    @staticmethod
    def _batch_iter(x: pd.DataFrame, batch_size: int):
        assert batch_size > 0
        i = 0
        while i < len(x):
            yield x.iloc[i:i + batch_size]
            i += batch_size
