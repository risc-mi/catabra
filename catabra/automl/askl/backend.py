from typing import Optional, Dict, Any, Tuple, Iterable
import shutil
import inspect
import re
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import joblib
import logging as py_logging
import time as py_time      # otherwise shadowed by parameter of method `fit()`
import yaml
import importlib
from smac.callbacks import IncorporateRunResultCallback
from smac.tae import StatusType

from ...util import io
from ...util import logging
from ...util.common import repr_timedelta
from ...analysis import grouped_split
from ..base import FittedEnsemble, AutoMLBackend
from .scorer import get_scorer
from . import explanation


explanation.TransformationExplainer.register_factory('auto-sklearn', explanation.askl_explainer_factory,
                                                     errors='ignore')


# load add-ons
for _d in (Path(__file__).parent / 'addons').iterdir():
    if _d.suffix.lower() == '.py':
        try:
            importlib.import_module('.addons.' + _d.stem, package=__package__)
        except ImportError:
            # required packages are not available => skip
            pass


class _EnsembleLoggingHandler(py_logging.Handler):
    """Logging handler for printing _comprehensible_ messages whenever a new ensemble has been fit.
    Solution is a bit hacky and can easily break if some internals of auto-sklearn change."""

    def __init__(self, metric: str = 'cost', optimum: str = '0.', sign: str = '-1.', start_time: str = '0.', **kwargs):
        super(_EnsembleLoggingHandler, self).__init__(**kwargs)
        self.metric = metric
        self.optimum = float(optimum)
        self.sign = float(sign)
        self.start_time = float(start_time)
        # Counting ensembles analogous to models in `_SMACLoggingCallback` does not work, as the counter is
        # never increased. This might be because this class is being re-instantiated again and again.

    def emit(self, record: py_logging.LogRecord):
        if record.levelname == 'INFO':
            if hasattr(record.msg, 'weights_') and hasattr(record.msg, 'trajectory_'):
                n_models = sum(w > 0 for w in record.msg.weights_)
                value = self.optimum - self.sign * record.msg.trajectory_[-1]
                try:
                    logging.log(
                        'New ensemble fitted:\n'
                        '    ensemble_val_{:s}: {:f}\n'
                        '    n_constituent_models: {:d}\n'
                        '    total_elapsed_time: {:s}'
                        .format(self.metric, value, n_models, repr_timedelta(record.created - self.start_time))
                    )
                except RecursionError:
                    raise
                except:     # noqa
                    self.handleError(record)


# Hacky, but class must be in some installed package such that `logging` finds it. We do not want to rely on CaTabRa
# being installed.
setattr(py_logging, '_AutoSklearnHandler', _EnsembleLoggingHandler)


class _SMACLoggingCallback(IncorporateRunResultCallback):
    """Callback for printing messages whenever a new model has been trained."""

    def __init__(self, main_metric: Tuple[str, float, float], other_metrics: Iterable[Tuple[str, float, float]],
                 estimator_name: str, start_time: float = 0.):
        self.main_metric = main_metric
        self.other_metrics = other_metrics
        self.estimator_choice = estimator_name + ':__choice__'
        self.start_time = start_time
        self._n = 0     # interestingly, counting models seems to work even if multiple jobs are used

    def __call__(self, smbo: 'SMBO', run_info: 'RunInfo', result: 'RunValue', time_left: float) -> Optional[bool]:
        try:
            if result.status == StatusType.SUCCESS and result.additional_info:
                self._n += 1
                val, train, test, other = _get_metrics_from_run_value(result, self.main_metric, self.other_metrics)
                msg = 'New model #{:d} trained:\n    val_{:s}: {:f}\n'.format(self._n, self.main_metric[0], val)
                for k, v in other.items():
                    msg += '    val_{:s}: {:f}\n'.format(k, v)
                if not pd.isna(test):
                    msg += '    test_{:s}: {:f}\n'.format(self.main_metric[0], test)
                if not pd.isna(train):
                    msg += '    train_{:s}: {:f}\n'.format(self.main_metric[0], train)
                msg += '    type: {:s}\n' \
                       '    total_elapsed_time: {:s}'.format(run_info.config._values[self.estimator_choice],
                                                             repr_timedelta(result.endtime - self.start_time))
                logging.log(msg)
        except:  # noqa
            pass
        return None


def _get_metrics_from_run_value(run_value: 'RunValue', main_metric: Tuple[str, float, float],
                                other_metrics: Iterable[Tuple[str, float, float]]) -> Tuple[float, float, float, dict]:
    # metrics must be passed as triples `(name, optimum, sign)`

    val = main_metric[1] - (main_metric[2] * run_value.cost)
    train = main_metric[1] - (main_metric[2] * run_value.additional_info['train_loss'])
    test = main_metric[1] - (main_metric[2] * run_value.additional_info.get('test_loss', np.nan))

    # additional metrics are only available for validation set for single models,
    # even if test data is provided
    other = {}
    for metric in other_metrics:
        cost = run_value.additional_info.get(metric[0])
        if cost is not None:
            other[metric[0]] = metric[1] - (metric[2] * cost)

    return val, train, test, other


def _model_predict(model, x, batch_size: Optional[int] = None, proba: bool = False, copy: bool = False) -> np.ndarray:
    # copied and adapted from autosklearn.automl._model_predict()

    if copy:
        x = x.copy()
    if isinstance(model, sklearn.ensemble.VotingRegressor):
        # `VotingRegressor` is not meant for multi-output regression and hence averages on wrong axis
        # `VotingRegressor.transform()` returns array of shape `(n_samples, n_estimators)` in case of single target,
        # and `(n_targets, n_samples, n_estimators)` in case of multiple targets
        prediction = np.average(model.transform(x), axis=-1, weights=model._weights_not_none)
        if prediction.ndim == 2:
            prediction = prediction.T
    else:
        predict_func = model.predict_proba if proba else model.predict
        if batch_size is not None and hasattr(model, 'batch_size'):
            prediction = predict_func(x, batch_size=batch_size)
        else:
            prediction = predict_func(x)
        if proba:
            np.clip(prediction, 0., 1., out=prediction)

    assert prediction.shape[0] == x.shape[0], \
        f'Prediction shape {model} is {prediction.shape} while X_ has shape {x.shape}'

    return prediction


def strip_autosklearn(obj):
    """
    Strip all autosklearn components from a given object. That means, if the given object contains an instance of an
    autosklearn class, which is a mere wrapper for an sklearn class, only the corresponding sklearn object is retained.
    :param obj: The object to process.
    :return: The processed object, which may be `obj` itself if `obj` does not contain any autosklearn components.
    Note that there is no guarantee that all autosklearn components occurring in `obj` are found, meaning some could
    still remain in the output.
    """
    if isinstance(obj, (list, tuple)):
        new = [strip_autosklearn(v) for v in obj]
        if all(o is n for o, n in zip(obj, new)):
            # avoid creating a new instance if possible
            return obj
        elif isinstance(obj, tuple):
            return tuple(new)
        else:
            return new
    elif isinstance(obj, set):
        aux = list(obj)
        new = [strip_autosklearn(v) for v in aux]
        if all(o is n for o, n in zip(aux, new)):
            # avoid creating a new instance if possible
            return obj
        else:
            return set(new)
    elif isinstance(obj, dict):
        new = {k: strip_autosklearn(v) for k, v in obj.items()}
        if all(old is new[k] for k, old in obj.items()):
            # avoid creating a new instance if possible
            return obj
        else:
            return new
    else:
        try:
            if obj.__class__.__module__.startswith('sklearn.'):
                dct = obj.__dict__
                # If `obj` has an attribute with and without trailing "_", only process the "_"-version and set the
                # result to the non-"_"-version as well. Reason: the non-"_"-version usually represents an unfitted
                # transformer/estimator, which, if it is an autosklearn object, may still lack the required
                # `estimator`/`transformer`/`preprocessor` attribute.
                special_attrs = {k for k in dct if not k.endswith('_') and k + '_' in dct}
                dct_new = {k: strip_autosklearn(v) for k, v in dct.items() if k not in special_attrs}
                if all(new is dct[k] for k, new in dct_new.items()):
                    # avoid creating a new instance if possible
                    return obj
                else:
                    new = obj.__class__.__new__(obj.__class__)
                    new.__dict__.update(dct_new)
                    new.__dict__.update({k: dct_new[k + '_'] for k in special_attrs})
                    return new
            elif obj.__class__.__module__.startswith('autosklearn.'):
                # special classes that MUST be returned unchanged, because although they have attributes like
                # `estimator`/`transformer`/`preprocessor` etc., their `transform()`/`predict()`/`predict_proba()`
                # method does not simply apply these objects :(
                if obj.__class__.__name__ in ('OrdinalEncoding', 'Nystroem', 'SelectPercentileClassification',
                                              'SelectClassificationRates'):
                    return obj

                # special classes whose `transform()` method is the identity function, but which are not detected by
                # the code below
                if obj.__class__.__name__ in ('NoPreprocessing',):
                    return 'passthrough'

                # try to find out whether `obj.transform()` is the identity function
                try:
                    lines = inspect.getsource(obj.transform).split('\n')
                    # strip whitespace at beginning and end
                    lines = [ln.strip() for ln in lines]
                    # drop empty lines and comments
                    lines = [ln for ln in lines if ln and not ln.startswith('#')]
                    if len(lines) == 2:
                        args = re.findall('def[ ]*transform[ ]*\(self,[ ]*([a-zA-Z_0-9]+)[:,)= ].+', lines[0])
                        if len(args) == 1 and args[0] and re.match('return[ ]+' + args[0], lines[1]) is not None:
                            return 'passthrough'
                except:     # noqa
                    pass

                for attr in ('choice', 'estimator', 'transformer', 'preprocessor', 'column_transformer'):
                    sub = getattr(obj, attr, None)
                    if sub == 'passthrough':
                        return 'passthrough'
                    elif sub is not None:
                        return strip_autosklearn(sub)

                steps = getattr(obj, 'steps', None)
                if isinstance(steps, list):
                    assert all(isinstance(s, tuple) for s in steps)
                    steps = [(s[0], strip_autosklearn(s[1])) for s in steps]
                    steps = [(n, t) for n, t in steps if t not in (None, 'passthrough')]
                    if steps:
                        if len(steps) == 1:
                            return steps[0][1]
                        else:
                            steps.append(('dummy', 'passthrough'))
                            return sklearn.pipeline.Pipeline(steps)
                    else:
                        return 'passthrough'
        except:     # noqa
            pass
        return obj


class AutoSklearnBackend(AutoMLBackend):

    @classmethod
    def name(cls) -> str:
        return 'auto-sklearn'

    def __init__(self, **kwargs):
        super(AutoSklearnBackend, self).__init__(**kwargs)
        self.converted_bool_columns_ = None     # tuple of columns of bool dtype that must be converted to float

    @property
    def model_ids_(self) -> list:
        if self.model_ is None:
            raise ValueError('AutoSklearnBackend must be fit to training data before model_ids_ can be accessed.')
        # actual identifiers are triples `(seed, ID, budget)`, from which we can safely restrict ourselves to ID
        return [_id for _, _id, _ in self._get_model_keys()]

    def summary(self) -> dict:
        try:
            # `show_models()` throws an exception if no models or only dummy models were trained
            models = self.model_.show_models().values()
            models = [self._summarize_model(m) for m in models]
        except:     # noqa
            name = self._get_estimator_name()
            # dummy estimators
            models = [{'model_id': _id, name: m.__class__.__name__}
                      for (_, _id, _), m in self.model_.automl_.models_.items()]
        return dict(
            automl=self.name(),
            task=self.task,
            models=models
        )

    def training_history(self) -> pd.DataFrame:
        # mixture of
        # * `autosklearn.automl.AutoML.performance_over_time_`,
        # * `autosklearn.automl.AutoML.cv_results_`,
        # * `autosklearn.estimators.AutoSklearnEstimator.leaderboard()`

        # A dict mapping model ids to their configurations
        configs = self.model_.automl_.runhistory_.ids_config

        metric_dict = {metric.name: [] for metric in self.model_.automl_._scoring_functions}
        timestamp = []
        model_id = []
        val_metric = []
        train_metric = []
        test_metric = []
        duration = []
        types = []
        main_metric = \
            (self.model_.automl_._metric.name, self.model_.automl_._metric._optimum, self.model_.automl_._metric._sign)
        other_metrics = [(metric.name, metric._optimum, metric._sign) for metric in self.model_.automl_._scoring_functions]
        for run_key, run_value in self.model_.automl_.runhistory_.data.items():
            if run_value.status == StatusType.SUCCESS and run_value.additional_info \
                    and 'num_run' in run_value.additional_info.keys():
                timestamp.append(
                    pd.Timestamp(run_value.endtime, unit='s', tz='utc').tz_convert(py_time.tzname[0]).tz_localize(None)
                )
                model_id.append(run_value.additional_info['num_run'])
                val, train, test, other = _get_metrics_from_run_value(run_value, main_metric, other_metrics)
                val_metric.append(val)
                train_metric.append(train)
                test_metric.append(test)
                for metric_name, values in metric_dict.items():
                    values.append(other.get(metric_name, np.NaN))
                duration.append(run_value.time)
                run_config = configs[run_key.config_id]._values
                types.append(run_config[f'{self._get_estimator_name()}:__choice__'])

        result = pd.DataFrame(data=dict(model_id=model_id, timestamp=timestamp, type=types))
        result['val_' + main_metric[0]] = val_metric
        for name, values in metric_dict.items():
            result['val_' + name] = values
        result['train_' + main_metric[0]] = train_metric
        if not np.isnan(test_metric).all():
            result['test_' + main_metric[0]] = test_metric
        result['duration'] = duration

        if self.model_.automl_.ensemble_ is not None:
            result['ensemble_weight'] = 0.
            for i, weight in enumerate(self.model_.automl_.ensemble_.weights_):
                (_, model_id, _) = self.model_.automl_.ensemble_.identifiers_[i]
                result.loc[result['model_id'] == model_id, 'ensemble_weight'] = weight
            aux = pd.DataFrame(self.model_.automl_.ensemble_performance_history)
            if aux.empty:
                aux = getattr(self.model_.automl_.ensemble_, 'trajectory_', [])
                if aux:
                    # set ensemble score of all models trained after last model occurring in ensemble to final
                    # trajectory value
                    aux = main_metric[1] - main_metric[2] * aux[-1]
                    result['ensemble_val_' + main_metric[0]] = np.nan
                    result.loc[result['timestamp'] >= result.loc[result['ensemble_weight'] > 0., 'timestamp'].max(),
                               'ensemble_val_' + main_metric[0]] = aux
            else:
                result['ensemble_val_' + main_metric[0]] = np.nan
                if 'ensemble_test_score' in aux.columns:
                    result['ensemble_test_' + main_metric[0]] = np.nan
                for i in range(len(result)):
                    # find first ensemble fitted after current and before next model, if any
                    mask = result['timestamp'].iloc[i] < aux['Timestamp']
                    if i + 1 < len(result):
                        mask &= result['timestamp'].iloc[i + 1] > aux['Timestamp']
                    if mask.any():
                        j = aux.loc[mask, 'Timestamp'].idxmin()
                        result.loc[result.index[i], 'ensemble_val_' + main_metric[0]] = aux.loc[j, 'ensemble_optimization_score']
                        if 'ensemble_test_score' in aux.columns:
                            result.loc[result.index[i], 'ensemble_test_' + main_metric[0]] = aux.loc[j, 'ensemble_test_score']
                result['ensemble_val_' + main_metric[0]].fillna(method='ffill', inplace=True)
                if 'ensemble_test_score' in aux.columns:
                    result['ensemble_test_' + main_metric[0]].fillna(method='ffill', inplace=True)

        result.sort_values('timestamp', inplace=True, ascending=True)
        return result

    def fitted_ensemble(self, ensemble_only: bool = True) -> FittedEnsemble:
        keys = self._get_model_keys(ensemble_only=ensemble_only)
        if ensemble_only:
            voting_keys = keys
        else:
            voting_keys = self._get_model_keys(ensemble_only=True)
        return FittedEnsemble(
            name=self.name(),
            task=self.task,
            models={k[1]: self._get_pipeline(k) for k in keys},
            meta_input=[_id for _, _id, _ in voting_keys],
            meta_estimator=
            [self.model_.automl_.ensemble_.weights_[self.model_.automl_.ensemble_.identifiers_.index(k)]
             for k in voting_keys]
        )

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame, groups: Optional[np.ndarray] = None,
            sample_weights: Optional[np.ndarray] = None, time: Optional[int] = None, jobs: Optional[int] = None,
            dataset_name: Optional[str] = None) -> 'AutoSklearnBackend':
        if time is None:
            time = self.config.get('time_limit')
        if jobs is None:
            jobs = self.config.get('jobs')
        tmp_folder = self.tmp_folder
        if tmp_folder is not None and tmp_folder.exists():
            shutil.rmtree(tmp_folder)
        kwargs = dict(
            # TODO: Tweak autosklearn to accept no time limit.
            time_left_for_this_task=600 if time is None or time < 0 else time * 60,  # `time` is given in minutes!
            n_jobs=-1 if jobs is None else jobs,
            tmp_folder=tmp_folder if tmp_folder is None else tmp_folder.as_posix(),
            delete_tmp_folder_after_terminate=tmp_folder is None,
            load_models=True
        )
        for config_param, askl_param in (('ensemble_size', None), ('ensemble_nbest', None),
                                         ('ensemble_nbest', 'max_models_on_disc'), ('memory_limit', None)):
            value = self.config.get(config_param)
            if value is not None:
                kwargs[askl_param or config_param] = value

        # include/exclude pipeline components
        specific = self.config.get(self.name(), {})
        include = specific.get('include')
        exclude = specific.get('exclude')
        if include is not None:
            include = include.copy()
        if exclude is not None:
            exclude = exclude.copy()
        self._validate_include_exclude_params(include, exclude, y_train.shape[1] > 1)
        if exclude is not None and len(exclude) == 0:
            exclude = None
        kwargs['include'] = include
        kwargs['exclude'] = exclude

        bool_cols = [c for c in x_train.columns if x_train[c].dtype.kind == 'b']
        if bool_cols and not any(x_train[c].dtype.name == 'category' for c in x_train.columns):
            # autosklearn tries to impute missing values in bool columns using sklearn's SimpleImputer. This does not
            # work if no categorical columns are present as well.
            # There is no need to convert dtypes when applying the model, as this happens automatically under the hood.
            self.converted_bool_columns_ = tuple(bool_cols)
            x_train = x_train.copy()
            x_train[bool_cols] = x_train[bool_cols].astype(np.float32)
        else:
            self.converted_bool_columns_ = ()

        # unlabeled samples
        # TODO: Tweak autosklearn to handle unlabeled samples.
        mask = y_train.notna().all(axis=1).values
        if not mask.all():
            logging.warn('auto-sklearn backend does not support unlabeled samples for training.'
                         f' Dropping {len(y_train) - mask.sum()} unlabeled samples ({100 * (1 - mask.mean()):.2f}%).')
            x_train = x_train[mask]
            y_train = y_train[mask]
            if groups is not None:
                groups = groups[mask]

        # TODO: Tweak autosklearn to handle sample weights.
        if sample_weights is not None:
            logging.warn('auto-sklearn backend does not support sample weights for training.')

        # resampling strategy
        resampling_strategy = specific.get('resampling_strategy')
        resampling_args = specific.get('resampling_strategy_arguments')
        if resampling_args is None:
            resampling_args = {}
        else:
            resampling_args = resampling_args.copy()
            if 'groups' in resampling_args:
                raise ValueError('"groups" must not occur in the resampling strategy arguments.')
        if resampling_strategy is not None:
            cls = getattr(grouped_split, resampling_strategy, None) or \
                  getattr(sklearn.model_selection, resampling_strategy, None)
            if cls is not None and issubclass(cls, (sklearn.model_selection.BaseCrossValidator,
                                                    sklearn.model_selection._split.BaseShuffleSplit,
                                                    sklearn.model_selection._split._RepeatedSplits)):
                if 'folds' in resampling_args:
                    resampling_args['n_splits'] = resampling_args['folds']
                    del resampling_args['folds']
                resampling_strategy = cls(**resampling_args)
                resampling_args = {}
        if groups is not None:
            resampling_args['groups'] = groups
        resampling_strategy = self._get_resampling_strategy(resampling_strategy, resampling_args)
        if resampling_strategy is not None:
            kwargs['resampling_strategy'] = resampling_strategy
        if not (groups is None and resampling_strategy is None):
            kwargs['resampling_strategy_arguments'] = resampling_args

        # metric and scoring functions
        metrics = self.config.get(self.task + '_metrics', [])
        if len(metrics) > 0:
            kwargs['metric'] = get_scorer(metrics[0])
            if len(metrics) > 1:
                kwargs['scoring_functions'] = [get_scorer(m) for m in metrics[1:]]
        metric = kwargs.get('metric')

        if self.task == 'regression':
            from autosklearn.regression import AutoSklearnRegressor as Estimator
        elif groups is None and resampling_strategy is None and include is None and exclude is None:
            # TODO: Use AutoSklearn2Classifier even if grouping is specified.
            del kwargs['include']
            del kwargs['exclude']
            from autosklearn.experimental.askl2 import AutoSklearn2Classifier as Estimator
        else:
            from autosklearn.classification import AutoSklearnClassifier as Estimator

        # logging 1: ensemble building
        if kwargs.get('n_jobs', 1) == 1:
            # only works if 1 job is used, because otherwise only <Future: ...> messages are logged
            import autosklearn.util.logging_ as logging_
            with open(io.make_path(logging_.__file__).parent / 'logging.yaml', mode='rt') as fh:
                logging_config = yaml.safe_load(fh)
            askl_handler = {'level': 'INFO', 'class': 'logging._AutoSklearnHandler', 'start_time': str(py_time.time())}
            if metric is not None:
                askl_handler.update(
                    metric=str(metric.name),
                    optimum=str(metric._optimum),
                    sign=str(metric._sign)
                )
            logging_config['handlers']['_askl_handler'] = askl_handler
            logging_config['loggers']['Client-EnsembleBuilder'] = {
                'level': 'INFO',
                'handlers': ['_askl_handler']
            }
            kwargs['logging_config'] = logging_config

        # logging 2: individual models
        if metric is not None:
            kwargs['get_trials_callback'] = _SMACLoggingCallback(
                (metric.name, metric._optimum, metric._sign),
                [(m.name, m._optimum, m._sign) for m in kwargs.get('scoring_functions', [])],
                self._get_estimator_name(),
                start_time=py_time.time()
            )
        # Unfortunately, ensembles are logged before the individual models (for some strange reason). There seems to
        # be nothing we can do about it ...

        if kwargs.get('n_jobs', 1) != 1:
            kwargs['seed'] = 42

        for k, v in specific.items():
            if k not in ('include', 'exclude', 'resampling_strategy', 'resampling_strategy_arguments'):
                if k in kwargs:
                    logging.warn(f'Ignoring auto-sklearn config argument "{k}",'
                                 ' because it has been set automatically already.')
                else:
                    kwargs[k] = v

        self.model_ = Estimator(**kwargs)
        self.model_.fit(x_train, y_train, dataset_name=dataset_name)
        if kwargs.get('ensemble_size', 1) > 0:
            if kwargs.get('n_jobs', 1) != 1:
                # `fit_ensemble()` must be called before `refit()`
                # not sure whether this is actually needed, but should not take too much time, so we do it anyway
                self.model_.fit_ensemble(y_train)
            self.model_.refit(x_train, y_train)
        return self

    def predict(self, x: pd.DataFrame, jobs: Optional[int] = None, batch_size: Optional[int] = None, model_id=None,
                calibrated: bool = 'auto') -> np.ndarray:
        if jobs is None:
            jobs = self.config.get('jobs')
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
        elif model_id is None:
            y = self.model_.predict(x, n_jobs=-1 if jobs is None else jobs, batch_size=batch_size)
        else:
            y = self._predict_single(x, model_id, batch_size, False)

        return y

    def predict_proba(self, x: pd.DataFrame, jobs: Optional[int] = None, batch_size: Optional[int] = None,
                      model_id=None, calibrated: bool = 'auto') -> np.ndarray:
        if jobs is None:
            jobs = self.config.get('jobs')
        if model_id is None:
            calibrated = calibrated is not False
            y = self.model_.predict_proba(x, n_jobs=-1 if jobs is None else jobs, batch_size=batch_size)
        else:
            calibrated = calibrated is True
            y = self._predict_single(x, model_id, batch_size, True)
        if calibrated:
            y = self.calibrate(y)
        return y

    def predict_all(self, x: pd.DataFrame, jobs: Optional[int] = None, batch_size: Optional[int] = None) \
            -> Dict[Any, np.ndarray]:
        if jobs is None:
            jobs = self.config.get('jobs')
        return self._predict_all(x, -1 if jobs is None else jobs, batch_size, False)

    def predict_proba_all(self, x: pd.DataFrame, jobs: Optional[int] = None, batch_size: Optional[int] = None) \
            -> Dict[Any, np.ndarray]:
        if jobs is None:
            jobs = self.config.get('jobs')
        return self._predict_all(x, -1 if jobs is None else jobs, batch_size, True)

    @classmethod
    def get_versions(cls) -> dict:
        from autosklearn import __version__ as askl_version
        from smac import __version__ as smac_version
        return {'auto-sklearn': askl_version, 'smac': smac_version, 'pandas': pd.__version__,
                'scikit-learn': sklearn.__version__}

    def _predict_single(self, x: pd.DataFrame, model_id, batch_size: Optional[int], proba: bool) -> np.ndarray:
        # get predictions of a single constituent model

        # perform checks, load models
        assert len(x) >= 1
        self._ensure_models()

        # copied from autosklearn.automl.AutoML.predict()
        # in fact, there is no need to call `feature_validator.transform()`, so we do not include it in FittedEnsemble
        x = self.model_.automl_.InputValidator.feature_validator.transform(x.copy())
        model = self._get_fitted_model_by_id(model_id)
        return _model_predict(model, x, batch_size=batch_size, proba=proba, copy=False)

    def _predict_all(self, x: pd.DataFrame, jobs: int, batch_size: Optional[int], proba: bool) -> Dict[Any, np.ndarray]:
        # get predictions of all constituent models and entire ensemble

        # perform checks, load models
        assert len(x) >= 1
        self._ensure_models()

        # copied from autosklearn.automl.AutoML.predict()
        # in fact, there is no need to call `feature_validator.transform()`, so we do not include it in FittedEnsemble
        x = self.model_.automl_.InputValidator.feature_validator.transform(x.copy())
        models = self._get_fitted_models()
        model_ids = self._get_model_keys(ensemble_only=True)

        all_predictions = joblib.Parallel(n_jobs=jobs)(
            joblib.delayed(_model_predict)(
                models[key],
                x,
                batch_size=batch_size,
                proba=proba,
                copy=True
            )
            for key in model_ids
        )
        out = {k: v for k, v in zip(model_ids, all_predictions)}

        if self.model_.automl_.ensemble_ is not None:
            predictions = self.model_.automl_.ensemble_.predict(all_predictions)
            if proba:
                np.clip(predictions, 0.0, 1.0, out=predictions)
            out['__ensemble__'] = predictions

        return out

    def _ensure_models(self):
        if self.model_.automl_.models_ is None or len(self.model_.automl_.models_) == 0 or \
                self.model_.automl_.ensemble_ is None:
            self.model_.automl_._load_models()      # noqa; hacky, because accesses private method

    def _get_estimator_name(self) -> str:
        return 'regressor' if self.task == 'regression' else 'classifier'

    def _summarize_model(self, model: dict) -> dict:
        # 3 possibilities:
        #   * "sklearn_classifier" or "sklearn_regressor" key with non-None value
        #       * "holdout", "holdout-iterative-fit", user-specified resampling strategy
        #   * "sklearn_classifier" or "sklearn_regressor" key whose value is None => rely on "classifier"/"regressor"
        #       * "cv-iterative-fit"
        #   * "voting_model" and "estimators" keys => rely on first item of "estimators"
        #       * "cv"
        estimator_key = self._get_estimator_name()
        out = {k: v for k, v in model.items() if isinstance(v, (str, int, float, bool, np.ndarray, np.number))}
        estimator = model.get('sklearn_' + estimator_key)
        dp = model.get('data_preprocessor')
        ba = model.get('balancing')
        fp = model.get('feature_preprocessor')
        voting_model = None
        if estimator is None:
            estimator = model.get(estimator_key)
            if estimator is None:
                voting_model = model.get('voting_model')        # instance of `VotingClassifier` or `VotingRegressor`
                estimators = model.get('estimators') or []
                if len(estimators) >= 1:
                    estimator = estimators[0]
                    dp = estimator.get('data_preprocessor')
                    ba = estimator.get('balancing')
                    fp = estimator.get('feature_preprocessor')
                    estimator = estimator.get('sklearn_' + estimator_key)   # instance of sklearn estimator
        for key, obj in [('voting_model', voting_model), ('data_preprocessor', dp),
                         ('balancing', ba), ('feature_preprocessor', fp), (estimator_key, estimator)]:
            if obj is not None:
                obj = getattr(obj, 'choice', None) or obj
                try:
                    out[key] = ' '.join(repr(obj).replace('\n', ' ').split())
                except:     # noqa
                    out[key] = obj.__class__.__name__ + '(<unknown params>)'
        return out

    def _get_pipeline(self, key) -> dict:
        pipeline = self.model_.automl_.models_.get(key)
        if pipeline is None:
            # cv => rely on `self.model_.automl_.cv_models_.get(key)`
            return dict(estimator=strip_autosklearn(self.model_.automl_.cv_models_.get(key)))
        elif hasattr(pipeline, 'steps'):
            estimator = pipeline.steps[-1][1].choice.estimator
            if estimator is None:
                # cv => rely on `self.model_.automl_.cv_models_.get(key)`
                return dict(estimator=strip_autosklearn(self.model_.automl_.cv_models_.get(key)))
            else:
                steps = [strip_autosklearn(obj[1]) for obj in pipeline.steps[:-1]]
                steps = [s for s in steps if s != 'passthrough']
                return dict(
                    preprocessing=steps,
                    estimator=estimator
                )
        else:
            # dummy estimator
            return dict(estimator=pipeline)

    def _get_fitted_model_by_id(self, model_id):
        models = [v for (_, _id, _), v in self._get_fitted_models().items() if _id == model_id]
        if len(models) == 1:
            return models[0]
        elif len(models) == 0:
            raise ValueError(f'No fitted model with ID {model_id} found.')
        else:
            raise ValueError(f'{len(models)} fitted models with ID {model_id} found.')

    def _get_model_keys(self, ensemble_only: bool = True) -> list:
        if self.model_.automl_.ensemble_ is None or not ensemble_only:
            return list(self.model_.automl_.models_.keys())
        else:
            return self.model_.automl_.ensemble_.get_selected_model_identifiers()

    def _get_fitted_models(self) -> dict:
        # copied from autosklearn.automl.AutoML.predict()
        if len(self.model_.automl_.models_) == 0:       # `.models_` could be a list, which makes the code below crash
            raise ValueError('No fitted models found.')
        try:
            for tmp_model in self.model_.automl_.models_.values():
                if tmp_model.__class__.__name__ in ('DummyRegressor', 'DummyClassifier'):
                    sklearn.utils.validation.check_is_fitted(tmp_model)
                else:
                    sklearn.utils.validation.check_is_fitted(tmp_model.steps[-1][-1])
            return self.model_.automl_.models_
        except sklearn.exceptions.NotFittedError:
            try:
                sklearn.utils.validation.check_is_fitted(list(self.model_.automl_.cv_models_.values())[0])
                return self.model_.automl_.cv_models_
            except sklearn.exceptions.NotFittedError:
                raise ValueError('No fitted models found.')

    def _get_resampling_strategy(self, resampling_strategy, resampling_strategy_args: Optional[dict] = None):
        # Copied from autosklearn.evaluation.train_evaluator.TrainEvaluator.get_splitter()

        # TODO: Find out what "-iterative-fit" and "partial-" are, and how they differ from standard
        #   "holdout" and "cv".
        # TODO: Replace all "cv"-like strategies by user-defined resampling strategies, even if no groups are specified.
        #   This would simplify the whole setup and make it more uniform, as discussed below.

        # If "holdout" or "holdout-iterative-fit" are used:
        #   * Each constituent of the final ensemble is a single pipeline, consisting of a data preprocessor,
        #       a balancer (in classification tasks), a feature preprocessor, and a final estimator. The pipeline is
        #       fitted on the internal training partition.
        #   * Calling `refit()` is not necessary, but fits the pipeline again on the whole training set. This can
        #       improve predictive performance.
        #   * The constituents of the ensemble can be accessed via the `.automl_.models_` attribute, which is a dict of
        #       pipeline objects. Each step contains both the hyperparameter configuration as well as an actual
        #       trained transformer/estimator object.
        # If "cv" or "cv-iterative" are used:
        #   * Each constituent of the final ensemble is a sklearn `VotingClassifier` or `VotingRegressor` instance with
        #       as many base estimators as there are folds. Each of these base estimators is a full autosklearn
        #       pipeline that is fitted on the respective training portion of internal cross validation.
        #       The hyperparameters of all these pipelines are identical, but the trained estimators need not be.
        #   * Calling `refit()` is not necessary, but fits the pipelines again on the whole training set.
        #       Hypothesis: After calling `refit()`, all pipelines are identical and thus redundant.
        #   * The `Voting[Classifier|Regressor]` instances can be accessed via the `.automl_.cv_models_` attribute, and
        #       the pipelines via their `.estimators_` attribute (mind the "_"!). In contrast, the `.automl_.models_`
        #       attribute only contains the hyperparameter configuration of each constituent, but no actual
        #       transformer/estimator instances.
        # If a user-specified resampling strategy is used:
        #   * Like in "holdout" and "holdout-iterative-fit", each constituent of the final ensemble is a single
        #       pipeline. Before calling `refit()` these pipelines only contain the hyperparameter configuration,
        #       afterward they also contain actual transformer/estimator instances.
        #   * `refit()` must be called to fit the pipelines on the whole training data.

        if resampling_strategy_args is None or resampling_strategy_args.get('groups') is None:
            # rely on autosklearn's default setup
            return resampling_strategy
        elif resampling_strategy is None:
            resampling_strategy = 'holdout'
        elif not isinstance(resampling_strategy, str):
            if not isinstance(resampling_strategy, (sklearn.model_selection.GroupShuffleSplit,
                                                    sklearn.model_selection.GroupKFold,
                                                    sklearn.model_selection.LeaveOneGroupOut,
                                                    sklearn.model_selection.LeavePGroupsOut,
                                                    grouped_split.StratifiedGroupShuffleSplit,
                                                    grouped_split.StratifiedGroupKFold)):
                logging.warn('Grouping information might not be taken into account by specified'
                             f' resampling strategy {resampling_strategy}.')
            return resampling_strategy

        if not resampling_strategy_args.get('shuffle', True):
            raise ValueError('In grouped splitting, resampling strategy argument "shuffle" must be set to true.')

        train_size = resampling_strategy_args.get('train_size')
        if train_size is None:
            # don't set to 0.67 if 0
            train_size = 0.67
        test_size = 1. - train_size

        if self.task in ('binary_classification', 'multiclass_classification'):
            if resampling_strategy in ('holdout', 'holdout-iterative-fit'):
                cv = grouped_split.StratifiedGroupShuffleSplit(n_splits=1, test_size=test_size, random_state=1)
            elif resampling_strategy in ('cv', 'cv-iterative-fit', 'partial-cv', 'partial-cv-iterative-fit'):
                cv = grouped_split.StratifiedGroupKFold(
                    n_splits=resampling_strategy_args['folds'],
                    shuffle=True,
                    random_state=1,
                )
            else:
                raise ValueError(resampling_strategy)
        else:
            if resampling_strategy in ('holdout', 'holdout-iterative-fit'):
                cv = sklearn.model_selection.GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=1)
            elif resampling_strategy in ('cv', 'partial-cv', 'partial-cv-iterative-fit'):
                cv = sklearn.model_selection.GroupKFold(n_splits=resampling_strategy_args['folds'])
            else:
                raise ValueError(resampling_strategy)
        return cv

    def _validate_include_exclude_params(self, include: Optional[dict], exclude: Optional[dict], multioutput: bool):
        # Copied from autosklearn.pipeline.base.BasePipeline._validate_include_exclude_params()
        # `include` and `exclude` are modified in place

        if include is None and exclude is None:
            return
        elif not (include is None or exclude is None):
            for k in list(include):
                ve = exclude.get(k)
                if ve is not None:
                    include[k] = [v for v in include[k] if v not in ve]
                    del exclude[k]

        if self.task == 'regression':
            from autosklearn.pipeline.regression import SimpleRegressionPipeline as Pipeline
            dataset_properties = dict(multioutput=multioutput, sparse=False)
        else:
            from autosklearn.pipeline.classification import SimpleClassificationPipeline as Pipeline
            dataset_properties = dict(
                multilabel=self.task == 'multilabel_classification',
                multiclass=self.task == 'multiclass_classification'
            )
        from autosklearn.pipeline.components.base import AutoSklearnChoice

        pip = Pipeline(dataset_properties=dataset_properties)
        supported_steps = {step[0]: list(step[1].get_available_components(dataset_properties=dataset_properties))
                           for step in pip.steps if isinstance(step[1], AutoSklearnChoice)}
        for argument in (include, exclude):
            if argument is not None:
                for key, candidate_components in list(argument.items()):
                    available_components = supported_steps.get(key)
                    if available_components is None:
                        del argument[key]
                    else:
                        if not isinstance(candidate_components, (list, tuple, set)):
                            candidate_components = [candidate_components]
                        argument[key] = [c for c in candidate_components if c in available_components]

        if include is not None:
            # check that no component list is empty
            for k, v in include.items():
                if len(v) == 0:
                    raise ValueError(f'No valid components for {k} found. Choose among {supported_steps[k]}.')
        if exclude is not None:
            # drop empty lists
            for k, v in list(exclude.items()):
                if len(v) == 0:
                    del exclude[k]
