#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import importlib
import inspect
import logging as py_logging
import re
import shutil
import time as py_time  # otherwise shadowed by parameter of method `fit()`
from distutils.version import LooseVersion
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from autosklearn import __version__ as askl_version
from smac.callbacks import IncorporateRunResultCallback
from smac.runhistory.runhistory import RunHistory
from smac.tae import StatusType

from catabra.automl.askl import explanation
from catabra.automl.askl.scorer import get_scorer
from catabra.automl.base import AutoMLBackend
from catabra.automl.fitted_ensemble import FittedEnsemble, _model_predict
from catabra.util import io, logging, metrics, split
from catabra.util.common import repr_list, repr_timedelta
from catabra.util.preprocessing import FeatureFilter

explanation.TransformationExplainer.register_factory('auto-sklearn', explanation.askl_explainer_factory,
                                                     errors='ignore')

_addons_success = []    # list of pairs `(name, versions)`
_addons_failure = []    # list of pairs `(name, exception)`
# load add-ons
for _d in (Path(__file__).parent / 'addons').iterdir():
    if _d.suffix.lower() == '.py':
        try:
            _pkg = importlib.import_module('catabra.automl.askl.addons.' + _d.stem, package=__package__)
            _addons_success.append((_d.stem, getattr(_pkg, 'get_versions', lambda: {})()))
        except ImportError as _e:
            # required packages are not available => skip
            _addons_failure.append((_d.stem, _e))


def get_addons() -> Tuple[tuple, tuple]:
    """
    Get all successfully and unsuccessfully loaded add-on modules.

    Returns
    -------
    Tuple[tuple, tuple]
        Pair `(success, failure)`, where `success` is a tuple of pairs `(name, version_dict)` and `failure` is a tuple
        of pairs `(name, exception)`.
    """
    return tuple(_addons_success), tuple(_addons_failure)


# we have to use a global variable, because class _EnsembleLoggingHandler can only be passed string arguments
_monitors = {}


class _EnsembleLoggingHandler(py_logging.Handler):
    """
    Logging handler for printing _comprehensible_ messages whenever a new ensemble has been fit.
    Solution is a bit hacky and can easily break if some internals of auto-sklearn change.
    """

    def __init__(self, metric: str = 'cost', optimum: str = '0.', sign: str = '-1.', start_time: str = '0.',
                 monitor_name=None, **kwargs):
        super(_EnsembleLoggingHandler, self).__init__(**kwargs)
        self.metric = metric
        self.optimum = float(optimum)
        self.sign = float(sign)
        self.start_time = float(start_time)
        self.monitor_name = monitor_name
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
                    monitor = _monitors.get(self.monitor_name)
                    if monitor is not None:
                        monitor.update(
                            event='ensemble',
                            timestamp=record.created,
                            elapsed_time=record.created - self.start_time,
                            text='n_constituent_models: {:d}'.format(n_models),
                            **{'ensemble_val_' + self.metric: value}
                        )
                except RecursionError:
                    raise
                except:     # noqa
                    self.handleError(record)


# Hacky, but class must be in some installed package such that `logging` finds it. We do not want to rely on CaTabRa
# being installed.
setattr(py_logging, '_AutoSklearnHandler', _EnsembleLoggingHandler)


class _SMACLoggingCallback(IncorporateRunResultCallback):
    """
    Callback for logging model training and printing messages whenever a new model has been trained.
    """

    def __init__(self, main_metric: Optional[Tuple[str, float, float]],
                 other_metrics: Iterable[Tuple[str, float, float]], estimator_name: str, start_time: float = 0.,
                 monitor_name=None):
        self.main_metric = main_metric
        self.other_metrics = other_metrics
        self.estimator_choice = estimator_name + ':__choice__'
        self.start_time = start_time
        self.monitor_name = monitor_name
        self._n = 0     # interestingly, counting models seems to work even if multiple jobs are used
        self.runhistory = RunHistory()      # own run history, used if training is interrupted

    def __call__(
            self, smbo: 'SMBO', run_info: 'RunInfo', result: 'RunValue', time_left: float # noqa F821
    ) -> Optional[bool]:
        try:
            if self.main_metric is not None and result.status == StatusType.SUCCESS and result.additional_info:
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

                monitor = _monitors.get(self.monitor_name)
                if monitor is not None:
                    monitor.update(
                        event='model',
                        timestamp=result.endtime,
                        elapsed_time=result.endtime - self.start_time,
                        text='type: {:s}'.format(run_info.config._values[self.estimator_choice]),
                        **{'val_' + self.main_metric[0]: val},
                        **{'val_' + k: v for k, v in other.items()}
                    )

            self.runhistory.add(
                run_info.config,
                result.cost,
                result.time,
                result.status,
                instance_id=run_info.instance,
                seed=run_info.seed,
                budget=run_info.budget,
                starttime=result.starttime,
                endtime=result.endtime,
                additional_info=result.additional_info
            )
        except:  # noqa
            pass
        return None


def _get_metrics_from_run_value(run_value: 'RunValue', main_metric: Tuple[str, float, float], # noqa F821
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


def strip_autosklearn(obj):
    """
    Strip all autosklearn components from a given object. That means, if the given object contains an instance of an
    autosklearn class, which is a mere wrapper for an sklearn class, only the corresponding sklearn object is retained.

    Parameters
    ----------
    obj:
        The object to process.

    Returns
    -------
    The processed object, which may be `obj` itself if `obj` does not contain any autosklearn components.
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
                                              'SelectClassificationRates', 'TfidfEncoder'):
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
                    assert all(isinstance(s, (tuple, list)) for s in steps)
                    steps = [(s[0], strip_autosklearn(s[1])) for s in steps]
                    steps = [(n, t) for n, t in steps if t not in (None, 'passthrough')]
                    if steps:
                        if len(steps) == 1:
                            return steps[0][1]
                        else:
                            if not hasattr(steps[-1][1], 'predict'):
                                # add "passthrough" iff last step is no estimator
                                steps.append(('dummy', 'passthrough'))
                            return sklearn.pipeline.Pipeline(steps)
                    else:
                        return 'passthrough'
        except:     # noqa
            pass
        return obj


class AutoSklearnBackend(AutoMLBackend):

    def __init__(self, **kwargs):
        super(AutoSklearnBackend, self).__init__(**kwargs)
        self.converted_bool_columns_ = None     # tuple of columns of bool dtype that must be converted to float
        self._feature_filter = None

    @property
    def name(self) -> str:
        return 'auto-sklearn'

    @property
    def model_ids_(self) -> list:
        if self.model_ is None:
            raise ValueError('AutoSklearnBackend must be fit to training data before model_ids_ can be accessed.')
        # actual identifiers are triples `(seed, ID, budget)`, from which we can safely restrict ourselves to ID
        return [_id for _, _id, _ in self._get_model_keys()]

    @property
    def feature_filter_(self) -> Optional[FeatureFilter]:
        return getattr(self, '_feature_filter', None)

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
            automl=self.name,
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

        scoring_functions = self.model_.scoring_functions or []
        metric_dict = {metric.name: [] for metric in scoring_functions}
        timestamp = []
        model_id = []
        val_metric = []
        train_metric = []
        test_metric = []
        duration = []
        types = []
        main_metric = (self.model_.metric.name, self.model_.metric._optimum, self.model_.metric._sign)
        other_metrics = [(metric.name, metric._optimum, metric._sign) for metric in scoring_functions]
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

        result = pd.DataFrame(data=dict(model_id=model_id))
        result['timestamp'] = pd.to_datetime(timestamp)
        if hasattr(self, 'fit_start_time_'):
            result['total_elapsed_time'] = \
                result['timestamp'] - \
                pd.Timestamp(self.fit_start_time_, unit='s', tz='utc').tz_convert(py_time.tzname[0]).tz_localize(None)
        result['type'] = types
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
                        result.loc[result.index[i], 'ensemble_val_' + main_metric[0]] = \
                            aux.loc[j, 'ensemble_optimization_score']
                        if 'ensemble_test_score' in aux.columns:
                            result.loc[result.index[i], 'ensemble_test_' + main_metric[0]] = \
                                aux.loc[j, 'ensemble_test_score']
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
            name=self.name,
            task=self.task,
            models={k[1]: self._get_pipeline(k) for k in keys},
            meta_input=[_id for _, _id, _ in voting_keys],
            meta_estimator=
            [self.model_.automl_.ensemble_.weights_[self.model_.automl_.ensemble_.identifiers_.index(k)]
             for k in voting_keys],
            calibrator=self.calibrator_
        )

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame, groups: Optional[np.ndarray] = None,
            sample_weights: Optional[np.ndarray] = None, time: Optional[int] = None, jobs: Optional[int] = None,
            dataset_name: Optional[str] = None, monitor=None) -> 'AutoSklearnBackend':
        if time is None:
            time = self.config.get('time_limit')
        if jobs is None:
            jobs = self.config.get('jobs')
        tmp_folder = self.tmp_folder
        if tmp_folder is not None and tmp_folder.exists():
            shutil.rmtree(tmp_folder)
        kwargs = dict(
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

        if _addons_success:
            logging.log(
                'Successfully loaded the following auto-sklearn add-on module(s): ' +
                ', '.join([n for n, _ in _addons_success])
            )
        else:
            logging.log('No auto-sklearn add-on modules loaded.')

        # include/exclude pipeline components
        prefix = self.name + '_'
        specific = {k[len(prefix):]: v for k, v in self.config.items() if k.startswith(prefix)}
        include = specific.get('include')
        exclude = specific.get('exclude')
        if include is not None:
            include = include.copy()
        if exclude is not None:
            exclude = exclude.copy()
        _validate_include_exclude_params(include, exclude, self.task, y_train.shape[1] > 1)
        if exclude is not None and len(exclude) == 0:
            exclude = None
        kwargs['include'] = include
        kwargs['exclude'] = exclude

        self._feature_filter = None
        if LooseVersion(askl_version) < LooseVersion('0.15.0'):
            # string features are only supported in version >= 0.15.0
            string_cols = [c for c in x_train.columns if x_train[c].dtype.name == 'string']
            if string_cols:
                logging.warn(
                    f'Columns {repr_list(string_cols, brackets=False)} have string data type, but auto-sklearn does'
                    f' not support text features in version {askl_version}.'
                    ' Install auto-sklearn version >= 0.15.0 for text support.'
                )
                x_train = x_train.drop(string_cols, axis=1)
                self._feature_filter = FeatureFilter().fit(x_train)

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
        #   https://github.com/automl/auto-sklearn/issues/288
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
        if resampling_strategy == 'CustomPredefinedSplit':
            resampling_strategy = split.CustomPredefinedSplit.from_data(x_train, resampling_args['columns'])
            x_train = x_train.drop(resampling_args['columns'], axis=1)
            resampling_args = {}
            self._feature_filter = FeatureFilter().fit(x_train)
        elif resampling_strategy is not None:
            cls = getattr(split, resampling_strategy, None) or \
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
        resampling_strategy = self._get_resampling_strategy(resampling_strategy, resampling_args, y_train)
        if resampling_strategy is not None:
            kwargs['resampling_strategy'] = resampling_strategy
        if not (groups is None and resampling_strategy is None):
            kwargs['resampling_strategy_arguments'] = resampling_args

        # metric and scoring functions
        task_metrics = self.config.get(self.task + '_metrics', [])
        if len(task_metrics) > 0:
            kwargs['metric'] = get_scorer(task_metrics[0])
            if len(task_metrics) > 1:
                kwargs['scoring_functions'] = [get_scorer(m) for m in task_metrics[1:]]
        metric = kwargs.get('metric')

        # TODO: If grouping is specified, use auto-sklearn 2.0 and choose resampling- and budget allocation strategy
        #   based on number of groups rather than samples. Also show a warning with instructions how to avoid using
        #   auto-sklearn 2.0.
        if self.task == 'regression':
            from autosklearn.regression import AutoSklearnRegressor as Estimator
            logging.log('Using auto-sklearn 1.0 (regression not supported by 2.0).')
        elif groups is None and resampling_strategy is None and include is None and exclude is None:
            try:
                from autosklearn.experimental.askl2 import (
                    AutoSklearn2Classifier as Estimator,
                )
                del kwargs['include']
                del kwargs['exclude']
                logging.log('Using auto-sklearn 2.0.')
            except ModuleNotFoundError:
                from autosklearn.classification import (
                    AutoSklearnClassifier as Estimator,
                )
                logging.log('Using auto-sklearn 1.0 (2.0 not found).')
        else:
            from autosklearn.classification import AutoSklearnClassifier as Estimator
            logging.log('Using auto-sklearn 1.0. Avoid sample grouping and set each of resampling_strategy, include'
                        ' and exclude to None to use 2.0.')

        self.fit_start_time_ = py_time.time()
        if monitor is None:
            monitor_name = ''
        else:
            monitor.set_params(**self.config)
            monitor_name = str(np.random.uniform(10 ** 10))
            _monitors[monitor_name] = monitor

        # logging 1: ensemble building
        if kwargs.get('n_jobs', 1) == 1:
            # only works if 1 job is used, because otherwise only <Future: ...> messages are logged
            import autosklearn.util.logging_ as logging_
            with open(io.make_path(logging_.__file__).parent / 'logging.yaml', mode='rt') as fh:
                logging_config = yaml.safe_load(fh)
            askl_handler = {'level': 'INFO', 'class': 'logging._AutoSklearnHandler',
                            'start_time': str(self.fit_start_time_), 'monitor_name': monitor_name}
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
        smac_callback = _SMACLoggingCallback(
            metric if metric is None else (metric.name, metric._optimum, metric._sign),
            [(m.name, m._optimum, m._sign) for m in kwargs.get('scoring_functions', [])],
            self._get_estimator_name(),
            start_time=self.fit_start_time_,
            monitor_name=monitor_name
        )
        # Unfortunately, ensembles are logged before the individual models (for some strange reason). There seems to
        # be nothing we can do about it ...

        for k, v in specific.items():
            if k not in ('include', 'exclude', 'resampling_strategy', 'resampling_strategy_arguments'):
                if k in kwargs:
                    logging.warn(f'Ignoring auto-sklearn config argument "{k}",'
                                 ' because it has been set automatically already.')
                else:
                    kwargs[k] = v

        self.model_ = Estimator(**kwargs)
        if getattr(self.model_, 'get_trials_callback', False) is None:
            # cannot be passed to constructor of AutoSklearn2Classifier
            self.model_.get_trials_callback = smac_callback
        interrupted = False
        try:
            self.model_.fit(x_train, y_train, dataset_name=dataset_name)
        except KeyboardInterrupt:
            logging.warn('Hyperparameter optimization interrupted after'
                         f' {repr_timedelta(py_time.time() - self.fit_start_time_, subsecond_resolution=2)}.'
                         ' Trying to build final ensemble with models trained so far,'
                         ' but consistency of internal state cannot be ensured. Use result with care.')
            self.model_.automl_._budget_type = None
            self.model_.automl_.runhistory_ = smac_callback.runhistory
            self.model_.automl_.trajectory_ = []        # not needed
            interrupted = True

        if kwargs.get('ensemble_size', 1) > 0:
            if interrupted:
                try:
                    # `fit_ensemble()` normally does not need to be called manually (and may even raise an exception).
                    # However, if the fitting process was interrupted it might be necessary to call it.
                    self.model_.fit_ensemble(y_train)
                except:     # noqa
                    pass
            # `refit()` should always be called, and _must_ be called if a custom resampling strategy is used. See
            # comment in `_get_resampling_strategy()` for details.
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
                y = metrics.multiclass_proba_to_pred(y)
            else:
                y = (y >= 0.5).astype(np.int32)
        elif model_id is None:
            y = self.model_.predict(self._prepare_for_predict(x), n_jobs=-1 if jobs is None else jobs,
                                    batch_size=batch_size)
        else:
            y = self._predict_single(self._prepare_for_predict(x), model_id, batch_size, False)

        return y

    def predict_proba(self, x: pd.DataFrame, jobs: Optional[int] = None, batch_size: Optional[int] = None,
                      model_id=None, calibrated: bool = 'auto') -> np.ndarray:
        if jobs is None:
            jobs = self.config.get('jobs')
        x = self._prepare_for_predict(x)
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
        return self._predict_all(self._prepare_for_predict(x), -1 if jobs is None else jobs, batch_size, False)

    def predict_proba_all(self, x: pd.DataFrame, jobs: Optional[int] = None, batch_size: Optional[int] = None) \
            -> Dict[Any, np.ndarray]:
        if jobs is None:
            jobs = self.config.get('jobs')
        return self._predict_all(self._prepare_for_predict(x), -1 if jobs is None else jobs, batch_size, True)

    def get_versions(self) -> dict:
        from smac import __version__ as smac_version
        out = {'auto-sklearn': askl_version, 'smac': smac_version, 'pandas': pd.__version__,
               'scikit-learn': sklearn.__version__}
        for _, v in _addons_success:
            out.update(v)
        return out

    def _prepare_for_predict(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.feature_filter_ is not None:
            x = self.feature_filter_.transform(x)
        return x

    def _predict_single(self, x: pd.DataFrame, model_id, batch_size: Optional[int], proba: bool) -> np.ndarray:
        # get predictions of a single constituent model

        # perform checks, load models
        assert len(x) >= 1
        self._ensure_models()

        # copied from autosklearn.automl.AutoML.predict()
        # in fact, there is no need to call `feature_validator.transform()`, so we do not include it in FittedEnsemble
        x = self.model_.automl_.InputValidator.feature_validator.transform(x.copy())
        model = self._get_fitted_model_by_id(model_id)
        return _model_predict(dict(estimator=model), x, batch_size=batch_size, proba=proba, copy=False)

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
                dict(estimator=models[key]),
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
        steps = None
        pipeline = self.model_.automl_.models_.get(key)
        if pipeline is None:
            # cv => rely on `self.model_.automl_.cv_models_.get(key)`
            # This is mostly a legacy feature, since built-in CV resampling strategies are not used anymore.
            estimator = strip_autosklearn(self.model_.automl_.cv_models_.get(key))
        elif hasattr(pipeline, 'steps'):
            estimator = pipeline.steps[-1][1].choice.estimator
            if estimator is None:
                # cv => rely on `self.model_.automl_.cv_models_.get(key)`
                # This is mostly a legacy feature, since built-in CV resampling strategies are not used anymore.
                estimator = strip_autosklearn(self.model_.automl_.cv_models_.get(key))
            else:
                steps = [strip_autosklearn(obj[1]) for obj in pipeline.steps[:-1]]
                steps = [s for s in steps if s != 'passthrough']
        else:
            # dummy estimator
            estimator = pipeline

        # This way, the feature filter is added to each model separately, which means it is applied several times to
        # the exact same input when applying the whole ensemble. This is obviously not optimal.
        if self.feature_filter_ is not None:
            if steps is None:
                steps = [self.feature_filter_]
            else:
                steps = [self.feature_filter_] + steps

        return dict(estimator=estimator, preprocessing=steps)

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
            # This is mostly a legacy feature, since built-in CV resampling strategies are not used anymore.
            try:
                sklearn.utils.validation.check_is_fitted(list(self.model_.automl_.cv_models_.values())[0])
                return self.model_.automl_.cv_models_
            except sklearn.exceptions.NotFittedError:
                raise ValueError('No fitted models found.')

    def _get_resampling_strategy(self, resampling_strategy, resampling_strategy_args: Optional[dict] = None, y=None):
        # Copied from autosklearn.evaluation.train_evaluator.TrainEvaluator.get_splitter()

        # Meaning of "-iterative-fit" and "partial-" (see autosklearn.evaluation.train_evaluator.eval_holdout() etc.):
        #   * "-iterative-fit" seems to indicate that internally, instead of completely fitting models for each split
        #       one after another, models are fit some iterations for each split, then for a couple of more iterations,
        #       and so on, until all are fully converged. Advantage: If there is a timeout, partially fitted models
        #       exist for every split, meaning that the current hyperparameter configuration can be evaluated. If the
        #       selected model class does not support iterative fitting, the evaluator tacitly falls back to the
        #       corresponding non-iterative strategy.
        #    * "partial-": ???
        # In general, the iterative- and partial variants only seem to affect the operational behavior of the
        # resampling strategies (especially wrt. timeouts), not their semantics.

        # If "holdout" or "holdout-iterative-fit" are used:
        #   * Each constituent of the final ensemble is a single pipeline, consisting of a data preprocessor,
        #       a balancer (in classification tasks), a feature preprocessor, and a final estimator. The pipeline is
        #       fitted on the internal training partition.
        #   * Calling `refit()` is not necessary, but fits the pipeline again on the whole training set. This changes
        #       the contents of `.automl_.models_` in place and can potentially improve predictive performance.
        #   * The constituents of the ensemble can be accessed via the `.automl_.models_` attribute, which is a dict of
        #       pipeline objects. Each step contains both the hyperparameter configuration and an actual trained
        #       transformer/estimator object.
        # If "cv" or "cv-iterative" are used:
        #   * Each constituent of the final ensemble is a sklearn `VotingClassifier` or `VotingRegressor` instance with
        #       as many base estimators as there are folds. Each of these base estimators is a full autosklearn
        #       pipeline that is fitted on the respective training portion of internal cross validation.
        #       The hyperparameters of all these pipelines are identical, but the trained estimators need not be.
        #   * The `Voting[Classifier|Regressor]` instances can be accessed via the `.automl_.cv_models_` attribute, and
        #       the pipelines via their `.estimators_` attribute (mind the "_"!).
        #   * Calling `refit()` is not necessary, but fits the pipelines again on the whole training set.
        #       Before calling `refit()`, `.automl_.models_` only contains hyperparameter configurations.
        #       After calling `refit()` it contains actual transformer/estimator instances. The cross-validation models
        #       in `.automl_.cv_models_` are unaffected by `refit()`, and can thus be still accessed afterward.
        # If a user-specified resampling strategy is used:
        #   * Like in "holdout" and "holdout-iterative-fit", each constituent of the final ensemble is a single
        #       pipeline. Before calling `refit()` these pipelines only contain the hyperparameter configuration,
        #       afterward they also contain actual transformer/estimator instances.
        #   * `automl_.cv_models_` is never set, even if the used strategy is "CV-like" (e.g., grouped CV). This, for
        #       instance, considerably affects the size of the trained model on disk, which might be confusing for
        #       users.
        #   * `refit()` must be called to fit the pipelines on the whole training data.
        # In general, `automl_.predict()` always first tries `automl_.models_` and then, if not possible because no
        #   transformer/estimator instances are found, resorts to `automl_.cv_models_`.

        if resampling_strategy_args is None:
            resampling_strategy_args = {}

        no_groups = resampling_strategy_args.get('groups') is None

        if resampling_strategy in ('cv', 'cv-iterative-fit', 'partial-cv', 'partial-cv-iterative-fit'):
            # use custom strategy for the sake of uniformity, regardless of whether grouping is specified
            # `.automl_.cv_models_` will remain None, reducing the required disk space considerably
            pass
        elif no_groups:
            # rely on autosklearn's default setup
            return resampling_strategy
        elif not isinstance(resampling_strategy, str) and resampling_strategy is not None:
            if not isinstance(resampling_strategy, (sklearn.model_selection.GroupShuffleSplit,
                                                    sklearn.model_selection.GroupKFold,
                                                    sklearn.model_selection.LeaveOneGroupOut,
                                                    sklearn.model_selection.LeavePGroupsOut,
                                                    split.StratifiedGroupShuffleSplit,
                                                    split.StratifiedGroupKFold)):
                logging.warn('Grouping information might not be taken into account by specified'
                             f' resampling strategy {resampling_strategy}.')
            return resampling_strategy

        if resampling_strategy is None:
            resampling_strategy = 'holdout'

        shuffle = resampling_strategy_args.get('shuffle', True)
        if not (no_groups or shuffle):
            raise ValueError('In grouped splitting, resampling strategy argument "shuffle" must be set to true.')

        train_size = resampling_strategy_args.get('train_size')
        if train_size is None:
            # don't set to 0.67 if 0
            train_size = 0.67
        test_size = 1. - train_size

        if no_groups:
            # "CV-like" strategy
            if self.task in ('binary_classification', 'multiclass_classification'):
                if shuffle:
                    try:
                        import warnings
                        from copy import deepcopy
                        with warnings.catch_warnings():
                            warnings.simplefilter('error')
                            cv = sklearn.model_selection.StratifiedKFold(
                                n_splits=resampling_strategy_args['folds'],
                                shuffle=shuffle,
                                random_state=1,
                            )
                            test_cv = deepcopy(cv)
                            next(test_cv.split(y, y))
                    except UserWarning as e:
                        if 'The least populated class in y has only' in e.args[0]:
                            from autosklearn.evaluation.splitter import (
                                CustomStratifiedKFold,
                            )
                            cv = CustomStratifiedKFold(
                                n_splits=resampling_strategy_args['folds'],
                                shuffle=shuffle,
                                random_state=1,
                            )
                        else:
                            raise e
                else:
                    cv = sklearn.model_selection.KFold(n_splits=resampling_strategy_args['folds'], shuffle=False)
            elif resampling_strategy in ('cv', 'partial-cv', 'partial-cv-iterative-fit'):
                cv = sklearn.model_selection.KFold(
                    n_splits=resampling_strategy_args['folds'],
                    shuffle=shuffle,
                    random_state=1 if shuffle else None,
                )
            else:
                raise ValueError(resampling_strategy)
        elif self.task in ('binary_classification', 'multiclass_classification'):
            if resampling_strategy in ('holdout', 'holdout-iterative-fit'):
                cv = split.StratifiedGroupShuffleSplit(n_splits=1, test_size=test_size, random_state=1)
            elif resampling_strategy in ('cv', 'cv-iterative-fit', 'partial-cv', 'partial-cv-iterative-fit'):
                cv = split.StratifiedGroupKFold(
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


def _validate_include_exclude_params(include: Optional[dict], exclude: Optional[dict], task: str, multioutput: bool):
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

    if task == 'regression':
        from autosklearn.pipeline.regression import SimpleRegressionPipeline as Pipeline
        dataset_properties = dict(multioutput=multioutput, sparse=False)
    else:
        from autosklearn.pipeline.classification import (
            SimpleClassificationPipeline as Pipeline,
        )
        dataset_properties = dict(
            multilabel=task == 'multilabel_classification',
            multiclass=task == 'multiclass_classification'
        )
    from autosklearn.pipeline.components.base import AutoSklearnChoice

    pip = Pipeline(dataset_properties=dataset_properties)
    supported_steps = {
        step[0]: (list(step[1].get_components()),
                  list(step[1].get_available_components(dataset_properties=dataset_properties)))
        for step in pip.steps if isinstance(step[1], AutoSklearnChoice)
    }
    unavailable_components = []
    missing_components = []
    for argument, is_include in ((include, True), (exclude, False)):
        if argument is not None:
            for key, candidate_components in list(argument.items()):
                all_components, available_components = supported_steps.get(key, (None, None))
                if available_components is None:
                    del argument[key]
                else:
                    if not isinstance(candidate_components, (list, tuple, set)):
                        candidate_components = [candidate_components]
                    if is_include:
                        unavailable_components += [f'{key}/{c}' for c in candidate_components
                                                   if c in all_components and c not in available_components]
                        missing_components += [f'{key}/{c}' for c in candidate_components if c not in all_components]
                    argument[key] = [c for c in candidate_components if c in available_components]

    if unavailable_components:
        logging.warn(
            'The following component(s) are not available for the current prediction task: ' +
            ', '.join(unavailable_components)
        )
    if missing_components:
        msg = 'The following component(s) could not be found: ' + ', '.join(missing_components)
        if _addons_failure:
            msg += '\nMaybe they are contained in add-on modules that could not be loaded:\n    ' +\
                   '\n    '.join([f'{n}: {e.msg}' for n, e in _addons_failure])
        logging.warn(msg)

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
