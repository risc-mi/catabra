from typing import Optional, Dict, Any
import shutil
import numpy as np
import pandas as pd
import sklearn
import joblib

from ...util import logging
from ...analysis import grouped_split
from ..base import AutoMLBackend
from .scorer import get_scorer


def _model_predict(model, x, batch_size: Optional[int] = None, proba: bool = False, copy: bool = False,
                   multioutput_regression: bool = False) -> np.ndarray:
    # copied from autosklearn.automl._model_predict()

    if copy:
        x = x.copy()
    if multioutput_regression and isinstance(model, sklearn.ensemble.VotingRegressor):
        prediction = np.average(model.transform(x), axis=2).T
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


class AutoSklearnBackend(AutoMLBackend):

    @classmethod
    def name(cls) -> str:
        return 'auto-sklearn'

    def __init__(self, **kwargs):
        super(AutoSklearnBackend, self).__init__(**kwargs)
        self._multioutput = False

    @property
    def model_ids_(self) -> list:
        if self.model_ is None:
            raise ValueError('AutoSklearnBackend must be fit to training data before model_ids_ can be accessed.')
        # actual identifiers are triples `(seed, ID, budget)`, from which we can safely restrict ourselves to ID
        return [_id for _, _id, _ in self._get_model_keys()]

    def summary(self) -> dict:
        return dict(
            automl=self.name(),
            task=self.task,
            models=[self._summarize_model(m) for m in self.model_.show_models().values()]
        )

    def training_history(self) -> pd.DataFrame:
        # mixture of
        # * `autosklearn.automl.AutoML.performance_over_time_`,
        # * `autosklearn.automl.AutoML.cv_results_`,
        # * `autosklearn.estimators.AutoSklearnEstimator.leaderboard()`
        from smac.tae import StatusType
        import time     # no global import, shadowed by `time` parameter of method `fit()`

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
        metric_optimum = self.model_.automl_._metric._optimum
        metric_sign = self.model_.automl_._metric._sign
        metric_name = self.model_.automl_._metric.name
        for run_key, run_value in self.model_.automl_.runhistory_.data.items():
            if run_value.status == StatusType.SUCCESS and run_value.additional_info \
                    and 'num_run' in run_value.additional_info.keys():
                timestamp.append(pd.Timestamp(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run_value.endtime))))
                model_id.append(run_value.additional_info['num_run'])
                val_metric.append(metric_optimum - (metric_sign * run_value.cost))
                train_metric.append(metric_optimum - (metric_sign * run_value.additional_info['train_loss']))
                test_metric.append(metric_optimum - (metric_sign * run_value.additional_info.get('test_loss', np.nan)))
                duration.append(run_value.time)
                run_config = configs[run_key.config_id]._values
                types.append(run_config[f'{self._get_estimator_name()}:__choice__'])

                # additional metrics are only available for validation set for single models,
                # even if test data is provided
                for metric in self.model_.automl_._scoring_functions:
                    if metric.name in run_value.additional_info.keys():
                        metric_cost = run_value.additional_info[metric.name]
                        metric_value = metric._optimum - (metric._sign * metric_cost)
                    else:
                        metric_value = np.NaN
                    metric_dict[metric.name].append(metric_value)

        result = pd.DataFrame(data=dict(model_id=model_id, timestamp=timestamp, type=types))
        result['val_' + metric_name] = val_metric
        for name, values in metric_dict.items():
            result['val_' + name] = values
        result['train_' + metric_name] = train_metric
        if not np.isnan(test_metric).all():
            result['test_' + metric_name] = test_metric
        result['duration'] = duration

        if self.model_.automl_.ensemble_ is not None:
            result['ensemble_weight'] = 0.
            for i, weight in enumerate(self.model_.automl_.ensemble_.weights_):
                (_, model_id, _) = self.model_.automl_.ensemble_.identifiers_[i]
                result.loc[result['model_id'] == model_id, 'ensemble_weight'] = weight
            aux = pd.DataFrame(self.model_.automl_.ensemble_performance_history)
            if not aux.empty:
                result['ensemble_val_' + metric_name] = np.nan
                if 'ensemble_test_score' in aux.columns:
                    result['ensemble_test_' + metric_name] = np.nan
                for i in result.index:
                    mask = result.loc[i, 'timestamp'] < aux['Timestamp']
                    if mask.any():
                        j = aux.loc[mask, 'Timestamp'].idxmin()
                        result.loc[i, 'ensemble_val_' + metric_name] = aux.loc[j, 'ensemble_optimization_score']
                        if 'ensemble_test_score' in aux.columns:
                            result.loc[i, 'ensemble_test_' + metric_name] = aux.loc[j, 'ensemble_test_score']
                result['ensemble_val_' + metric_name].fillna(method='ffill', inplace=True)
                if 'ensemble_test_score' in aux.columns:
                    result['ensemble_test_' + metric_name].fillna(method='ffill', inplace=True)

        result.sort_values('timestamp', inplace=True, ascending=True)
        return result

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame, groups: Optional[np.ndarray] = None,
            time: Optional[int] = None, jobs: Optional[int] = None,
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
        if include is None:
            include = {}
        else:
            include = include.copy()
        if exclude is None:
            exclude = {}
        else:
            exclude = exclude.copy()
        if 'regressor' in include:
            del include['regressor']
        if 'regressor' in exclude:
            del exclude['regressor']
        for k in list(include):
            ve = exclude.get(k, [])
            include[k] = [v for v in include[k] if v not in ve]
            del exclude[k]
        kwargs['include'] = include
        kwargs['exclude'] = exclude

        # unlabeled samples
        # TODO: Tweak autosklearn to handle unlabeled samples.
        mask = y_train.notna().all(axis=1).values
        if not mask.all():
            x_train = x_train[mask]
            y_train = y_train[mask]
            if groups is not None:
                groups = groups[mask]

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

        self._multioutput = y_train.shape[1] > 1
        if self.task == 'regression':
            from autosklearn.regression import AutoSklearnRegressor as Estimator
        elif groups is None and resampling_strategy is None and len(include) == len(exclude) == 0:
            # TODO: Use AutoSklearn2Classifier even if grouping is specified.
            del kwargs['include']
            del kwargs['exclude']
            from autosklearn.experimental.askl2 import AutoSklearn2Classifier as Estimator
        else:
            from autosklearn.classification import AutoSklearnClassifier as Estimator

        # TODO: Set up proper logging, e.g., for printing to console whenever new pipeline has been fit.
        self.model_ = Estimator(**kwargs)
        self.model_.fit(x_train, y_train, dataset_name=dataset_name)
        if kwargs.get('ensemble_size', 1) > 0:
            self.model_.refit(x_train, y_train)
        return self

    def predict(self, x: pd.DataFrame, jobs: Optional[int] = None, batch_size: Optional[int] = None, model_id=None) \
            -> np.ndarray:
        if jobs is None:
            jobs = self.config.get('jobs')
        if model_id is None:
            return self.model_.predict(x, n_jobs=-1 if jobs is None else jobs, batch_size=batch_size)
        else:
            return self._predict_single(x, model_id, batch_size, False)

    def predict_proba(self, x: pd.DataFrame, jobs: Optional[int] = None, batch_size: Optional[int] = None,
                      model_id=None) -> np.ndarray:
        if jobs is None:
            jobs = self.config.get('jobs')
        if model_id is None:
            return self.model_.predict_proba(x, n_jobs=-1 if jobs is None else jobs, batch_size=batch_size)
        else:
            return self._predict_single(x, model_id, batch_size, True)

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

    def _predict_single(self, x: pd.DataFrame, model_id, batch_size: Optional[int], proba: bool) -> np.ndarray:
        # get predictions of a single constituent model

        # perform checks, load models
        assert len(x) >= 1
        self._ensure_models()

        # copied from autosklearn.automl.AutoML.predict()
        x = self.model_.automl_.InputValidator.feature_validator.transform(x.copy())
        model = self._get_fitted_model_by_id(model_id)
        return _model_predict(model, x, batch_size=batch_size, proba=proba, copy=False,
                              multioutput_regression=(self.task == 'regression' and self._multioutput))

    def _predict_all(self, x: pd.DataFrame, jobs: int, batch_size: Optional[int], proba: bool) -> Dict[Any, np.ndarray]:
        # get predictions of all constituent models and entire ensemble

        # perform checks, load models
        assert len(x) >= 1
        self._ensure_models()

        # copied from autosklearn.automl.AutoML.predict()
        x = self.model_.automl_.InputValidator.feature_validator.transform(x.copy())
        models = self._get_fitted_models()
        model_ids = self._get_model_keys(ensemble_only=True)

        all_predictions = joblib.Parallel(n_jobs=jobs)(
            joblib.delayed(_model_predict)(
                models[key],
                x,
                batch_size=batch_size,
                proba=proba,
                copy=True,
                multioutput_regression=(self.task == 'regression' and self._multioutput)
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
