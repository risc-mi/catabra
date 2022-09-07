from typing import Optional
import warnings
import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, UniformFloatHyperparameter
from ConfigSpace.conditions import EqualsCondition

from autosklearn.pipeline.components.classification import add_classifier
from autosklearn.pipeline.components.regression import add_regressor
from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm, AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import DENSE, SIGNED_DATA, UNSIGNED_DATA, PREDICTIONS

with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=FutureWarning)
    import xgboost


class _XGBoostBase:

    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.3, gamma: float = 0,
                 min_child_weight: float = 1, max_delta_step: float = 0, subsample: float = 1,
                 colsample_bytree: float = 1, colsample_bylevel: float = 1, colsample_bynode: float = 1,
                 reg_alpha: float = 0, reg_lambda: float = 0, scale_pos_weight: Optional[float] = None,
                 random_state=None, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.estimator = None
        self._classify = True
        # dummy attributes, needed for method `.set_hyperparameters()`
        self._gamma_choice = False
        self._alpha_choice = False
        self._lambda_choice = False

    def fit(self, X, y, sample_weight=None):
        kwargs = dict(
            n_estimators=int(self.n_estimators),
            max_depth=int(self.max_depth),
            learning_rate=float(self.learning_rate),
            gamma=float(self.gamma),
            min_child_weight=float(self.min_child_weight),
            max_delta_step=float(self.max_delta_step),
            subsample=float(self.subsample),
            colsample_bytree=float(self.colsample_bytree),
            colsample_bylevel=float(self.colsample_bylevel),
            colsample_bynode=float(self.colsample_bynode),
            reg_alpha=float(self.reg_alpha),
            reg_lambda=float(self.reg_lambda),
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state
        )
        if self._classify:
            self.estimator = xgboost.XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                objective='binary:logistic' if len(np.unique(y)) <= 2 else 'multi:softmax',
                **kwargs
            )
        else:
            self.estimator = xgboost.XGBRegressor(
                eval_metric='mae',
                objective='reg:squarederror',
                **kwargs
            )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=FutureWarning)
            self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            'shortname': 'XGB',
            'handles_multilabel': False,
            'handles_multioutput': False,
            'is_deterministic': False,
            'input': (DENSE, SIGNED_DATA, UNSIGNED_DATA),
            'output': (PREDICTIONS,)
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        # search space according to https://arxiv.org/abs/2106.03253
        cs = ConfigurationSpace()
        n_estimators = UniformIntegerHyperparameter('n_estimators', lower=100, upper=4000, default_value=100)
        learning_rate = UniformFloatHyperparameter(
            name='learning_rate', lower=np.exp(-7), upper=1, default_value=0.3, log=True
        )
        max_depth = UniformIntegerHyperparameter(
            name='max_depth', lower=1, upper=10, default_value=6
        )
        subsample = UniformFloatHyperparameter(
            name='subsample', lower=0.2, upper=1, default_value=1
        )
        colsample_bytree = UniformFloatHyperparameter(
            name='colsample_bytree', lower=0.2, upper=1, default_value=1
        )
        colsample_bylevel = UniformFloatHyperparameter(
            name='colsample_bylevel', lower=0.2, upper=1, default_value=1
        )
        min_child_weight = UniformFloatHyperparameter(
            name='min_child_weight', lower=np.exp(-16), upper=np.exp(5), default_value=1, log=True
        )
        gamma_choice = CategoricalHyperparameter(
            '_gamma_choice', [False, True]
        )
        gamma = UniformFloatHyperparameter(
            'gamma', lower=np.exp(-16), upper=np.exp(2), default_value=1, log=True
        )
        alpha_choice = CategoricalHyperparameter(
            '_alpha_choice', [False, True]
        )
        reg_alpha = UniformFloatHyperparameter(
            'reg_alpha', lower=np.exp(-16), upper=np.exp(2), default_value=1, log=True
        )
        lambda_choice = CategoricalHyperparameter(
            '_lambda_choice', [False, True]
        )
        reg_lambda = UniformFloatHyperparameter(
            'reg_lambda', lower=np.exp(-16), upper=np.exp(2), default_value=1, log=True
        )
        cs.add_hyperparameters([
            n_estimators, learning_rate, max_depth, subsample, colsample_bytree, colsample_bylevel, min_child_weight,
            gamma_choice, gamma, alpha_choice, reg_alpha, lambda_choice, reg_lambda
        ])
        cs.add_conditions([
            EqualsCondition(gamma, gamma_choice, True),
            EqualsCondition(reg_alpha, alpha_choice, True),
            EqualsCondition(reg_lambda, lambda_choice, True)
        ])
        return cs


class XGBoostClassifier(_XGBoostBase, AutoSklearnClassificationAlgorithm):

    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.3, gamma: float = 0,
                 min_child_weight: float = 1, max_delta_step: float = 0, subsample: float = 1,
                 colsample_bytree: float = 1, colsample_bylevel: float = 1, colsample_bynode: float = 1,
                 reg_alpha: float = 0, reg_lambda: float = 0, scale_pos_weight: Optional[float] = None,
                 random_state=None, **kwargs):
        _XGBoostBase.__init__(
            self,
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, gamma=gamma,
            min_child_weight=min_child_weight, max_delta_step=max_delta_step, subsample=subsample,
            colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel, colsample_bynode=colsample_bynode,
            reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight, random_state=random_state,
            **kwargs
        )
        self._classify = True

    @staticmethod
    def get_properties(dataset_properties=None):
        properties = _XGBoostBase.get_properties(dataset_properties=dataset_properties)
        properties.update(
            name='XGBoost Classifier',
            handles_regression=False,
            handles_classification=True,
            handles_multiclass=True
        )
        return properties


class XGBoostRegressor(_XGBoostBase, AutoSklearnRegressionAlgorithm):

    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.3, gamma: float = 0,
                 min_child_weight: float = 1, max_delta_step: float = 0, subsample: float = 1,
                 colsample_bytree: float = 1, colsample_bylevel: float = 1, colsample_bynode: float = 1,
                 reg_alpha: float = 0, reg_lambda: float = 0, scale_pos_weight: Optional[float] = None,
                 random_state=None, **kwargs):
        _XGBoostBase.__init__(
            self,
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, gamma=gamma,
            min_child_weight=min_child_weight, max_delta_step=max_delta_step, subsample=subsample,
            colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel, colsample_bynode=colsample_bynode,
            reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight, random_state=random_state,
            **kwargs
        )
        self._classify = False

    @staticmethod
    def get_properties(dataset_properties=None):
        properties = _XGBoostBase.get_properties(dataset_properties=dataset_properties)
        properties.update(
            name='XGBoost Regressor',
            handles_regression=True,
            handles_classification=False,
            handles_multiclass=False
        )
        return properties


# add components to auto-sklearn
add_classifier(XGBoostClassifier)
add_regressor(XGBoostRegressor)
