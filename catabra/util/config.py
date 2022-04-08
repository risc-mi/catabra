from typing import Optional
import copy


DEFAULT_CONFIG = {
    "automl": "auto-sklearn",       # AutoML backend; currently, only "auto-sklearn" is supported
    "ensemble_size": 10,            # maximum size of final ensemble
    "ensemble_nbest": 10,           # maximum number of best single models to use in final ensemble
    "memory_limit": 3072,           # memory limit for single models, in MB
    "time_limit": 10,               # default time limit for overall model training, in minutes; negative means no time limit; overwritten by command-line parameter
    "jobs": 1,                      # default number of jobs to use; negative means all available processors; overwritten by command-line parameter

    # binary classification performance metrics; see https://scikit-learn.org/stable/modules/model_evaluation.html
    # first element is main metric used for choosing best model
    "binary_classification_metrics": [
        "roc_auc",
        "accuracy",
        "balanced_accuracy"
    ],

    # multiclass classification performance metrics; see https://scikit-learn.org/stable/modules/model_evaluation.html
    # first element is main metric used for choosing best model
    "multiclass_classification_metrics": [
        "accuracy",
        "balanced_accuracy"
    ],

    # multilabel classification performance metrics; see https://scikit-learn.org/stable/modules/model_evaluation.html
    # first element is main metric used for choosing best model
    "multilabel_classification_metrics": [
        "f1_macro"
    ],

    # regression performance metrics; see https://scikit-learn.org/stable/modules/model_evaluation.html
    # first element is main metric used for choosing best model
    "regression_metrics": [
        "r2",
        "mean_absolute_error",
        "mean_squared_error"
    ],

    # auto-sklearn specific config; see https://automl.github.io/auto-sklearn/master/api.html
    "auto-sklearn": {
        "include": None,
        "exclude": None,
        "resampling_strategy": None,    # can also be the name of a subclass of `BaseCrossValidator`, `_RepeatedSplits` or `BaseShuffleSplit` in `sklearn.model_selection`
        "resampling_strategy_arguments": None
    }
}


def add_defaults(config: dict, default: Optional[dict] = None) -> dict:
    """
    Add default config values into a given config dict.
    :param config: The base config. Modified in place.
    :param default: The config with default values. None means `DEFAULT_CONFIG`.
    :return: The updated config dict.
    """
    if default is None:
        default = DEFAULT_CONFIG
    for k, v in default.items():
        if k not in config:
            config[k] = copy.deepcopy(v)
        elif k in ('auto-sklearn',):
            v_ = config[k]
            if isinstance(v_, dict) and isinstance(v, dict):
                for k0, v0 in v.items():
                    if k0 not in v_:
                        v_[k0] = copy.deepcopy(v0)
    return config
