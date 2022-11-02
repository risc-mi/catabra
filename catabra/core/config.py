from pathlib import Path
from typing import Optional, Union, Dict, Tuple
import copy

import pandas as pd
from pandas import DataFrame

from catabra.core import io


class Invocation:

    @property
    def start(self) -> pd.Timestamp:
        return self._start

    @property
    def table(self) -> Tuple[Union[str, Path, DataFrame], ...]:
        return self._table

    @property
    def split(self) -> str:
        return self._split

    @property
    def sample_weight(self) -> Optional[str]:
        return self._sample_weight

    @property
    def out(self) -> Union[str, Path]:
        return self._out

    @out.setter
    def out(self, value: str):
        self._out = value

    @property
    def jobs(self) -> int:
        return self._jobs

    def __init__(
        self,
        *table: Union[str, Path, pd.DataFrame],
        split: Optional[str] = None,
        sample_weight: Optional[str] = None,
        out: Union[str, Path, None] = None,
        jobs: Optional[int] = None
    ):
        self._start = pd.Timestamp.now()
        self._table = table
        self._split = split
        self._sample_weight = sample_weight
        self._out = out
        self._jobs = jobs

    def update(self, src: Dict):
        if src:
            if len(self._table) == 0:
                self._table = src.get('table') or []
                if '<DataFrame>' in self._table:
                    raise ValueError('Invocations must not contain "<DataFrame>" tables.')

            if self._split is None:
                self._split = src.get('split')
            if self._sample_weight is None:
                self._sample_weight = src.get('sample_weight')

            if self._out is None:
                self._out = src.get('out')
            if self._jobs is None:
                self._jobs = src.get('jobs')

        if self._split == '':
            self._split = None
        if self._sample_weight == '':
            self._sample_weight = None

        self._table = [io.make_path(tbl, absolute=True) if isinstance(tbl, (str, Path)) else tbl for tbl in self._table]

    def to_dict(self) -> dict:
        return dict(
            split=self._split,
            sample_weight=self._sample_weight,
            out=self._out,
            jobs=self._jobs,
            timestamp=self._start
        )


DEFAULT_CONFIG = {
    "automl": "auto-sklearn",       # AutoML backend; currently, only "auto-sklearn" is supported
    "ensemble_size": 10,            # maximum size of final ensemble
    "ensemble_nbest": 10,           # maximum number of best single models to use in final ensemble
    "memory_limit": 3072,           # memory limit for single models, in MB
    "time_limit": 1,                # default time limit for overall model training, in minutes; negative means no time limit; overwritten by command-line parameter
    "jobs": 1,                      # default number of jobs to use; negative means all available processors; overwritten by command-line parameter
    "copy_analysis_data": False,    # whether to copy data to be analyzed into output folder; can be True, False or maximum size to copy, in MB
    "copy_evaluation_data": False,  # whether to copy test data into output folder; same possible values as for "copy_analysis_data"
    "static_plots": True,           # whether to create static plots in PDF format using Matplotlib
    "interactive_plots": False,     # whether to create interactive plots in HTML format using plotly; if True, plotly must be installed separately
    "bootstrapping_repetitions": 0,  # number of bootstrapping repetitions when evaluating models; 0 means bootstrapping is disabled
    "explainer": "shap",            # name of the model explanation framework to use

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

    "ood": {
        # name of module/class for OOD detector
        # if 'source' is 'internal': name of one of the modules in catabra.ood.internal (e.g. soft_brownian_offset)
        # if 'source' is 'pyod': name of one of the modules in pyod.models (e.g. kde)
        # if 'source' is 'external': full import path consisting of modules and class (e.g. custom.module.CustomOOD)
        # if value is <None> no OOD detection is performed
        "class": "autoencoder",
        # Import from CaTabRa OODDetector subclasses ('internal'), PyOD ('pyod') or custom source ('external')
        "source": "internal",
        # Keyword arguments for different OOD detectors in the form 'name': value.
        # If none are specified default values are used.
        "kwargs": {}
    },

    # auto-sklearn specific config; see https://automl.github.io/auto-sklearn/master/api.html
    "auto-sklearn": {
        "include": None,
        "exclude": None,
        "resampling_strategy": None,    # can also be the name of a subclass of `BaseCrossValidator`, `_RepeatedSplits` or `BaseShuffleSplit` in `sklearn.model_selection`
        "resampling_strategy_arguments": None
    }
}


# only absolutely necessary (data) preprocessing, only standard ML algorithms
BASIC_CONFIG = {
    "ensemble_size": 5,
    "ensemble_nbest": 5,

    "auto-sklearn": {
        "include": {
            # data preprocessor cannot be configured
            "feature_preprocessor": [
                "no_preprocessing"
            ],
            "classifier": [
                "decision_tree",
                "k_nearest_neighbors",
                "lda",
                "liblinear_svc",
                "qda",
                "random_forest"
            ],
            "regressor": [
                "decision_tree",
                "k_nearest_neighbors",
                "liblinear_svr",
                "random_forest"
            ]
        }
    }
}


# only preprocessing steps that do not change feature space too much, only interpretable ML algorithms
INTERPRETABLE_CONFIG = {
    "ensemble_size": 5,
    "ensemble_nbest": 5,

    "auto-sklearn": {
        "include": {
            "feature_preprocessor": [
                "densifier",
                "extra_trees_preproc_for_classification",
                "extra_trees_preproc_for_regression",
                "fast_ica",
                "feature_agglomeration",
                # "kernel_pca",
                "kitchen_sinks",
                "liblinear_svc_preprocessor",
                "no_preprocessing",
                # "nystroem",
                "pca",
                # "polynomial",
                # "random_trees_embedding",
                "select_percentile_classification",
                "select_percentile_regression",
                "select_rates_classification",
                "select_rates_regression",
                "truncatedSVD"
            ],
            "classifier": [
                "adaboost",
                # "bernoulli_nb",
                "decision_tree",
                "extra_trees",
                # "gaussian_nb",
                "gradient_boosting",
                # "k_nearest_neighbors",
                "lda",
                "liblinear_svc",
                # "libsvm_svc",
                # "mlp",
                # "multinomial_nb",
                "passive_aggressive",
                # "qda",
                "random_forest",
                "sgd"
            ],
            "regressor": [
                "adaboost",
                "ard_regression",
                "decision_tree",
                "extra_trees",
                # "gaussian_process",
                "gradient_boosting",
                # "k_nearest_neighbors",
                "liblinear_svr",
                # "libsvm_svr",
                # "mlp",
                "random_forest",
                "sgd"
            ]
        }
    }
}

DEFAULT_CONFIGS = {
    None: DEFAULT_CONFIG,
    'basic': BASIC_CONFIG,
    'interpretable': INTERPRETABLE_CONFIG,
    'full': {} # TODO: check
}


def add_defaults(config: dict, default: Optional[str] = None) -> dict:
    """
    Add default config values into a given config dict.
    :param config: The base config. Modified in place.
    :param default: The config with default values. None means `DEFAULT_CONFIG`.
    :return: The updated config dict.
    """
    default = DEFAULT_CONFIGS.get(default, DEFAULT_CONFIG)
    print(default)
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
