from typing import Optional
import copy


DEFAULT_CONFIG = {
    "automl": "auto-sklearn",       # AutoML backend; currently, only "auto-sklearn" is supported
    "ensemble_size": 10,            # maximum size of final ensemble
    "ensemble_nbest": 10,           # maximum number of best single models to use in final ensemble
    "memory_limit": 3072,           # memory limit for single models, in MB
    "time_limit": 10,               # default time limit for overall model training, in minutes; negative means no time limit; overwritten by command-line parameter
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
        "class": "autoencoder",
        # Import from CaTabRa OODDetector subclasses ('internal') or PyOD ('pyod')
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
