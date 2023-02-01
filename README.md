# CaTabRa: Classification, Regression and Analysis and Tabular Data

## Description

**CaTabRa** is a Python package for automatically analyzing tabular data. This includes generating descriptive
statistics, creating out-of-distribution detectors, training prediction models for classification and regression
tasks, and evaluating/applying these models on unseen data.

CaTabRa is both a command-line tool and a library, which means it can be easily integrated into other projects.

## Installation

Install the relevant packages, as listed in `env.yml` and `requirements.txt`, in a new or existing environment. The
key requirements are Python 3.9, [pandas](https://pandas.pydata.org/) and
[auto-sklearn](https://automl.github.io/auto-sklearn/master/).

**IMPORTANT**: auto-sklearn currently only runs under Linux, which means that CaTabRa only runs under Linux, too. If
on Windows, you can use a virtual machine, like [WSL 2](https://docs.microsoft.com/en-us/windows/wsl/about).

## Configuration

CaTabRa can be configured via *config files* that can be passed to some commands, most notably `analyze` (see above).
Config files should be JSON files with a single top-level object. The possible parameters are described below.
See `util/config.py` for further information, including the default values of the parameters.

### General Configuration Parameters
* `"automl"`: AutoML backend to use. Currently, the only available option is `"auto-sklearn"`, but new backends can be
    added by subclassing `catabra.automl.base.AutoMLBackend`.
* `"ensemble_size"`: Maximum size of the final ensemble of prediction models. Passed to the AutoML backend.
* `"ensemble_nbest"`: Maximum number of best single models to use in the final ensemble. Passed to the AutoML
    backend.
* `"memory_limit"`: Memory limit for single prediction models, in MB. Passed to the AutoML backend.
* `"time_limit"`: Time limit for overall model training, in minutes; negative means no time limit. Overwritten by
    command-line argument `--time`.
* `"jobs"`: Number of jobs to use; negative means all available processors. Overwritten by command-line argument
    `--jobs`.
* `"copy_analysis_data"`: Whether to copy the data to be analyzed into the output folder. Can be `true`, `false` or the
    maximum size to copy, in MB.
* `"copy_evaluation_data"`: Whether to copy test data into the output folder. Can be `true`, `false` or the
    maximum size to copy, in MB.
* `"static_plots"`: Whether to create static plots of the training history, evaluation results, etc., using the
    [Matplotlib](https://matplotlib.org/) backend. These plots are saved as PDF vector graphics.
* `"interactive_plots"`: Whether to create interactive plots of the training history, evaluation results, etc., using
    the optional [plotly](https://plotly.com/python/) backend. These plots are saved as HTML files. **Note**: plotly
    is not installed by default, but must be installed separately!
* `"bootstrapping_repetitions"`: Number of repetitions to perform for calculating
    [bootstrapped performance metrics](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)).
    If `0`, bootstrapping is disabled; otherwise, the main performance metrics specified by config params
    `"binary_classification_metrics"` etc. are evaluated as many times on randomly drawn (with replacement) resamples
    of the final prediction results, to obtain summary statistics like mean, standard deviation, etc.
* `"explainer"`: Explanation backend to use. Currently, the only available option is `"shap"`, but new backends can be
    added by subclassing `catabra.explanation.base.EnsembleExplainer`.
* `"binary_classification_metrics"`: List of metrics to evaluate when training binary classification models. The first
    metric in the list is the "main" metric optimized by the AutoML backend, the other metrics merely provide insights
    into the training process. Note that when evaluating trained models on new data using command `evaluate`, *all*
    suitable metrics are computed automatically. See `doc/metrics.md` for a list of supported metrics.
* `"multiclass_classification_metrics"`: List of metrics to evaluate when training multiclass classification models.
    Has the same meaning as `"binary_classification_metrics"`.
* `"multilabel_classification_metrics"`: List of metrics to evaluate when training multilabel classification models.
    Has the same meaning as `"binary_classification_metrics"`.
* `"regression_metrics"`: List of metrics to evaluate when training regression models.
    Has the same meaning as `"binary_classification_metrics"`.

### auto-sklearn Specific Parameters

The following parameters are specific to the [auto-sklearn](https://automl.github.io/auto-sklearn/master/) backend.
See [this page](https://automl.github.io/auto-sklearn/master/api.html) for details.
* `"include"`: Components that are included in hyperparameter optimization, step-wise. Set to `null` to include all
    available components.
* `"exclude"`: Components that are excluded from hyperparameter optimization, step-wise. Set to `null` to exclude no
    components.
* `"resampling_strategy"`: The resampling strategy to use for internal validation. Note that if a string is passed,
    like `"holdout"` or `"cv"`, the 'grouped' version of the corresponding strategy is used if grouping is specified
    (e.g. via the `--group` argument of command `analyze`).
* `"resampling_strategy_arguments"`: Additional arguments for the resampling strategy, like the number of folds in
    *k*-fold cross validation.

## Code Overview

### Naming Conventions

tbd

### Notable Modules

* `__main__.py`
  * Implementation of the command-line interface.
  * Calls functions implemented in other modules.
* `analysis/__init__.py`
  * Implementation of analysis functionality, in function `analyze()`.
  * Prepares data, creates output directory, scans config, calls AutoML backend.
* `evaluation/__init__.py`
  * Implementation of evaluation functionality, in function `evaluate()`.
  * Prepares data, creates output directory, applies models, generates statistics and plots.
* `automl/base.py`
  * Abstract interface for AutoML backends.
  * Resembles the `BaseEstimator` interface of scikit-learn.
* `base/config.py`
  * Default configuration.
* `automl/askl/__init__.py`
  * Implementation of the auto-sklearn backend.
  * Necessary packages are only imported if this backend is actually used; other backends should follow a similar
      pattern.
* `ood/base.py`
  * Abstract base class for out-of-distribution (OOD) detection.
  * Contains factory method for creation of sub-classes.
* `ood/internal/`
  * Folder for OOD detectors implemented in CaTaBra.
* `ood/internal/`
  * Folder for OOD detectors implemented in CaTaBra.
* `ood/pyod.py`
  * Wrapper class for making PyOD detectors confirm with the base class. 
  * [PyOD](https://pyod.readthedocs.io/en/latest/) is a library for anonmaly detection
* `util/encofing.py`
  * Implementation of the feature encoding functionality.
  * Responsible for converting time-like attributes into numeric attributes, among other things.
  * Implements the scikit-learn `BaseEstimator` interface, with methods `fit()`, `transform()` and
      `inverse_transform()`.
* `util/io.py`
  * Utility I/O functions, for tables/DataFrames, JSON serialization, and pickling.
* `util/logging.py`
  * Utility functions for printing messages and prompting the user for input.
  * Functionality for mirroring `print()`-statements to a log file.