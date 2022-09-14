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

## Command-Line Interface

CaTabRa is conceived as both a *library* and a *command-line tool*. The most straight-forward way to use CaTabRa is
via the command-line. There is one main command that must be invoked in a directory where CaTabRa is visible, which
is either the directory containing this README file or *any* directory if CaTabRa was installed via pip:

```
$ python -m catabra ...
```

This command has various sub-commands, described below. You can get basic information about them by invoking

```
$ python -m catabra -h
```

### Analyzing New Data & Training Prediction Models

The first step when using CaTabRa is analyzing tabular data. *Analyzing*, in this context, includes generating
descriptive statistics, approximating the underlying data distribution for being able to detect out-of-distribution
(OOD) samples, and training classification- or regression models for a given set of target attributes.
The corresponding sub-command is `analyze` and can be invoked as follows:
```
$ python -m catabra analyze ...
```
Further information about the command can be obtained via
```
$ python -m catabra analyze -h
```

The command has one positional argument, `TABLE`, which must be the path to an existing CSV, Excel or HDF5 file
containing the data to analyze. It is also possible to provide paths to multiple files, in which case the
corresponding tables are merged column-wise. The table(s) must meet the following requirements:
* Each row must correspond to a single sample, and each column to an attribute.
* Attributes can be either features, labels, or auxiliary information that is ignored during analysis
    (e.g., identifiers). How this can be specified is described below.
* All attributes may contain `NaN` values.
* In principle, all data types are supported, but not all can be used for analysis.
  * Data types that can be used for analysis are numerical types (`float`, `int`, `bool`), `Categorical`, and time-like
      types (`Timestamp`, `TimeDelta`)
  * Note that immediately after reading a table, `object` data types are first tried to be converted into `TimeDelta`
      or `Timestamp` types (in that order) by using pandas' `to_timedelta()` and `to_datetime()` functions,
      respectively. If this is not possible and the number of unique values does not exceed a predefined threshold (100
      by default), the type is converted into `Categorical`. Otherwise, no conversion happens.
* Time-like columns (`Timestamp`, `TimeDelta`) are converted into floats by dividing through some automatically-chosen
    resolution (1 year, 1 day, 1 hour, 1 minute, 1 second). This happens in a transparent way during feature encoding,
    meaning the conversion can easily be undone.
* Attributes designated as labels (see below) are automatically converted into floats. In the case of regression tasks,
    labels should be numeric anyway, and in classification tasks the resulting values represent the `n` distinct
    classes as `0.0`, ..., `n - 1.0`. The actual class names can be arbitrary, of arbitrary data type, and are stored
    transparently in the encoder.

In addition to the positional `TABLE` argument, optional arguments can be provided as well:
* `--classify CLASSIFY [CLASSIFY_1 CLASSIFY_2 ...]`: Column(s) to classify. If only one is specified, the prediction
    task is either a binary of multiclass classification problem, depending on the number of unique values. If more than
    one is specified, the prediction task is a multilabel classification problem, i.e., each of the columns
    corresponds to a class, and each sample can belong to an arbitrary subset of the classes. This means that the
    values in these columns must be binary class indicators, e.g., boolean or `0`/`1` ints or floats.
    
    **Mutually exclusive with `--regress`!**
* `--regress REGRESS [REGRESS_1 REGRESS_2 ...]`: Column(s) to regress. Each of the columns is treated independently, so
    it does not make any difference how many are specified. Data types must be numeric.
    
    **Mutually exclusive with `--classify`!**
* `--group GROUP`: Column used for grouping samples for internal (cross) validation. If specified, all samples with the
    same value in this column are put into the same split, i.e., either all into the internal training data or all into
    the internal validation data. By default, samples are grouped by the row index of `TABLE` if it has name; otherwise,
    grouping is disabled.
* `--split SPLIT`: Column used for splitting the data into train- and test set. If specified, descriptive  statistics,
    OOD-detectors and prediction models are generated based exclusively on the training split and then automatically
    evaluated on the test split. The name and/or values of the column must contain the string "train", "test" or "val",
    to clearly indicate what is the training- and what is the test data.
* `--sample-weight SAMPLE_WEIGHT`: Column with sample weights, which are used for training, evaluating and
    explaining prediction models.
    Note, however, that not all AutoML-backends may support sample weights (auto-sklearn currently does not).
* `--ignore IGNORE`: List of columns to ignore when training prediction models. Automatically includes `GROUP` and
    `SPLIT`, if specified, but may contain further columns.
* `--out OUT`: Directory where to save all generated artifacts. Defaults to a directory located in the parent directory
    of `TABLE`, with a name following a fixed naming pattern. If `OUT` already exists, the user is prompted to specify
    whether it should be replaced; otherwise, it is automatically created. `.` serves as a shortcut for the current
    working directory.
* `--time TIME`: Time budget for model training, in minutes. Some AutoML backends require a fixed budget, others might
    not. Overwrites the `"time_limit"` config param. `0` means that no prediction models are trained.
* `--jobs JOBS`: Number of jobs to use. Overwrites the `"jobs"` config param.
* `--config CONFIG`: Path to a JSON file containing an alternative config dict. Merged with the default config specified
    via parameter `--default-config`. See Section Configuration below for details.
* `--default-config DEFAULT_CONFIG`: Default config to use. Possible values are `full` (default; full range of
    preprocessing steps and ML algorithms for model training), `basic` (only very basic preprocessing and ML algorithms)
    and `interpretable` (only inherently interpretable preprocessing and ML algorithms).
* `--from FROM`: Path to an invocation.json file. All command-line arguments not explicitly specified are taken from
    this file; this also includes `TABLE`.
    
    **Note**: If the invocation.json file specifies a grouping- or splitting column, or an alternative config file, but
    you want to disable grouping / splitting / alternative config, you can simply pass `--group` / `--split` /
    `--config` without argument.

### Evaluating CaTabRa Objects on New Data

After analyzing data, the resulting models can be evaluated on held-out test data.  The corresponding sub-command is
`evaluate` and can be invoked as follows:
```
$ python -m catabra evaluate ...
```
Further information about the command can be obtained via
```
$ python -m catabra evaluate -h
```

The command has one positional argument, `SOURCE`, which is the directory containing the existing CaTabRa object to
evaluate. This is the output directory of a previous invocation of `analyze`.

In addition, there optional arguments as well:
* `--on TABLE [TABLE_2 TABLE_3 ...]`: File(s) containing the table(s) on which the CaTabRa object shall be evaluated.
    The same restrictions apply to `TABLE` as with command `analyze`, described above. In particular, `TABLE` is
    expected to contain all necessary feature- and label columns. Labels may be `NaN`.
* `--split SPLIT`: Column used for splitting the data into disjoint subsets. If specified, each subset is evaluated
    individually. In contrast to command `analyze`, the name/values of the column do not need to carry any semantic
    information about training and test sets.
* `--sample-weight SAMPLE_WEIGHT`: Column with sample weights, which are used for evaluating prediction models.
* `--model-id MODEL_ID`: Identifier of the prediction model to evaluate. By default, the sole trained model or the
    entire ensemble are evaluated. Check out `SOURCE/model_summary.json` for all available model-IDs.
* `--explain [EXPLAIN ...]`: Explain prediction model(s). If passed without arguments, all models specified by
    `MODEL_ID` are explained; otherwise, `EXPLAIN` contains the model ID(s) to explain. By default, explanations are
    local, but this behavior can be overwritten by passing flag `--global`. See command `explain` for details.
* `--global`: Create global explanations rather than local ones.
* `--out OUT`: Directory where to save all generated artifacts. Defaults to a directory located in `SOURCE`, with a
    name following a fixed naming pattern. If `OUT` already exists, the user is prompted to specify whether it should
    be replaced; otherwise, it is automatically created. `.` serves as a shortcut for the current working directory.
* `--batch-size BATCH_SIZE`: Batch size used for applying the prediction model.
* `--jobs JOBS`: Number of jobs to use. Overwrites the `"jobs"` config param.
* `--threshold THRESHOLD`: Decision threshold for binary- and multilabel classification. Defaults to 0.5, unless
    specified in `FROM`.
* `--bootstrapping-repetitions BS_REPETITIONS`: Number of bootstrapping repetitions. Set to 0 to disable bootstrapping.
    Overwrites config param `"bootstrapping_repetitions"`.
* `--bootstrapping-metrics M_1 [M_2 M_3 ...]`: Metrics for which to report bootstrapped results. Can also be
    `__all__`, in which case all suitable metrics are included. Ignored if bootstrapping is disabled.
* `--from FROM`: Path to an invocation.json file. All command-line arguments not explicitly specified are taken from
    this file; this also includes `SOURCE` and `TABLE`.
    
    **Note**: If the invocation.json file specifies a model-ID or splitting column, but you want to evaluate the whole
    ensemble or disable splitting, you can simply pass `--model-id` / `--split` without argument.

### Explaining CaTabRa Objects (Prediction Models)

After analyzing data, the resulting models can be explained in terms of feature importance. The corresponding
sub-command is `explain` and can be invoked as follows:
```
$ python -m catabra explain ...
```
Further information about the command can be obtained via
```
$ python -m catabra explain -h
```

The command has one positional argument, `SOURCE`, which is the directory containing the existing CaTabRa object to
explain. This is the output directory of a previous invocation of `analyze`.

In addition, there optional arguments as well:
* `--on TABLE [TABLE_2 TABLE_3 ...]`: File(s) containing the table(s) on which the CaTabRa object shall be explained.
    The same restrictions apply to `TABLE` as with command `analyze`, described above. In particular, `TABLE` is
    expected to contain all necessary feature columns, but in contrast to `analyze` and `evaluate` does not need to
    contain any target columns.
* `--split SPLIT`: Column used for splitting the data into disjoint subsets. If specified, `SOURCE` is explained on
    each split separately.
* `--sample-weight SAMPLE_WEIGHT`: Column with sample weights, which may be used for explaining prediction models.
    Only relevant for global explanations depending on background samples.
* `--model-id [MODEL_ID ...]`: Identifier(s) of the prediction model(s) to explain. By default, all models in the final
    ensemble are explained. In contrast to `evaluate`, more than one ID can be specified.
    Check out `SOURCE/model_summary.json` for all available model-IDs.
* `--global`: Create global explanations. If specified, `TABLE` might not be required (depends on the explanation
    backend). Mutually exclusive with `--local`. If neither `--global` nor `--local` are specified, global or local
    explanations are generated depending on the explanation backend.
* `--local`: Create local explanations, in which case `TABLE` is mandatory.
    Mutually exclusive with `--global`. If neither `--global` nor `--local` are specified, global or local explanations
    are generated depending on the explanation backend.
* `--out OUT`: Directory where to save all generated artifacts. Defaults to a directory located in `SOURCE`, with a
    name following a fixed naming pattern. If `OUT` already exists, the user is prompted to specify whether it should
    be replaced; otherwise, it is automatically created. `.` serves as a shortcut for the current working directory.
* `--batch-size BATCH_SIZE`: Batch size used for generating explanations.
* `--jobs JOBS`: Number of jobs to use. Overwrites the `"jobs"` config param.
* `--from FROM`: Path to an invocation.json file. All command-line arguments not explicitly specified are taken from
    this file; this also includes `SOURCE` and `TABLE`.
    
    **Note**: If the invocation.json file specifies a model-ID or splitting column, but you want to explain all models
    in the ensemble or disable splitting, you can simply pass `--model-id` / `--split` without argument.

### Applying CaTabRa Objects to New Data

tbd

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
* `automl/askl/__init__.py`
  * Implementation of the auto-sklearn backend.
  * Necessary packages are only imported if this backend is actually used; other backends should follow a similar
      pattern.
* `util/config.py`
  * Default configuration.
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