# Command-Line Interface

CaTabRa is conceived as both a *library* and a *command-line tool*. The most straight-forward way to use CaTabRa is
via the command-line. There is one main command:

```
$ python -m catabra ...
```

This command has various sub-commands, listed below. You can get basic information about them by invoking

```
$ python -m catabra -h
```

## Analyzing New Data & Training Prediction Models

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
* `--ignore IGNORE`: List of columns to ignore when training prediction models. Automatically includes `GROUP`, `SPLIT`
    and `SAMPLE_WEIGHT`, if specified, but may contain further columns.
* `--calibrate CALIBRATE`: Value in column `SPLIT` defining the subset to calibrate the trained classifier on.
    If omitted, no calibration happens.
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
* `--monitor MONITOR`: Enable live monitoring of the training progress. The optional `MONITOR` argument specifies which
    backend to use (default is `"plotly"`). Keyword arguments can be passed as sequences of assignments, as in
    `"plotly update_interval=0"`.
* `--from FROM`: Path to an invocation.json file. All command-line arguments not explicitly specified are taken from
    this file; this also includes `TABLE`.
    
    **Note**: If the invocation.json file specifies a grouping- or splitting column, or an alternative config file, but
    you want to disable grouping / splitting / alternative config, you can simply pass `--group` / `--split` /
    `--config` without argument.

## Calibrating CaTabRa Classifiers

Trained CaTabRa classifiers can be calibrated to ensure that the probability estimates they return correspond to the
"true" confidence of the model. Citing scikit-learn:

> Well calibrated classifiers are probabilistic classifiers for which the output of the `predict_proba()`
method can be directly interpreted as a confidence level. For instance, a well calibrated (binary)
classifier should classify the samples such that among the samples to which it gave a `predict_proba`-value
close to 0.8, approximately 80% actually belong to the positive class.

The corresponding sub-command is `calibrate` and can be invoked as follows:
```
$ python -m catabra calibrate ...
```
Further information about the command can be obtained via
```
$ python -m catabra calibrate -h
```

The command has one positional argument, `SOURCE`, which is the directory containing the existing CaTabRa classifier to
calibrate. This is the output directory of a previous invocation of `analyze`.

In addition, there are optional arguments as well:
* `--on TABLE [TABLE_2 TABLE_3 ...]`: File(s) containing the table(s) on which the CaTabRa object shall be calibrated.
    The same restrictions apply to `TABLE` as with command `analyze`, described above. In particular, `TABLE` is
    expected to contain all necessary feature- and label columns. Labels may be `NaN`.
* `--split SPLIT`: Column used for splitting the data into disjoint subsets. In conjunction with `SUBSET` this enables
    restricting the data used for calibration to a subset of `TABLE`.
* `--subset SUBSET`: Value in column `SPLIT` to consider for calibration. For instance, if the column specified by
    `SPLIT` contains values `"train"`, `"val"` and `"test"`, and `SUBSET` is set to `"val"`, the classifier is
    calibrated only on the `"val"`-entries.
* `--method METHOD`: Calibration method. Must be one of `"sigmoid"`, `"isotonic"` or `"auto"` (default).
* `--sample-weight SAMPLE_WEIGHT`: Column with sample weights, which are used for calibration.
* `--out OUT`: Directory where to save all generated artifacts. Defaults to a directory located in `SOURCE`, with a
    name following a fixed naming pattern. If `OUT` already exists, the user is prompted to specify whether it should
    be replaced; otherwise, it is automatically created. `.` serves as a shortcut for the current working directory.
* `--from FROM`: Path to an invocation.json file. All command-line arguments not explicitly specified are taken from
    this file; this also includes `SOURCE` and `TABLE`.
    
    **Note**: If the invocation.json file specifies a splitting column and a calibration subset, but you want to
    calibrate on all data in `TABLE`, you can simply pass `--split` and/or `--subset` without argument.

## Evaluating CaTabRa Objects on New Data

After analyzing data, the resulting (possibly calibrated) models can be evaluated on held-out test data.
The corresponding sub-command is `evaluate` and can be invoked as follows:
```
$ python -m catabra evaluate ...
```
Further information about the command can be obtained via
```
$ python -m catabra evaluate -h
```

The command has one positional argument, `SOURCE`, which is the directory containing the existing CaTabRa object to
evaluate. This is the output directory of a previous invocation of `analyze`.

In addition, there are optional arguments as well:
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
    specified in `FROM`. In binary classification this can also be the name of a built-in thresholding strategy,
    possibly followed by "on" and the split on which to calculate the threshold. Splits must be specified by the name
    of the subdirectory containing the corresponding evaluation results. See /doc/metrics.md for a list of built-in
    thresholding strategies.
* `--bootstrapping-repetitions BS_REPETITIONS`: Number of bootstrapping repetitions. Set to 0 to disable bootstrapping.
    Overwrites config param `"bootstrapping_repetitions"`.
* `--bootstrapping-metrics M_1 [M_2 M_3 ...]`: Metrics for which to report bootstrapped results. Can also be
    `__all__`, in which case all suitable metrics are included. Ignored if bootstrapping is disabled.
* `--from FROM`: Path to an invocation.json file. All command-line arguments not explicitly specified are taken from
    this file; this also includes `SOURCE` and `TABLE`.
    
    **Note**: If the invocation.json file specifies a model-ID or splitting column, but you want to evaluate the whole
    ensemble or disable splitting, you can simply pass `--model-id` / `--split` without argument.

## Explaining CaTabRa Objects (Prediction Models)

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

In addition, there are optional arguments as well:
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
* `--explainer EXPLAINER`: Name of the explainer to use. Defaults to the first explainer specified in config param
    `"explainer"`. Note that only explainers that were fitted to training data during `analyze` can be used, as well as
    explainers that do not need to be fit to training data. Pass `-e` without arguments to get a list of all available
    explainers.
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
* `--aggregation-mapping`: Mapping from target column names to lists of source column names in `ON`, whose explanations
    will be aggregated by the explainer's aggregation function. Must be the name of a JSON file containing a
    corresponding dict.
* `--jobs JOBS`: Number of jobs to use. Overwrites the `"jobs"` config param.
* `--from FROM`: Path to an invocation.json file. All command-line arguments not explicitly specified are taken from
    this file; this also includes `SOURCE` and `TABLE`.
    
    **Note**: If the invocation.json file specifies a model-ID or splitting column, but you want to explain all models
    in the ensemble or disable splitting, you can simply pass `--model-id` / `--split` without argument.

## Applying CaTabRa Objects to New Data

Trained models (and OOD detectors) can be applied to new, unlabeled data. The corresponding sub-command is `apply` and
can be invoked as follows:
```
$ python -m catabra apply ...
```
Further information about the command can be obtained via
```
$ python -m catabra apply -h
```

The command has one positional argument, `SOURCE`, which is the directory containing the existing CaTabRa object to
apply. This is the output directory of a previous invocation of `analyze`.

In addition, there are optional arguments as well:
* `--on TABLE [TABLE_2 TABLE_3 ...]`: File(s) containing the table(s) the CaTabRa object shall be applied to.
    The same restrictions apply to `TABLE` as with command `analyze`, described above. In particular, `TABLE` is
    expected to contain all necessary feature columns, but in contrast to `analyze` and `evaluate` does not need to
    contain any target columns.
* `--model-id [MODEL_ID ...]`: Identifier(s) of the prediction model(s) to apply. By default, all models in the final
    ensemble are explained. In contrast to `evaluate`, more than one ID can be specified.
    Check out `SOURCE/model_summary.json` for all available model-IDs.
* `--explain [EXPLAIN ...]`: Explain prediction model(s). If passed without arguments, all models specified by
    `MODEL_ID` are explained; otherwise, `EXPLAIN` contains the model ID(s) to explain.
* `--no-ood`: Disable out-of-distribution (OOD) detection.
* `--out OUT`: Directory where to save all generated artifacts. Defaults to a directory located in `SOURCE`, with a
    name following a fixed naming pattern. If `OUT` already exists, the user is prompted to specify whether it should
    be replaced; otherwise, it is automatically created. `.` serves as a shortcut for the current working directory.
* `--batch-size BATCH_SIZE`: Batch size used for applying the model(s).
* `--jobs JOBS`: Number of jobs to use. Overwrites the `"jobs"` config param.
* `--from FROM`: Path to an invocation.json file. All command-line arguments not explicitly specified are taken from
    this file; this also includes `SOURCE` and `TABLE`.
    
    **Note**: If the invocation.json file specifies a model-ID, but you want to apply all models
    in the ensemble, you can simply pass `--model-id` without arguments.