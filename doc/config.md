## Configuring CaTabRa

CaTabRa can be configured via *config files*. These config files control the behavior of the CaTabRa command-line
interface (as described in `doc/cli.md`).

Config files are JSON files with a single top-level object. The possible parameters are described below.
See `core/config.py` for further information, including the default values of the parameters.

## General Configuration Parameters
* `"automl"`: AutoML backend to use. Currently, the only available option is `"auto-sklearn"`
    ([Link](https://automl.github.io/auto-sklearn/master/)), but new backends can be added by subclassing
    `catabra.automl.base.AutoMLBackend`.
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
* `"explainer"`: Explanation backend to use. Currently, the only available option is `"shap"`
    ([Link](https://github.com/slundberg/shap)), but new backends can be added by subclassing
    `catabra.explanation.base.EnsembleExplainer`.
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
* `"ood_class"`: Name of the module or class used for out-of-distribution (OOD) detection, or `None` to disable
    creating OOD-detectors. The possible values depend on the value of config param `"ood_source"`:
  * If `"ood_source"` is `"internal"`: name of one of the modules in `catabra.ood.internal`.
  * If `"source"` is `"pyod"`: name of one of the modules in `pyod.models`.
      **Note**: [PyOD](https://github.com/yzhao062/pyod) is not installed by default, but must be installed separately!
  * If `"source"` is `"external"`: full import path consisting of module and class (e.g. `custom.module.CustomOOD`).
* `"ood_source"`: The source from which to take the OOD-detector specified by `"ood_class"` (see above).
* `"ood_kwargs"`: Additional keyword arguments passed to the selected OOD-detector.

## auto-sklearn Specific Parameters

The following parameters are specific to the [auto-sklearn](https://automl.github.io/auto-sklearn/master/) backend.
See [this page](https://automl.github.io/auto-sklearn/master/api.html) for details.
* `"auto-sklearn_include"`: Components that are included in hyperparameter optimization, step-wise. Set to `null` to include all
    available components.
* `"auto-sklearn_exclude"`: Components that are excluded from hyperparameter optimization, step-wise. Set to `null` to exclude no
    components.
* `"auto-sklearn_resampling_strategy"`: The resampling strategy to use for internal validation. Note that if a string is passed,
    like `"holdout"` or `"cv"`, the 'grouped' version of the corresponding strategy is used if grouping is specified
    (e.g. via the `--group` argument of command `analyze`).
* `"auto-sklearn_resampling_strategy_arguments"`: Additional arguments for the resampling strategy, like the number of folds in
    *k*-fold cross validation.