# Metrics

## Built-in Regression Metrics

This section lists all built-in regression metrics that are implemented in the `util.metrics` module.

### R²

* Implementation: `r2`
* Also known as: coefficient of determination, R squared
* Range: (-inf, 1]
* Optimum: 1
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score),
    [Wikipedia](https://en.wikipedia.org/wiki/Coefficient_of_determination)

### Mean Absolute Error

* Implementation: `mean_absolute_error`
* Also known as: MAE
* Range: [0, inf)
* Optimum: 0
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error),
    [Wikipedia](https://en.wikipedia.org/wiki/Mean_absolute_error)

### Mean Squared Error

* Implementation: `mean_squared_error`
* Also known as: MSE
* Range: [0, inf)
* Optimum: 0
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error),
    [Wikipedia](https://en.wikipedia.org/wiki/Mean_squared_error)
* **Note**: Equivalent to `mean_tweedie_deviance` with `power=0`.

### Root Mean Squared Error

* Implementation: `root_mean_squared_error`
* Also known as: RMSE, root-mean-square deviation, RMSD
* Range: [0, inf)
* Optimum: 0
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error),
    [Wikipedia](https://en.wikipedia.org/wiki/Root-mean-square_deviation)

### Mean Squared Logarithmic Error

* Implementation: `mean_squared_log_error`
* Also known as: MSLE
* Range: [0, inf)
* Optimum: 0
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_log_error.html#sklearn.metrics.mean_squared_log_error)
* **Note**: Only defined for non-negative inputs.

### Median Absolute Error

* Implementation: `median_absolute_error`
* Also known as: MedAE
* Range: [0, inf)
* Optimum: 0
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html#sklearn.metrics.median_absolute_error)

### Mean Absolute Percentage Error

* Implementation: `mean_absolute_percentage_error`
* Also known as: MAPE, mean absolute percentage deviation, MAPD
* Range: [0, inf)
* Optimum: 0
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html#sklearn.metrics.mean_absolute_percentage_error)

### Max Error

* Implementation: `max_error`
* Also known as: maximum residual error
* Range: [0, inf)
* Optimum: 0
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.max_error.html#sklearn.metrics.max_error)

### Explained Variance

* Implementation: `explained_variance`
* Also known as: explained variance regression score
* Range: (-inf, 1]
* Optimum: 1
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score)

### Mean Poisson Deviance

* Implementation: `mean_poisson_deviance`
* Range: [0, inf)
* Optimum: 0
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_poisson_deviance.html#sklearn.metrics.mean_poisson_deviance)
* **Note**: Equivalent to `mean_tweedie_deviance` with `power=1`.

### Mean Gamma Deviance

* Implementation: `mean_gamma_deviance`
* Range: [0, inf)
* Optimum: 0
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_gamma_deviance.html#sklearn.metrics.mean_gamma_deviance)
* **Note**: Equivalent to `mean_tweedie_deviance` with `power=2`.

### Mean Tweedie Deviance

* Implementation: `mean_tweedie_deviance`
* Range: [0, inf)
* Optimum: 0
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_tweedie_deviance.html#sklearn.metrics.mean_tweedie_deviance)
* **Note**: Parameter `power` is the Tweedie power parameter. With `power=0` this metric is equivalent to
    `mean_squared_error`, with `power=1` it is equivalent to `mean_poisson_deviance`, and with `power=2` it is
    equivalent to `mean_gamma_deviance`.

## Built-in Classification Metrics

This section lists all built-in classification metrics that are implemented in the `util.metrics` module.

### Area under Receiver Operator Characteristic Curve

* Implementation:
  * binary: `roc_auc`
  * multiclass: `roc_auc_ovr`, `roc_auc_ovr_weighted`, `roc_auc_ovo`, `roc_auc_ovo_weighted`
  * multilabel: `roc_auc_micro`, `roc_auc_macro`, `roc_auc_samples`, `roc_auc_weighted`
* Also known as: ROC-AUC
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **yes**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score),
    [Wikipedia](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
* **Note**: `roc_auc_ovr` and `roc_auc_ovo` return macro-averaged values by default.
* **Note**: Refer to Section "Averaging" for information about micro-, macro-, samples- and weighted averaging.

### Average Precision

* Implementation:
  * binary: `average_precision`
  * multilabel: `average_precision_micro`, `average_precision_macro`, `average_precision_samples`,
      `average_precision_weighted`
* Also known as: AP, mean average precision (mAP) in case of `average_precision_macro`
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **yes**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score),
    [Wikipedia](https://en.wikipedia.org/w/index.php?title=Information_retrieval&oldid=793358396#Average_precision)
* **Note**: *Not* equivalent to `pr_auc`, but similar.
* **Note**: Refer to Section "Averaging" for information about micro-, macro-, samples- and weighted averaging.

### Area under Precision-Recall Curve

* Implementation:
  * binary: `pr_auc`
  * multilabel: `pr_auc_micro`, `pr_auc_macro`, `pr_auc_samples`, `pr_auc_weighted`
* Also known as: PR-AUC
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **yes**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve)
* **Note**: *Not* equivalent to `average_precision`, but similar.
* **Note**: Refer to Section "Averaging" for information about micro-, macro-, samples- and weighted averaging.

### Brier Score

* Implementation:
  * binary: `brier_loss`
  * multilabel: `brier_loss_micro`, `brier_loss_macro`, `brier_loss_samples`, `brier_loss_weighted`
* Range: [0, 1]
* Optimum: 0
* Accepts probabilities: **yes**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html#sklearn.metrics.brier_score_loss),
    [Wikipedia](https://en.wikipedia.org/wiki/Brier_score)
* **Note**: Refer to Section "Averaging" for information about micro-, macro-, samples- and weighted averaging.

### Hinge Loss

* Implementation:
  * binary: `hinge_loss`
  * multilabel: `hinge_loss_micro`, `hinge_loss_macro`, `hinge_loss_samples`, `hinge_loss_weighted`
* Range: [0, inf)
* Optimum: 0
* Accepts probabilities: **yes**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hinge_loss.html#sklearn.metrics.hinge_loss),
    [Wikipedia](https://en.wikipedia.org/wiki/Hinge_loss)
* **Note**: Refer to Section "Averaging" for information about micro-, macro-, samples- and weighted averaging.

### Log Loss

* Implementation:
  * binary: `log_loss`
  * multilabel: `log_loss_micro`, `log_loss_macro`, `log_loss_samples`, `log_loss_weighted`
* Also known as: logistic loss, cross-entropy loss
* Range: [0, inf)
* Optimum: 0
* Accepts probabilities: **yes**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss),
    [Wikipedia](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression)
* **Note**: Refer to Section "Averaging" for information about micro-, macro-, samples- and weighted averaging.

### Calibration Curve

* Implementation:
  * binary: `calibration_curve`
* Accepts probabilities: **yes**
* **Note**: Not actually a metric, but a curve whose *x*-values correspond to threshold-bins and whose *y*-values
    correspond to the fraction of positive samples in each bin. Ideally, the curve should be monotonically increasing.

### Confusion Matrix

* Implementation:
  * binary & multiclass: `confusion_matrix`
* Accepts probabilities: **no**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix),
    [Wikipedia](https://en.wikipedia.org/wiki/Confusion_matrix)
* **Note**: Not actually a metric.

### Accuracy

* Implementation:
  * binary: `accuracy`
  * multiclass: `accuracy`, `accuracy_micro`, `accuracy_macro`, `accuracy_weighted`
  * multilabel: `accuracy`, `accuracy_micro`, `accuracy_macro`, `accuracy_samples`, `accuracy_weighted`
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **no**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score)
* **Note**: *Not* equivalent to `jaccard`, although this is claimed by the scikit-learn documentation.
* **Note**: `accuracy` is defined for multiclass and multilabel problems even without specifying an averaging policy.
* **Note**: Refer to Section "Averaging" for information about micro-, macro-, samples- and weighted averaging.

### Balanced Accuracy

* Implementation:
  * binary: `balanced_accuracy`
  * multiclass: `balanced_accuracy_micro`, `balanced_accuracy_macro`, `balanced_accuracy_weighted`
  * multilabel: `balanced_accuracy_micro`, `balanced_accuracy_macro`, `balanced_accuracy_samples`,
      `balanced_accuracy_weighted`
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **no**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score)
* **Note**: Refer to Section "Averaging" for information about micro-, macro-, samples- and weighted averaging.
* **Note**: Closely related to `informedness`, which is `balanced_accuracy * 2 - 1` in the binary case.

### F1

* Implementation:
  * binary: `f1`
  * multiclass: `f1_micro`, `f1_macro`, `f1_weighted`
  * multilabel: `f1_micro`, `f1_macro`, `f1_samples`, `f1_weighted`
* Also known as: balanced F-score, F-measure
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **no**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score),
    [Wikipedia](https://en.wikipedia.org/wiki/F1_score)
* **Note**: Special case of the F-beta metric, with beta=1. Harmonic mean of sensitivity and positive predictive value.
* **Note**: Refer to Section "Averaging" for information about micro-, macro-, samples- and weighted averaging.

### Sensitivity

* Implementation:
  * binary: `sensitivity`
  * multiclass: `sensitivity_micro`, `sensitivity_macro`, `sensitivity_weighted`
  * multilabel: `sensitivity_micro`, `sensitivity_macro`, `sensitivity_samples`, `sensitivity_weighted`
* Also known as: recall, true positive rate, hit rate
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **no**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score),
    [Wikipedia](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
* **Note**: Refer to Section "Averaging" for information about micro-, macro-, samples- and weighted averaging.

### Specificity

* Implementation:
  * binary: `specificity`
  * multiclass: `specificity_micro`, `specificity_macro`, `specificity_weighted`
  * multilabel: `specificity_micro`, `specificity_macro`, `specificity_samples`, `specificity_weighted`
* Also known as: selectivity, true negative rate
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **no**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score),
    [Wikipedia](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
* **Note**: Can be computed in the same way as sensitivity, by exchanging the positive and negative class.
* **Note**: Refer to Section "Averaging" for information about micro-, macro-, samples- and weighted averaging.

### Positive Predictive Value

* Implementation:
  * binary: `positive_predictive_value`
  * multiclass: `positive_predictive_value_micro`, `positive_predictive_value_macro`,
      `positive_predictive_value_weighted`
  * multilabel: `positive_predictive_value_micro`, `positive_predictive_value_macro`,
      `positive_predictive_value_samples`, `positive_predictive_value_weighted`
* Also known as: precision, PPV
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **no**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score),
    [Wikipedia](https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values)
* **Note**: Refer to Section "Averaging" for information about micro-, macro-, samples- and weighted averaging.

### Negative Predictive Value

* Implementation:
  * binary: `negative_predictive_value`
  * multiclass: `negative_predictive_value_micro`, `negative_predictive_value_macro`,
      `negative_predictive_value_weighted`
  * multilabel: `negative_predictive_value_micro`, `negative_predictive_value_macro`,
      `negative_predictive_value_samples`, `negative_predictive_value_weighted`
* Also known as: NPV
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **no**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score),
    [Wikipedia](https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values)
* **Note**: Can be computed in the same way as positive predictive value, by exchanging the positive and negative class.
* **Note**: Refer to Section "Averaging" for information about micro-, macro-, samples- and weighted averaging.

### Cohen's Kappa

* Implementation:
  * binary: `cohen_kappa`
  * multiclass: `cohen_kappa`, `cohen_kappa_micro`, `cohen_kappa_macro`, `cohen_kappa_weighted`
  * multilabel: `cohen_kappa_micro`, `cohen_kappa_macro`, `cohen_kappa_samples`, `cohen_kappa_weighted`
* Range: [-1, 1]
* Optimum: 1
* Accepts probabilities: **no**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html#sklearn.metrics.cohen_kappa_score),
    [Wikipedia](https://en.wikipedia.org/wiki/Cohen%27s_kappa)
* **Note**: `cohen_kappa` is defined for multiclass problems even without specifying an averaging policy.
* **Note**: Refer to Section "Averaging" for information about micro-, macro-, samples- and weighted averaging.

### Hamming Loss

* Implementation:
  * binary: `hamming_loss`
  * multiclass: `hamming_loss`, `hamming_loss_micro`, `hamming_loss_macro`, `hamming_loss_weighted`
  * multilabel: `hamming_loss`, `hamming_loss_micro`, `hamming_loss_macro`, `hamming_loss_samples`,
      `hamming_loss_weighted`
* Also known as: Hamming distance
* Range: [0, 1]
* Optimum: 0
* Accepts probabilities: **no**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html#sklearn.metrics.hamming_loss),
    [Wikipedia](https://en.wikipedia.org/wiki/Hamming_distance)
* **Note**: `hamming_loss` is equivalent to `1 - accuracy`.
* **Note**: `hamming_loss` is defined for multiclass problems even without specifying an averaging policy.
* **Note**: Refer to Section "Averaging" for information about micro-, macro-, samples- and weighted averaging.

### Jaccard Index

* Implementation:
  * binary: `jaccard`
  * multiclass: `jaccard_micro`, `jaccard_macro`, `jaccard_weighted`
  * multilabel: `jaccard_micro`, `jaccard_macro`, `jaccard_samples`, `jaccard_weighted`
* Also known as: Jaccard similarity coefficient, intersection over union, IoU
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **no**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html#sklearn.metrics.jaccard_score),
    [Wikipedia](https://en.wikipedia.org/wiki/Jaccard_index)
* **Note**: *Not* equivalent to `accuracy`, although this is claimed by the scikit-learn documentation.
* **Note**: Refer to Section "Averaging" for information about micro-, macro-, samples- and weighted averaging.

### Matthews Correlation Coefficient

* Implementation:
  * binary: `matthews_correlation_coefficient`
  * multiclass: `matthews_correlation_coefficient`, `matthews_correlation_coefficient_micro`,
      `matthews_correlation_coefficient_macro`, `matthews_correlation_coefficient_weighted`
  * multilabel: `matthews_correlation_coefficient_micro`, `matthews_correlation_coefficient_macro`,
      `matthews_correlation_coefficient_samples`, `matthews_correlation_coefficient_weighted`
* Also known as: MCC, phi coefficient
* Range: [-1, 1]
* Optimum: 1
* Accepts probabilities: **no**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html#sklearn.metrics.matthews_corrcoef),
    [Wikipedia](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient)
* **Note**: `matthews_correlation_coefficient` is defined for multiclass problems even without specifying an averaging
    policy.
* **Note**: Refer to Section "Averaging" for information about micro-, macro-, samples- and weighted averaging.

### Informedness

* Implementation:
  * binary: `informedness`
  * multiclass: `informedness_micro`, `informedness_macro`, `informedness_samples`, `informedness_weighted`
  * multilabel: `informedness_micro`, `informedness_macro`, `informedness_samples`, `informedness_weighted`
* Also known as: Youden index, Youden's J statistic
* Range: [-1, 1]
* Optimum: 1
* Accepts probabilities: **no**
* Documentation:
    [Wikipedia](https://en.wikipedia.org/wiki/Youden%27s_J_statistic)
* **Note**: Refer to Section "Averaging" for information about micro-, macro-, samples- and weighted averaging.
* **Note**: Informedness has a [natural generalization to the multiclass case](https://arxiv.org/abs/2010.16061),
    which is currently not implemented.
* **Note**: Closely related to `balanced_accuracy`, which is `(informedness + 1) / 2` in the binary case.

### Markedness

* Implementation:
  * binary: `markedness`
  * multiclass: `markedness_micro`, `markedness_macro`, `markedness_samples`, `markedness_weighted`
  * multilabel: `markedness_micro`, `markedness_macro`, `markedness_samples`, `markedness_weighted`
* Also known as: deltaP
* Range: [-1, 1]
* Optimum: 1
* Accepts probabilities: **no**
* Documentation:
    [Wikipedia](https://en.wikipedia.org/wiki/Confusion_matrix)
* **Note**: Refer to Section "Averaging" for information about micro-, macro-, samples- and weighted averaging.
* **Note**: Markedness has a [natural generalization to the multiclass case](https://arxiv.org/abs/2010.16061),
    which is currently not implemented.

### True Positives

* Accepts probabilities: **no**
* **Note**: Not actually a metric, but total number of true positives, i.e., correctly predicted positive samples.
    (1,1)-th entry of `confusion_matrix`.

### True Negatives

* Accepts probabilities: **no**
* **Note**: Not actually a metric, but total number of true negatives, i.e., correctly predicted negative samples.
    (0,0)-th entry of `confusion_matrix`.

### False Positives

* Accepts probabilities: **no**
* **Note**: Not actually a metric, but total number of false positives, i.e., negative samples wrongly predicted as
    positive. (0,1)-th entry of `confusion_matrix`.

### False Negatives

* Accepts probabilities: **no**
* **Note**: Not actually a metric, but total number of false negatives, i.e., positive samples wrongly predicted as
    negative. (1,0)-th entry of `confusion_matrix`.

### Balance Score

* Implementation:
  * binary: `balance_score`
* Accepts probabilities: **yes**
* **Note**: Equal to sensitivity at decision threshold `balance_threshold`, which by definition is (approximately)
    equal to specificity at that threshold. Moreover, it can be shown to be (approximately) equal to accuracy and
    balanced accuracy at that threshold, too.

### Prevalence Score

* Implementation:
  * binary: `prevalence_score`
* Accepts probabilities: **yes**
* **Note**: Equal to sensitivity at decision threshold `prevalence_threshold`, which can be shown to be
    (approximately) equal to positive predictive value and F1-score at that threshold.

## Built-in Classification Thresholding Strategies

### Balance Threshold

* Implementation: `balance_threshold`
* Usage in `--threshold` command-line argument: `"balance"`
* Definition: The decision threshold which minimizes the difference between sensitivity and specificity.

### Prevalence Threshold

* Implementation: `prevalence_threshold`
* Usage in `--threshold` command-line argument: `"prevalence"`
* Definition: The decision threshold which minimizes the difference between the total number of condition positive
    samples and the number of predicted positive samples.

### (0,1)-Threshold

* Implementation: `zero_one_threshold`
* Usage in `--threshold` command-line argument: `"zero_one"`, `"zero_one(<specificity_weight>)"`
* Definition: The decision threshold which minimizes the Euclidean distance between `(0, 1)` and
    `(1 - specificity, sensitivity)`.

### Argmax Threshold

* Implementation: `argmax_threshold`
* Usage in `--threshold` command-line argument: `"argmax <metric_name>"`
* Definition: The decision threshold which maximizes a given metric.

## Averaging

### Regression

If multiple regression targets are specified, regression metrics are computed for each target individually and for all
targets combined. The latter simply calls the corresponding functions on the ground-truth and prediction matrices and
relies on scikit-learn's built-in policy to handle such cases. Normally (though not necessarily always), this proceeds
by simply taking the unweighted mean of the individual metrics.

### Multiclass Classification

Binary classification metrics that do not naturally apply to multiclass problems, like the F1 score, can be computed
per-class and then averaged to obtain a single scalar value. To that end, the multiclass problem is cast as a special
case of a multilabel problem where always exactly one element of the multilabel indicator matrix is 1. The possible
averaging policies are **micro**, **macro**, **samples** and **weighted**; see below for details. The desired averaging
policy can be selected either by using a properly suffixed version of the function, like `f1_micro`, or by passing a
suitable value for parameter `average` of the non-suffixed function.

### Multilabel Classification

Binary classification metrics can be applied to each class of a multilabel problem separately, and then averaged to
obtain a single scalar value. Three averaging policies are supported by default, and can be specified either by using
a properly suffixed version of the function, or via the `average` parameter of the original function:
* **micro**: Metrics are computed globally by counting the total true positives, true negatives, false positives and
    false negatives across all classes.
* **macro**: Unweighted mean of per-class metric values.
* **samples**: Unweighted mean of per-sample metric values; only makes sense for multilabel tasks.
* **weighted**: Weighted mean of per-class metric values, with weights corresponding to the number of instances of each
    class.

**Note**: Some metrics, like `accuracy`, are defined for multilabel problems even without averaging. What is reported
in metrics.xlsx are still the averaged versions, though.

## Calculating Metrics from Raw Predictions

By default, CaTabRa automatically calculates suitable performance metrics when evaluating trained prediction models,
and saves them to disk in files called metrics.xlsx and (optionally) bootstrapping.xlsx. These metrics can easily be
computed manually as well; all that is required are sample-wise predictions (as saved in `"predictions.xlsx"`) and the
corresponding data encoder that can be easily obtained from a `CaTabRaLoader` object:

```python
from catabra.util import io
from catabra import evaluation

loader = io.CaTabRaLoader("CaTabRa_dir")
metrics, bootstrapping = evaluation.calc_metrics(
    "predictions.xlsx",
    loader.get_encoder(),
    bootstrapping_repetitions=...,
    bootstrapping_metrics=...
)
```