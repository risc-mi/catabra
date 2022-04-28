# Built-in Metrics

This document contains a short description of all built-in metrics that are by default reported in the metrics.xlsx
files generated upon evaluation.

Each of the metrics is implemented by a function in the `util.metrics` module.

## Regression Metrics

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

## Classification Metrics

### Area under Receiver Operator Characteristic Curve (Binary)

* Implementation: `roc_auc`
* Also known as: ROC-AUC
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **yes**
* Applicable to multiclass problems: **no**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score),
    [Wikipedia](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

### Area under Receiver Operator Characteristic Curve - One-vs-Rest (Multiclass)

* Implementation: `roc_auc_ovr`
* Also known as: ROC-AUC
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **yes**
* Applicable to multiclass problems: **yes**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score),
    [Wikipedia](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
* **Note**: Equivalent to `roc_auc` with `multi_class="ovr"`.

### Area under Receiver Operator Characteristic Curve - One-vs-One (Multiclass)

* Implementation: `roc_auc_ovo`
* Also known as: ROC-AUC
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **yes**
* Applicable to multiclass problems: **yes**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score),
    [Wikipedia](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
* **Note**: Equivalent to `roc_auc` with `multi_class="ovo"`.

### Area under Receiver Operator Characteristic Curve - One-vs-Rest Weighted (Multiclass)

* Implementation: `roc_auc_ovr_weighted`
* Also known as: ROC-AUC
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **yes**
* Applicable to multiclass problems: **yes**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score),
    [Wikipedia](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
* **Note**: Equivalent to `roc_auc` with `multi_class="ovr"` and `average="weighted"`.

### Area under Receiver Operator Characteristic Curve - One-vs-One Weighted (Multiclass)

* Implementation: `roc_auc_ovo_weighted`
* Also known as: ROC-AUC
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **yes**
* Applicable to multiclass problems: **yes**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score),
    [Wikipedia](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
* **Note**: Equivalent to `roc_auc` with `multi_class="ovo"` and `average="weighted"`.

### Average Precision

* Implementation: `average_precision`
* Also known as: AP
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **yes**
* Applicable to multiclass problems: **no**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score),
    [Wikipedia](https://en.wikipedia.org/w/index.php?title=Information_retrieval&oldid=793358396#Average_precision)
* **Note**: *Not* equivalent to `pr_auc`, but similar.

### Area under Precision-Recall Curve

* Implementation: `pr_auc`
* Also known as: PR-AUC
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **yes**
* Applicable to multiclass problems: **no**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve)
* **Note**: *Not* equivalent to `average_precision`, but similar.

### Mean Average Precision

* Also known as: mAP
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **yes**
* Applicable to multiclass problems: **yes**
* Documentation:
    [Wikipedia](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision)
* **Note**: Unweighted mean of per-class average precision.

### Brier Score

* Implementation: `brier_loss`
* Range: [0, 1]
* Optimum: 0
* Accepts probabilities: **yes**
* Applicable to multiclass problems: **no**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html#sklearn.metrics.brier_score_loss),
    [Wikipedia](https://en.wikipedia.org/wiki/Brier_score)

### Hinge Loss

* Implementation: `hinge_loss`
* Range: [0, inf)
* Optimum: 0
* Accepts probabilities: **yes**
* Applicable to multiclass problems: **no**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hinge_loss.html#sklearn.metrics.hinge_loss),
    [Wikipedia](https://en.wikipedia.org/wiki/Hinge_loss)

### Log Loss

* Implementation: `log_loss`
* Also known as: logistic loss, cross-entropy loss
* Range: [0, inf)
* Optimum: 0
* Accepts probabilities: **yes**
* Applicable to multiclass problems: **no**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss),
    [Wikipedia](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression)

### Calibration Curve

* Implementation: `calibration_curve`
* Accepts probabilities: **yes**
* Applicable to multiclass problems: **no**
* **Note**: Not actually a metric, but a curve whose *x*-values correspond to threshold-bins and whose *y*-values
    correspond to the fraction of positive samples in each bin. Ideally, the curve should be monotonically increasing.

### Confusion Matrix

* Implementation: `confusion_matrix`
* Accepts probabilities: **no**
* Applicable to multiclass problems: **yes**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix),
    [Wikipedia](https://en.wikipedia.org/wiki/Confusion_matrix)

### Accuracy

* Implementation: `accuracy`
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **no**
* Applicable to multiclass problems: **yes**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score)
* **Note**: *Not* equivalent to `jaccard`, although this is claimed by the scikit-learn documentation.

### Balanced Accuracy

* Implementation: `balanced_accuracy`
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **no**
* Applicable to multiclass problems: **yes**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score)

### F1

* Implementation: `f1`
* Also known as: balanced F-score, F-measure
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **no**
* Applicable to multiclass problems: **yes** (micro-, macro- and weighted averaging)
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score),
    [Wikipedia](https://en.wikipedia.org/wiki/F1_score)
* **Note**: Special case of the F-beta metric, with beta=1. Harmonic mean of sensitivity and specificity.

### Sensitivity

* Implementation: `sensitivity`
* Also known as: recall, true positive rate, hit rate
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **no**
* Applicable to multiclass problems: **yes** (micro-, macro- and weighted averaging)
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score),
    [Wikipedia](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)

### Specificity

* Implementation: `specificity`
* Also known as: selectivity, true negative rate
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **no**
* Applicable to multiclass problems: **yes** (micro-, macro- and weighted averaging)
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score),
    [Wikipedia](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
* **Note**: Can be computed in the same way as sensitivity, by exchanging the positive and negative class.

### Positive Predictive Value

* Implementation: `positive_predictive_value`
* Also known as: precision, PPV
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **no**
* Applicable to multiclass problems: **yes** (micro-, macro- and weighted averaging)
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score),
    [Wikipedia](https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values)

### Negative Predictive Value

* Implementation: `negative_predictive_value`
* Also known as: NPV
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **no**
* Applicable to multiclass problems: **yes** (micro-, macro- and weighted averaging)
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score),
    [Wikipedia](https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values)
* **Note**: Can be computed in the same way as positive predictive value, by exchanging the positive and negative class.

### Cohen's Kappa

* Implementation: `cohen_kappa`
* Range: [-1, 1]
* Optimum: 1
* Accepts probabilities: **no**
* Applicable to multiclass problems: **yes**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html#sklearn.metrics.cohen_kappa_score),
    [Wikipedia](https://en.wikipedia.org/wiki/Cohen%27s_kappa)

### Hamming Loss

* Implementation: `hamming_loss`
* Also known as: Hamming distance
* Range: [0, 1]
* Optimum: 0
* Accepts probabilities: **no**
* Applicable to multiclass problems: **yes**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html#sklearn.metrics.hamming_loss),
    [Wikipedia](https://en.wikipedia.org/wiki/Hamming_distance)
* **Note**: Equivalent to `1 - accuracy`.

### Jaccard Index

* Implementation: `jaccard`
* Also known as: Jaccard similarity coefficient
* Range: [0, 1]
* Optimum: 1
* Accepts probabilities: **no**
* Applicable to multiclass problems: **yes** (micro-, macro- and weighted averaging)
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html#sklearn.metrics.jaccard_score),
    [Wikipedia](https://en.wikipedia.org/wiki/Jaccard_index)
* **Note**: *Not* equivalent to `accuracy`, although this is claimed by the scikit-learn documentation.

### Matthews Correlation Coefficient

* Implementation: `matthews_correlation_coefficient`
* Also known as: MCC, phi coefficient
* Range: [-1, 1]
* Optimum: 1
* Accepts probabilities: **no**
* Applicable to multiclass problems: **yes**
* Documentation:
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html#sklearn.metrics.matthews_corrcoef),
    [Wikipedia](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient)

### True Positives

* Accepts probabilities: **no**
* Applicable to multiclass problems: **no**
* **Note**: Total number of true positives, i.e., correctly predicted positive samples.
    (1,1)-th entry of `confusion_matrix`.

### True Negatives

* Accepts probabilities: **no**
* Applicable to multiclass problems: **no**
* **Note**: Total number of true negatives, i.e., correctly predicted negative samples.
    (0,0)-th entry of `confusion_matrix`.

### False Positives

* Accepts probabilities: **no**
* Applicable to multiclass problems: **no**
* **Note**: Total number of false positives, i.e., negative samples wrongly predicted as positive.
    (0,1)-th entry of `confusion_matrix`.

### False Negatives

* Accepts probabilities: **no**
* Applicable to multiclass problems: **no**
* **Note**: Total number of false negatives, i.e., positive samples wrongly predicted as negative.
    (1,0)-th entry of `confusion_matrix`.

### Balance Threshold

* Implementation: `balance_threshold`
* Accepts probabilities: **yes**
* Applicable to multiclass problems: **no**
* **Note**: Not actually a metric, but the decision threshold which minimizes the difference between sensitivity and
    specificity.


### Balance Score

* Implementation: `balance_score`
* Accepts probabilities: **yes**
* Applicable to multiclass problems: **no**
* **Note**: Equivalent to sensitivity at decision threshold `balance_threshold`, which by definition is (approximately)
    equal to specificity at that threshold. Moreover, it can be shown to be (approximately) equal to accuracy and
    balanced accuracy at that threshold, too.

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
averaging policies are **micro**, **macro** and **weighted**; see below for details. The desired averaging policy can
be selected by passing a suitable value for parameter `average` of the corresponding function.

### Multilabel Classification

Binary classification metrics can be applied to each class of a multilabel problem separately, and then averaged to
obtain a single scalar value. Three averaging policies are supported by default, and can be specified via the `average`
parameter of the corresponding function:
* **micro**: Metrics are computed globally by counting the total true positives, true negatives, false positives and
    false negatives across all classes.
* **macro**: Unweighted mean of per-class metric values.
* **weighted**: Weighted mean of per-class metric values, with weights corresponding to the number of positive
    instances of each class.

**Note**: Some metrics, like `accuracy`, are defined for multilabel problems even without averaging. What is reported
in metrics.xlsx are still the averaged versions, though.