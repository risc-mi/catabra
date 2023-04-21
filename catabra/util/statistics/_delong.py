# code copied and adapted from https://github.com/jiesihu/AUC_Delongtest__python
#   (no license specified)

from typing import Optional, Tuple

import numpy as np
from scipy import stats


def delong_test(y_true: np.ndarray, y_hat_1: np.ndarray, y_hat_2: np.ndarray, sample_weight=None) -> float:
    """
    Compute the p-value of the DeLong test for the null hypothesis that two ROC-AUCs are equal.

    Parameters
    ----------
    y_true: np.ndarray
        Ground truth, 1D array of shape `(n_samples,)` with values in {0, 1}.
    y_hat_1: np.ndarray
        Predictions of the first classifier, 1D array of shape `(n_samples,)` with arbitrary values. Larger values
        correspond to a higher predicted probability that a sample belongs to the positive class.
    y_hat_2: np.ndarray
        Predictions of the second classifier, 1D array of shape `(n_samples,)` with arbitrary values. Larger values
        correspond to a higher predicted probability that a sample belongs to the positive class.
    sample_weight : np.ndarray, optional
        Sample weights. None defaults to uniform weights.

    Returns
    -------
    p_value: float
        p-value for the null hypothesis that the ROC-AUCs of the two classifiers are equal. If this value is smaller
        than a certain pre-defined threshold (e.g., 0.05) the null hypothesis can be rejected, meaning that there is a
        statistically significant difference between the two ROC-AUCs.

    See Also
    --------
    roc_auc_confidence_interval: Confidence interval for the ROC-AUC of a given classifier.
    """

    order, n_positive, ordered_sample_weight = _compute_ground_truth_statistics(y_true, sample_weight)
    predictions_sorted_transposed = np.vstack((y_hat_1, y_hat_2))[:, order]
    aucs, sigma = fast_delong(predictions_sorted_transposed, n_positive, sample_weight=ordered_sample_weight)
    return _calc_pvalue(aucs[0], aucs[1], sigma)


def roc_auc_confidence_interval(y_true: np.ndarray, y_hat: np.ndarray, alpha: float = 0.95,
                                sample_weight: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
    """
    Return the confidence interval and ROC-AUC of given ground-truth and model predictions.

    Parameters
    ----------
    y_true: np.ndarray
        Ground truth, 1D array of shape `(n_samples,)` with values in {0, 1}.
    y_hat: np.ndarray
        Predictions of the classifier, 1D array of shape `(n_samples,)` with arbitrary values. Larger values
        correspond to a higher predicted probability that a sample belongs to the positive class.
    alpha: float, default=0.95
        Confidence level, between 0 and 1.
    sample_weight: np.ndarray, optional
        Sample weights. None defaults to uniform weights.

    Returns
    -------
    auc: float
        ROC-AUC of the given ground-truth and predictions.
    ci_left: float
        Left endpoint of the confidence interval.
    ci_right: float
        Right endpoint of the confidence interval.

    Notes
    -----
    The output always satisfies `0 <= ci_left <= auc <= ci_right <= 1`.

    See Also
    --------
    delong_test: Statistical test for the null hypothesis that the ROC-AUCs of two classifiers are equal.
    """

    auc, auc_cov = _delong_roc_variance(y_true, y_hat, sample_weight=sample_weight)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)
    ci.clip(min=0, max=1, out=ci)
    return auc, ci[0], ci[1]


def _compute_midrank(x: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
    # AUC comparison adapted from https://github.com/Netflix/vmaf/
    jj = np.argsort(x)
    x_sorted = x[jj]
    if sample_weight is None:
        cumulative_weight = None
    else:
        cumulative_weight = np.cumsum(sample_weight[jj])
    n = len(x)
    t = np.zeros(n, dtype=np.float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        if cumulative_weight is None:
            t[i:j] = 0.5 * (i + j - 1) + 1
        else:
            t[i:j] = cumulative_weight[i:j].mean()
        i = j
    out = np.empty(n, dtype=np.float)
    out[jj] = t
    return out


def fast_delong(predictions_sorted_transposed: np.ndarray, n_positive: int,
                sample_weight: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    The fast version of DeLong's method for computing the covariance of unadjusted AUC [1].

    Parameters
    ----------
    predictions_sorted_transposed: np.ndarray
        2D array of shape `(n_classifiers, n_samples)` sorted such that positive samples are first.
    n_positive: int
        Number of positive samples.
    sample_weight: np.ndarray, optional
        Sample weights. None defaults to uniform weights.

    Returns
    -------
    aucs: np.ndarray
        ROC-AUC of each classifier, 1D array of shape `(n_classifiers,)`.
    sigma: np.ndarray
        Covariance matrix, 2D array of shape `(n_classifiers, n_classifiers)`.

    References
    ----------
    .. [1] Xu Sun and Weichao Xu. Fast Implementation of DeLong's Algorithm for Comparing the Areas Under Correlated
           Receiver Operating Characteristic Curves. IEEE Signal Processing Letters 21(11): 1389-1393, 2014.
    """
    # Short variables are named as they are in the paper

    m = n_positive
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    if sample_weight is None:
        positive_weight = negative_weight = None
    else:
        positive_weight = sample_weight[:m]
        negative_weight = sample_weight[m:]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)

    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :], sample_weight=positive_weight)
        ty[r, :] = _compute_midrank(negative_examples[r, :], sample_weight=negative_weight)
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :], sample_weight=sample_weight)

    if sample_weight is None:
        total_positive_weights = m
        total_negative_weights = n
        aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) * 0.5) / n
    else:
        total_positive_weights = positive_weight.sum()
        total_negative_weights = negative_weight.sum()
        pair_weights = np.dot(positive_weight[..., np.newaxis], negative_weight[np.newaxis])
        total_pair_weights = pair_weights.sum()

        aucs = (positive_weight * (tz[:, :m] - tx)).sum(axis=1) / total_pair_weights

    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
    v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n

    return aucs, delongcov


def _calc_pvalue(auc_1: float, auc_2: float, sigma: np.ndarray) -> float:
    v = np.array([[1, -1]])     # (1, 2)
    z = np.abs(auc_1 - auc_2) / (np.sqrt(np.dot(np.dot(v, sigma), v.T)) + 1e-8)[0, 0]
    return 2 * (1 - stats.norm.cdf(z))


def _compute_ground_truth_statistics(ground_truth, sample_weight=None):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def _delong_roc_variance(ground_truth: np.ndarray, predictions: np.ndarray, sample_weight: Optional[np.ndarray] = None):
    order, label_1_count, ordered_sample_weight = _compute_ground_truth_statistics(ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fast_delong(predictions_sorted_transposed, label_1_count, sample_weight=ordered_sample_weight)

    return aucs[0], delongcov
