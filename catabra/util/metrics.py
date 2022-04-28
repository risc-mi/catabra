from typing import Union, Optional, Tuple
from functools import partial
import numpy as np
import pandas as pd
from sklearn import metrics as skl_metrics


# regression
r2 = skl_metrics.r2_score
mean_absolute_error = skl_metrics.mean_absolute_error
mean_squared_error = skl_metrics.mean_squared_error
root_mean_squared_error = partial(skl_metrics.mean_squared_error, squared=False),
mean_squared_log_error = skl_metrics.mean_squared_log_error
median_absolute_error = skl_metrics.median_absolute_error
mean_absolute_percentage_error = skl_metrics.mean_absolute_percentage_error
max_error = skl_metrics.max_error
explained_variance = skl_metrics.explained_variance_score
mean_tweedie_deviance = skl_metrics.mean_tweedie_deviance
mean_poisson_deviance = skl_metrics.mean_poisson_deviance
mean_gamma_deviance = skl_metrics.mean_gamma_deviance


# classification with probabilities
roc_auc = skl_metrics.roc_auc_score
average_precision = skl_metrics.average_precision_score
brier_loss = skl_metrics.brier_score_loss
hinge_loss = skl_metrics.hinge_loss
log_loss = skl_metrics.log_loss
roc_auc_ovr = partial(skl_metrics.roc_auc_score, multi_class='ovr')
roc_auc_ovo = partial(skl_metrics.roc_auc_score, multi_class='ovo')
roc_auc_ovr_weighted = partial(skl_metrics.roc_auc_score, multi_class='ovr', average='weighted')
roc_auc_ovo_weighted = partial(skl_metrics.roc_auc_score, multi_class='ovo', average='weighted')


def pr_auc_score(y_true, y_score, **kwargs) -> float:
    precision, recall, _ = skl_metrics.precision_recall_curve(y_true, y_score, **kwargs)
    return skl_metrics.auc(recall, precision)


def balance_score_threshold(y_true, y_score) -> Tuple[float, float]:
    """
    Compute the balance score and -threshold of a binary classification problem.
    :param y_true: Ground truth, with 0 representing the negative class and 1 representing the positive class. Must not
    contain NaN.
    :param y_score: Predicted scores, i.e., the higher a score the more confident the model is that the sample belongs
    to the positive class. Range is arbitrary.
    :return: Pair `(balance_score, balance_threshold)`, where `balance_threshold` is the decision threshold that
    minimizes the difference between sensitivity and specificity, i.e., it is defined as

        arg min_t |sensitivity(y_true, y_score >= t) - specificity(y_true, y_score >= t)|

    `balance_score` is the corresponding sensitivity value, which by definition is approximately equal to specificity
    and can furthermore be shown to be approximately equal to accuracy and balanced accuracy, too.
    """
    if len(y_true) != len(y_score):
        raise ValueError('Found input variables with inconsistent numbers of samples: %r' % [len(y_true), len(y_score)])
    elif len(y_true) == 0:
        return 0., 0.
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_score, pd.Series):
        y_score = y_score.values
    s = pd.DataFrame(data=dict(p=(y_true > 0), n=(y_true < 1)), index=y_score).groupby(level=0).sum()
    s.sort_index(ascending=False, inplace=True)
    n_pos = s['p'].sum()
    n_neg = s['n'].sum()
    tp = s['p'].cumsum().values
    tn = n_neg - s['n'].cumsum().values
    diff = tp * n_neg - tn * n_pos    # monotonically increasing; constant 0 if `n_pos == 0` or `n_neg == 0`
    i = np.argmin(np.abs(diff))
    th = s.index[i]
    if i + 1 < len(s):      # take midpoint of `i`-th and nearest *smaller* threshold
        th += s.index[i + 1]
        th *= 0.5
    return ((tn[i] / n_neg) if n_pos == 0 else (tp[i] / n_pos)), th


def balance_score(y_true, y_score) -> float:
    return balance_score_threshold(y_true, y_score)[0]


def balance_threshold(y_true, y_score) -> float:
    return balance_score_threshold(y_true, y_score)[1]


def calibration_curve(y_true: np.ndarray, y_score: np.ndarray, thresholds: Optional[np.ndarray] = None) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the calibration curve of a binary classification problem. The predicated class probabilities are binned and,
    for each bin, the fraction of positive samples is determined. These fractions can then be plotted against the
    midpoints of the respective bins. Ideally, the resulting curve will be monotonic increasing.
    :param y_true: Ground truth, array of shape `(n,)` with values among 0 and 1. Values must not be NaN.
    :param y_score: Predicated probabilities of the positive class, array of shape `(n,)` with arbitrary non-NaN values;
    in particular, the values do not necessarily need to correspond to probabilities or confidences.
    :param thresholds: The thresholds used for binning `y_score`. If None, suitable thresholds are determined
    automatically.
    :return: Pair `(fractions, thresholds)`, where `thresholds` is the array of thresholds of shape `(m,)`, and
    `fractions` is the corresponding array of fractions of positive samples in each bin, of shape `(m - 1,)`. Note that
    the `i`-th bin ranges from `thresholds[i]` to `thresholds[i + 1]`.
    """
    assert y_true.shape == y_score.shape
    if thresholds is None:
        thresholds = get_thresholds(y_score, n_max=40, add_half_one=False)
    if len(thresholds) < 2:
        return np.empty((0,), dtype=np.float32), thresholds
    return \
        np.array([y_true[(t1 <= y_score) & (y_score < t2)].mean() for t1, t2 in zip(thresholds[:-1], thresholds[1:])]),\
        np.array(thresholds)


def get_thresholds(y: np.ndarray, n_max: int = 100, add_half_one: Optional[bool] = None) -> list:
    n = min(np.isfinite(y).sum(), n_max)
    if n == 0:
        thresholds = set()
    elif n == 1:
        thresholds = {np.nanmin(y)}
    else:
        thresholds = np.sort(y[np.isfinite(y)])
        thresholds = set(thresholds[np.linspace(0, len(thresholds) - 1, n).round().astype(np.int32)])
    if add_half_one is True or (add_half_one is None and not (y < 0).any() and not (1 < y).any()):
        thresholds.update({0.5, 1.})
    thresholds = list(thresholds)
    thresholds.sort()
    return thresholds


# classification with thresholds
confusion_matrix = skl_metrics.confusion_matrix
precision_recall_fscore_support = skl_metrics.precision_recall_fscore_support
accuracy = skl_metrics.accuracy_score
balanced_accuracy = skl_metrics.balanced_accuracy_score
f1 = skl_metrics.f1_score
sensitivity = partial(skl_metrics.recall_score, zero_division=0)
specificity = partial(skl_metrics.recall_score, pos_label=0, zero_division=0)
positive_predictive_value = partial(skl_metrics.precision_score, zero_division=1)
negative_predictive_value = partial(skl_metrics.precision_score, pos_label=0, zero_division=1)
cohen_kappa = skl_metrics.cohen_kappa_score
matthews_correlation_coefficient = skl_metrics.matthews_corrcoef
jaccard = skl_metrics.jaccard_score
hamming_loss = skl_metrics.hamming_loss


def precision_recall_fscore_support_cm(*, tp=None, fp=None, tn=None, fn=None, beta: float = 1.0,
                                       average: Optional[str] = None, zero_division: Union[float, str] = 'warn'):
    tp, fp, tn, fn = _to_int_arrays(tp, fp, tn, fn)
    if np.isscalar(tp):
        b2 = beta * beta
        pr = _precision_cm_aux(tp, fp, tn, fn, zero_division=zero_division)
        rc = _recall_cm_aux(tp, fp, tn, fn, zero_division=zero_division)
        fb = (1 + b2) * (pr * rc) / (b2 * pr + rc)
        return pr, rc, fb, None
    elif average == 'binary':
        raise ValueError(_BINARY_AVERAGE_ERROR)
    elif average == 'micro':
        return precision_recall_fscore_support_cm(tp=tp.sum(), fp=fp.sum(), tn=tn.sum(), fn=fn.sum(), beta=beta,
                                                  zero_division=zero_division)
    else:
        b2 = beta * beta
        pr = _precision_cm_aux(tp, fp, tn, fn, zero_division=zero_division)
        rc = _recall_cm_aux(tp, fp, tn, fn, zero_division=zero_division)
        fb = (1 + b2) * (pr * rc) / (b2 * pr + rc)
        if average == 'macro':
            return pr.mean(), rc.mean(), fb.mean(), None
        else:
            support = tp + fn
            if average == 'weighted':
                s = support.sum()
                return (pr * support).sum() / s, (rc * support).sum() / s, (fb * support).sum() / s, None
            else:
                return pr, rc, fb, support


def precision_cm(*, tp=None, fp=None, tn=None, fn=None, average: Optional[str] = None, swap_pos_neg: bool = False,
                 zero_division: Union[float, str] = 'warn') -> Union[float, int, np.ndarray]:
    return _calc_single_cm_metric(_precision_cm_aux, tp, fp, tn, fn, average=average, zero_division=zero_division,
                                  swap_pos_neg=swap_pos_neg)


def _precision_cm_aux(tp, fp, tn, fn, zero_division: Union[float, str] = 'warn', swap_pos_neg: bool = False):
    if swap_pos_neg:
        return _safe_divide(tn, tn + fn, zero_division=zero_division)
    return _safe_divide(tp, tp + fp, zero_division=zero_division)


def recall_cm(*, tp=None, fp=None, tn=None, fn=None, average: Optional[str] = None,
              zero_division: Union[float, str] = 'warn', swap_pos_neg: bool = False) -> Union[float, int, np.ndarray]:
    return _calc_single_cm_metric(_recall_cm_aux, tp, fp, tn, fn, average=average, zero_division=zero_division,
                                  swap_pos_neg=swap_pos_neg)


def _recall_cm_aux(tp, fp, tn, fn, zero_division: Union[float, str] = 'warn', swap_pos_neg: bool = False):
    if swap_pos_neg:
        return _safe_divide(tn, tn + fp, zero_division=zero_division)
    return _safe_divide(tp, tp + fn, zero_division=zero_division)


def accuracy_cm(*, tp=None, fp=None, tn=None, fn=None, average: Optional[str] = 'binary', normalize: bool = True) \
        -> Union[float, int, np.ndarray]:
    """
    Calculate accuracy from a confusion matrix.
    ATTENTION! In the multilabel case, this implementation differs from the scikit-learn implementation.
    """
    return _calc_single_cm_metric(_accuracy_cm_aux, tp, fp, tn, fn, average=average, normalize=normalize)


def _accuracy_cm_aux(tp, fp, tn, fn, normalize: bool = True):
    correct = tp + tn
    if normalize:
        return _safe_divide(correct, correct + fp + fn, zero_division=1, inplace=True)
    else:
        return correct


def balanced_accuracy_cm(*, tp=None, fp=None, tn=None, fn=None, average: Optional[str] = 'binary',
                         adjusted: bool = False) -> Union[float, int, np.ndarray]:
    return _calc_single_cm_metric(_balanced_accuracy_cm_aux, tp, fp, tn, fn, average=average, adjusted=adjusted)


def _balanced_accuracy_cm_aux(tp, fp, tn, fn, adjusted: bool = True):
    score = (tp / (tp + fn) + tn / (tn + fp)) * 0.5
    if adjusted:
        score -= 0.5
        score *= 2
    return score


def fbeta_cm(*, tp=None, fp=None, tn=None, fn=None, average: Optional[str] = 'binary',
             zero_division: Union[str, float] = 'warn', beta: float = 1.) -> Union[float, int, np.ndarray]:
    return _calc_single_cm_metric(_fbeta_cm_aux, tp, fp, tn, fn, average=average, zero_division=zero_division,
                                  beta=beta)


def _fbeta_cm_aux(tp, fp, tn, fn, zero_division='warn', beta=1.):
    b2 = beta * beta
    return _safe_divide((1 + b2) * tp, (1 + b2) * tp + b2 * fn + fp, zero_division=zero_division, inplace=True)


def cohen_kappa_cm(*, tp=None, fp=None, tn=None, fn=None, average: Optional[str] = 'binary') \
        -> Union[float, int, np.ndarray]:
    return _calc_single_cm_metric(_cohen_kappa_cm_aux, tp, fp, tn, fn, average=average)


def _cohen_kappa_cm_aux(tp, fp, tn, fn):
    return _safe_divide(2 * (tp * tn - fp * fn), (tp + fp) * (fp + tn) + (tp + fn) * (fn + tn), zero_division=0,
                        inplace=True)


def matthews_correlation_coefficient_cm(*, tp=None, fp=None, tn=None, fn=None, average: Optional[str] = 'binary') \
        -> Union[float, int, np.ndarray]:
    return _calc_single_cm_metric(_matthews_correlation_coefficient_cm_aux, tp, fp, tn, fn, average=average)


def _matthews_correlation_coefficient_cm_aux(tp, fp, tn, fn):
    return _safe_divide(tp * tn - fp * fn, np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)), zero_division=0,
                        inplace=True)


def jaccard_cm(*, tp=None, fp=None, tn=None, fn=None, average: Optional[str] = 'binary',
               zero_division: Union[str, float] = 'warn') -> Union[float, int, np.ndarray]:
    return _calc_single_cm_metric(_jaccard_cm_aux, tp, fp, tn, fn, average=average, zero_division=zero_division)


def _jaccard_cm_aux(tp, fp, tn, fn, zero_division='warn'):
    return _safe_divide(tp, tp + fn + fp, zero_division=zero_division, inplace=False)


def hamming_loss_cm(*, tp=None, fp=None, tn=None, fn=None, average: Optional[str] = 'binary') \
        -> Union[float, int, np.ndarray]:
    return 1. - accuracy_cm(tp=tp, fp=fp, tn=tn, fn=fn, average=average, normalize=True)


f1_cm = partial(fbeta_cm, beta=1)
sensitivity_cm = partial(recall_cm, zero_division=0)
specificity_cm = partial(recall_cm, swap_pos_neg=True, zero_division=0)
positive_predictive_value_cm = partial(precision_cm, zero_division=1)
negative_predictive_value_cm = partial(precision_cm, swap_pos_neg=True, zero_division=1)


def _calc_single_cm_metric(func, tp, fp, tn, fn, average=None, weighted_zero_division=1, **kwargs):
    tp, fp, tn, fn = _to_int_arrays(tp, fp, tn, fn)
    if np.isscalar(tp):
        return func(tp, fp, tn, fn, **kwargs)
    elif average == 'binary':
        raise ValueError(_BINARY_AVERAGE_ERROR)
    elif average == 'micro':
        return func(tp.sum(), fp.sum(), tn.sum(), fn.sum(), **kwargs)
    else:
        value = func(tp, fp, tn, fn, **kwargs)
        if average == 'macro':
            return value.mean()
        elif average == 'weighted':
            weights = tp + fn
            return _safe_divide((value * weights).sum(), weights.sum(),
                                zero_division=weighted_zero_division, inplace=True)
        else:
            return value


def _to_int_array(x):
    if isinstance(x, pd.Series):
        assert x.dtype.kind in 'uib'
        return x.values
    elif isinstance(x, (int, bool)):
        return x
    else:
        assert x.dtype.kind in 'uib'
        assert x.ndim in (0, 1)
        return x


def _to_int_arrays(*args) -> tuple:
    out = tuple(_to_int_array(a) for a in args)
    assert all(np.isscalar(a) for a in out) or all(len(a) == len(out[0]) for a in out)
    return out


def _safe_divide(num, denom, zero_division: Union[str, float] = 0, inplace: bool = False):
    mask = denom == 0
    if np.any(mask):
        if zero_division == 'warn':
            zero_division = 0
        if np.isscalar(mask):
            return zero_division
        else:
            denom[mask] = 1
            if inplace and getattr(num, 'dtype', np.dtype(np.int8)).kind == 'f':
                num /= denom
            else:
                num = num / denom
            num[mask] = zero_division
            return num
    elif inplace and getattr(num, 'dtype', np.dtype(np.int8)).kind == 'f':
        num /= denom
        return num
    else:
        return num / denom


_BINARY_AVERAGE_ERROR = "Target is multilabel-indicator but average='binary'. Please choose another average setting," \
                        " one of [None, 'micro', 'macro', 'weighted']"
