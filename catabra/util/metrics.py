#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

from typing import Union, Optional, Tuple
from functools import partial
import warnings
import sys
import numpy as np
import pandas as pd
from sklearn import metrics as skl_metrics

from .bootstrapping import Bootstrapping


class _OperatorBase:

    @classmethod
    def op_name(cls) -> str:
        return cls.__name__.lstrip('_')

    @classmethod
    def get_function_name(cls, func) -> str:
        return getattr(func, '__name__', None) or repr(func)

    def __init__(self, func):
        if not callable(func):
            raise ValueError(f'Function must be callable, but found {func}')
        self._func = func

    @property
    def base_function(self):
        return self._func

    @property
    def __name__(self) -> str:
        return self.op_name() + '(' + self.get_function_name(self._func) + ')'

    def __repr__(self) -> str:
        return self.__name__

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class _micro_average(_OperatorBase):    # noqa
    # in case of multilabel problems, this implementation also works with class probabilities

    def __call__(self, y_true, y_pred, sample_weight: Optional[np.ndarray] = None, **kwargs):
        y_true, y_pred = _to_num_arrays(y_true, y_pred, rank=(1, 2))
        if y_true.ndim == 1:
            classes = np.unique(y_true)
            y_true_one_hot = y_true[..., np.newaxis] == classes
            y_pred_one_hot = y_pred[..., np.newaxis] == classes
            y_true = y_true_one_hot
            y_pred = y_pred_one_hot

        if sample_weight is not None:
            sample_weight = np.repeat(sample_weight, y_true.shape[1])

        return self._func(y_true.ravel(), y_pred.ravel(), sample_weight=sample_weight, **kwargs)


class _macro_average(_OperatorBase):    # noqa
    # in case of multilabel problems, this implementation also works with class probabilities

    def __call__(self, y_true, y_pred, **kwargs):
        y_true, y_pred = _to_num_arrays(y_true, y_pred, rank=(1, 2))
        if y_true.ndim == 1:
            classes = np.unique(y_true)
            return np.mean([self._func(y_true == c, y_pred == c, **kwargs) for c in classes])
        else:
            return np.mean([self._func(y_true[:, i], y_pred[:, i], **kwargs) for i in range(y_true.shape[1])])


class _samples_average(_OperatorBase):  # noqa
    # only defined for multilabel problems, and works for class probabilities and class indicators alike

    def __call__(self, y_true, y_pred, sample_weight: Optional[np.ndarray] = None, **kwargs):
        y_true, y_pred = _to_num_arrays(y_true, y_pred, rank=(2,))
        return np.average([self._func(y_true[i], y_pred[i], **kwargs) for i in range(y_true.shape[0])],
                          weights=sample_weight)


class _weighted_average(_OperatorBase):     # noqa
    # in case of multilabel problems, this implementation also works with class probabilities

    def __call__(self, y_true, y_pred, sample_weight: Optional[np.ndarray] = None, **kwargs):
        y_true, y_pred = _to_num_arrays(y_true, y_pred, rank=(1, 2))
        if y_true.ndim == 1:
            if sample_weight is None:
                classes, weights = np.unique(y_true, return_counts=True)
            else:
                classes = np.unique(y_true)
                weights = np.zeros((len(classes),), dtype=np.float32)
                for i, c in enumerate(classes):
                    weights[i] = ((y_true == c) * sample_weight).sum()
            return np.sum([w * self._func(y_true == c, y_pred == c, sample_weight=sample_weight, **kwargs)
                           for c, w in zip(classes, weights)]) / len(y_true)
        else:
            if sample_weight is None:
                weights = (y_true > 0).sum(axis=0)
            else:
                weights = ((y_true > 0) * sample_weight[..., np.newaxis]).sum(axis=0)
            return np.sum([w * self._func(y_true[:, i], y_pred[:, i], sample_weight=sample_weight, **kwargs)
                           for i, w in enumerate(weights)]) / weights.sum()


class _NegativeFunction(_OperatorBase):

    def __call__(self, *args, **kwargs):
        return -self._func(*args, **kwargs)

    @property
    def __name__(self) -> str:
        return '-' + self.get_function_name(self._func)

    @classmethod
    def make(cls, func):
        if isinstance(func, _NegativeFunction):
            return func.base_function
        else:
            return cls(func)


def to_score(func):
    """
    Convenience function for converting a metric into a (possibly different) metric that returns scores (i.e., higher
    values correspond to better results). That means, if the given metric returns scores already, it is returned
    unchanged. Otherwise, it is negated.
    :param func: The metric to convert, e.g., `accuracy`, `balanced_accuracy`, etc. Note that in case of classification
    metrics, both thresholded and non-thresholded metrics are accepted.
    :return: Either `func` itself or `-func`.
    """

    func_th = maybe_thresholded(func, threshold=0.5)
    y = np.array([0, 0, 1], dtype=np.float32)
    y_hat = y * 0.8 + 0.1       # some metrics (e.g., log_loss) don't accept exact 0. and 1.
    try:
        perfect = func_th(y, y_hat)
    except:     # noqa
        y = y[..., np.newaxis]
        y_hat = y_hat[..., np.newaxis]
        try:
            perfect = func_th(y, y_hat)
        except:     # noqa
            y[1] = 1.
            y[2] = 2.
            y_hat = np.eye(3, dtype=np.float32) * 0.8 + 0.1
            perfect = func_th(y, y_hat)

    worse = func_th(y, y_hat[::-1])
    if perfect < worse:
        return _NegativeFunction.make(func)
    else:
        return func


def get(name):
    """
    Retrieve a metric function given by its name.
    :param name: The name of the requested metric function. It may be of the form "`name` @ `threshold`", where `name`
    is the name of a thresholded classification metric (e.g., "accuracy") and `threshold` is the desired threshold.
    Furthermore, some synonyms are recognized as well, most notably "precision" for "positive_predictive_value" and
    "recall" for "sensitivity". `threshold` can also be the name of a thresholding strategy; see function
    `thresholded()` for details.
    :return: Metric function (callable).
    """
    if isinstance(name, str):
        s = name.split('@', 1)
        if len(s) == 1:
            name = s[0].strip()
            if name == 'precision' or name.startswith('precision_'):
                name = 'positive_predictive_value' + name[9:]
            elif name == 'recall' or name.startswith('recall_'):
                name = 'sensitivity' + name[6:]
            elif name == 'youden_index' or name.startswith('youden_index_'):
                name = 'informedness' + name[12:]
            elif name == 'mean_average_precision':
                name = 'average_precision_macro'
            return getattr(sys.modules[__name__], name)
        else:
            fn = get(s[0])
            th = s[1].strip()
            try:
                th = float(th)
            except ValueError:
                # assume `th` is the name of a thresholding strategy => pass on to `thresholded()`
                pass
            return thresholded(fn, threshold=th)
    return name


class _bootstrapped(_OperatorBase):     # noqa

    def __init__(self, func, n_repetitions: int = 100, agg='mean', seed=None, replace: bool = True,
                 size: Union[int, float] = 1., **kwargs):
        super(_bootstrapped, self).__init__(func)
        self._n_repetitions = n_repetitions
        self._agg = agg
        self._seed = seed
        self._replace = replace
        self._size = size
        self._kwargs = kwargs

    @property
    def n_repetitions(self) -> int:
        return self._n_repetitions

    @property
    def agg(self):
        return self._agg

    @property
    def seed(self):
        return self._seed

    @property
    def replace(self) -> bool:
        return self._replace

    @property
    def size(self):
        return self._size

    def __call__(self, y_true, y_hat, sample_weight=None, **kwargs):
        kwargs.update(self._kwargs)
        if kwargs:
            _func = partial(self._func, **kwargs)
        else:
            _func = self._func
        if sample_weight is None:
            _kwargs = None
        else:
            _kwargs = dict(sample_weight=sample_weight)
        bs = Bootstrapping(y_true, y_hat, kwargs=_kwargs, fn=_func, seed=self._seed, replace=self._replace,
                           size=self._size)
        bs.run(n_repetitions=self._n_repetitions)
        return bs.agg(self._agg)

    @classmethod
    def make(cls, func, n_repetitions: int = 100, agg='mean', seed=None, replace: bool = True,
             size: Union[int, float] = 1., **kwargs):
        """
        Convenience function for converting a metric into its bootstrapped version.
        :param func: The metric to convert, e.g., `roc_auc`, `accuracy`, `mean_squared_error`, etc.
        :param n_repetitions: Number of bootstrapping repetitions to perform. If 0, `func` is returned unchanged.
        :param agg: Aggregation to compute of bootstrapping results.
        :param seed: Random seed.
        :param replace: Whether to resample with replacement. If False, this does not actually correspond to
        bootstrapping.
        :param size: The size of the resampled data. If <= 1, it is multiplied with the number of samples in the given
        data. Bootstrapping normally assumes that resampled data have the same number of samples as the original data,
        so this parameter should be set to 1.
        :param kwargs: Additional keyword arguments that are passed to `func` upon application. Note that only
        arguments that do not need to be resampled can be passed here; in particular, this excludes `sample_weight`.
        :return: New metric that, when applied to `y_true` and `y_hat`, resamples the data, evaluates the metric on
        each resample, and returns som aggregation (typically average) of the results thus obtained.
        """
        if n_repetitions <= 0:
            if kwargs:
                return partial(func, **kwargs)
            else:
                return func

        return cls(func, n_repetitions=n_repetitions, agg=agg, seed=seed, replace=replace, size=size, **kwargs)


bootstrapped = _bootstrapped.make       # convenient alias


# regression
r2 = skl_metrics.r2_score
mean_absolute_error = skl_metrics.mean_absolute_error
mean_squared_error = skl_metrics.mean_squared_error
root_mean_squared_error = partial(skl_metrics.mean_squared_error, squared=False)
mean_squared_log_error = skl_metrics.mean_squared_log_error
median_absolute_error = skl_metrics.median_absolute_error
mean_absolute_percentage_error = skl_metrics.mean_absolute_percentage_error
explained_variance = skl_metrics.explained_variance_score
mean_tweedie_deviance = skl_metrics.mean_tweedie_deviance
mean_poisson_deviance = skl_metrics.mean_poisson_deviance
mean_gamma_deviance = skl_metrics.mean_gamma_deviance


def max_error(y_true, y_pred, sample_weight=None):
    return skl_metrics.max_error(y_true, y_pred)


# classification with probabilities
roc_auc = skl_metrics.roc_auc_score
roc_auc_micro = partial(roc_auc, average='micro')
roc_auc_macro = partial(roc_auc, average='macro')
roc_auc_samples = partial(roc_auc, average='samples')
roc_auc_weighted = partial(roc_auc, average='weighted')
roc_auc_ovr = partial(roc_auc, multi_class='ovr')
roc_auc_ovo = partial(roc_auc, multi_class='ovo')
roc_auc_ovr_weighted = partial(roc_auc, multi_class='ovr', average='weighted')
roc_auc_ovo_weighted = partial(roc_auc, multi_class='ovo', average='weighted')
average_precision = skl_metrics.average_precision_score
average_precision_micro = partial(average_precision, average='micro')
average_precision_macro = partial(average_precision, average='macro')
average_precision_samples = partial(average_precision, average='samples')
average_precision_weighted = partial(average_precision, average='weighted')
brier_loss = skl_metrics.brier_score_loss
brier_loss_micro = _micro_average(brier_loss)
brier_loss_macro = _macro_average(brier_loss)
brier_loss_samples = _samples_average(brier_loss)
brier_loss_weighted = _weighted_average(brier_loss)
hinge_loss = skl_metrics.hinge_loss
hinge_loss_micro = _micro_average(hinge_loss)
hinge_loss_macro = _macro_average(hinge_loss)
hinge_loss_samples = _samples_average(hinge_loss)
hinge_loss_weighted = _weighted_average(hinge_loss)
log_loss = skl_metrics.log_loss
log_loss_micro = _micro_average(log_loss)
log_loss_macro = _macro_average(log_loss)
log_loss_samples = _samples_average(log_loss)
log_loss_weighted = _weighted_average(log_loss)


def pr_auc(y_true, y_score, **kwargs) -> float:
    precision, recall, _ = skl_metrics.precision_recall_curve(y_true, y_score, **kwargs)
    return skl_metrics.auc(recall, precision)


pr_auc_micro = _micro_average(pr_auc)
pr_auc_macro = _macro_average(pr_auc)
pr_auc_samples = _samples_average(pr_auc)
pr_auc_weighted = _weighted_average(pr_auc)


def balance_score_threshold(y_true, y_score, sample_weight: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Compute the balance score and -threshold of a binary classification problem.
    :param y_true: Ground truth, with 0 representing the negative class and 1 representing the positive class. Must not
    contain NaN.
    :param y_score: Predicted scores, i.e., the higher a score the more confident the model is that the sample belongs
    to the positive class. Range is arbitrary.
    :param sample_weight: Sample weights.
    :return: Pair `(balance_score, balance_threshold)`, where `balance_threshold` is the decision threshold that
    minimizes the difference between sensitivity and specificity, i.e., it is defined as

        arg min_t |sensitivity(y_true, y_score >= t) - specificity(y_true, y_score >= t)|

    `balance_score` is the corresponding sensitivity value, which by definition is approximately equal to specificity
    and can furthermore be shown to be approximately equal to accuracy and balanced accuracy, too.
    """
    if len(y_true) != len(y_score):
        raise ValueError('Found input variables with inconsistent numbers of samples: %r' % [len(y_true), len(y_score)])
    elif sample_weight is not None and len(y_true) != len(sample_weight):
        raise ValueError(
            'Length of `sample_weight` differs from length of `y_true`: %r' % [len(y_true), len(sample_weight)]
        )
    elif len(y_true) == 0:
        return 0., 0.
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_score, pd.Series):
        y_score = y_score.values
    s = pd.DataFrame(data=dict(p=(y_true > 0).astype(np.float32), n=(y_true < 1).astype(np.float32)), index=y_score)
    if sample_weight is not None:
        s['p'] *= sample_weight
        s['n'] *= sample_weight
    s = s.groupby(level=0).sum()
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


def balance_score(y_true, y_score, sample_weight: Optional[np.ndarray] = None) -> float:
    return balance_score_threshold(y_true, y_score, sample_weight=sample_weight)[0]


def balance_threshold(y_true, y_score, sample_weight: Optional[np.ndarray] = None) -> float:
    return balance_score_threshold(y_true, y_score, sample_weight=sample_weight)[1]


def prevalence_score_threshold(y_true, y_score, sample_weight: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Compute the prevalence score and -threshold of a binary classification problem.
    :param y_true: Ground truth, with 0 representing the negative class and 1 representing the positive class. Must not
    contain NaN.
    :param y_score: Predicted scores, i.e., the higher a score the more confident the model is that the sample belongs
    to the positive class. Range is arbitrary.
    :param sample_weight: Sample weights, optional.
    :return: Pair `(prevalence_score, prevalence_threshold)`, where `prevalence_threshold` is the decision threshold
    that minimizes the difference between the number of positive samples in `y_true` (`m`) and the number of predicted
    positives. In other words, the threshold is set to the `m`-th largest value in `y_score`.
    If `sample_weight` is given, the threshold minimizes the difference between the total weight of all positive
    samples and the total weight of all samples predicted positive.
    `prevalence_score` is the corresponding sensitivity value, which can be shown to be approximately equal to positive
    predictive value and F1.
    """
    th = prevalence_threshold(y_true, y_score, sample_weight=sample_weight)
    return sensitivity(y_true, y_score >= th, sample_weight=sample_weight), th


def prevalence_score(y_true, y_score, sample_weight: Optional[np.ndarray] = None) -> float:
    return prevalence_score_threshold(y_true, y_score, sample_weight=sample_weight)[0]


def prevalence_threshold(y_true, y_score, sample_weight: Optional[np.ndarray] = None) -> float:
    if len(y_true) != len(y_score):
        raise ValueError('Found input variables with inconsistent numbers of samples: %r' % [len(y_true), len(y_score)])
    elif sample_weight is not None and len(y_true) != len(sample_weight):
        raise ValueError(
            'Length of `sample_weight` differs from length of `y_true`: %r' % [len(y_true), len(sample_weight)]
        )
    elif len(y_true) == 0:
        return 0.
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_score, pd.Series):
        y_score = y_score.values
    s = pd.DataFrame(data=dict(p=(y_true > 0).astype(np.float32), w=1), index=y_score)
    if sample_weight is not None:
        s['p'] *= sample_weight
        s['w'] *= sample_weight
    s = s.groupby(level=0).sum()
    s.sort_index(ascending=False, inplace=True)
    i = np.argmin(np.abs(s['w'].cumsum().values - s['p'].sum()))
    th = s.index[i]
    if i + 1 < len(s):  # take midpoint of `i`-th and nearest *smaller* threshold
        th += s.index[i + 1]
        th *= 0.5
    return th


def zero_one_threshold(y_true, y_score, sample_weight: Optional[np.ndarray] = None,
                       specificity_weight: float = 1.) -> float:
    """
    Compute the threshold corresponding to the (0,1)-criterion [1] of a binary classification problem.
    Although a popular strategy for selecting decision thresholds, [1] advocates maximizing informedness (aka Youden
    index) instead, which is equivalent to maximizing balanced accuracy.

    [1] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1444894/

    :param y_true: Ground truth, with 0 representing the negative class and 1 representing the positive class. Must not
    contain NaN.
    :param y_score: Predicted scores, i.e., the higher a score the more confident the model is that the sample belongs
    to the positive class. Range is arbitrary.
    :param sample_weight: Sample weights, optional.
    :param specificity_weight: The relative weight of specificity wrt. sensitivity. 1 means that sensitivity and
    specificity are weighted equally, a value < 1 means that sensitivity is weighted stronger than specificity, and
    a value > 1 means that specificity is weighted stronger than sensitivity. See the formula below for details.
    :return: Decision threshold that minimizes the Euclidean distance between the point `(0, 1)` and the point
    `(1 - specificity, sensitivity)`, i.e.,

        arg min_t (1 - sensitivity(y_true, y_score >= t)) ** 2 +
                  specificity_weight * (1 - specificity(y_true, y_score >= t)) ** 2
    """
    if len(y_true) != len(y_score):
        raise ValueError('Found input variables with inconsistent numbers of samples: %r' % [len(y_true), len(y_score)])
    elif sample_weight is not None and len(y_true) != len(sample_weight):
        raise ValueError(
            'Length of `sample_weight` differs from length of `y_true`: %r' % [len(y_true), len(sample_weight)]
        )
    elif len(y_true) == 0:
        return 0.
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_score, pd.Series):
        y_score = y_score.values
    s = pd.DataFrame(data=dict(p=(y_true > 0).astype(np.float32), n=(y_true < 1).astype(np.float32)), index=y_score)
    if sample_weight is not None:
        s['p'] *= sample_weight
        s['n'] *= sample_weight
    s = s.groupby(level=0).sum()
    s.sort_index(ascending=False, inplace=True)
    n_pos = s['p'].sum()
    n_neg = s['n'].sum()
    fn = n_pos - s['p'].cumsum().values
    fp = s['n'].cumsum().values
    obj = (fn * n_neg) ** 2 + specificity_weight * (fp * n_pos) ** 2
    i = np.argmin(obj)
    th = s.index[i]
    if i + 1 < len(s):  # take midpoint of `i`-th and nearest *smaller* threshold
        th += s.index[i + 1]
        th *= 0.5
    return th


def argmax_score_threshold(func, y_true, y_score, sample_weight: Optional[np.ndarray] = None,
                           discretize=100, **kwargs) -> Tuple[float, float]:
    """
    Compute the decision threshold that maximizes a given binary classification metric or callable.
    Since in most built-in classification metrics larger values indicate better results, there is no analogous
    `argmin_score_threshold()`.
    :param func: The metric or function ot maximize. If a string, function `get()` is called on it.
    :param y_true: Ground truth, with 0 representing the negative class and 1 representing the positive class. Must not
    contain NaN.
    :param y_score: Predicted scores, i.e., the higher a score the more confident the model is that the sample belongs
    to the positive class. Range is arbitrary.
    :param sample_weight: Sample weights, optional.
    :param discretize: Discretization steps for limiting the number of calls to `func`. If None, no discretization
    happens, i.e., all unique values in `y_score` are tried.
    :param kwargs: Additional keyword arguments passed to `func`.
    :return: Pair `(score, threshold)`, where `threshold` is the decision threshold that maximizes `func`, i.e.,

        arg max_t func(y_true, y_score >= t)

    `score` is the corresponding value of `func`.
    """
    if isinstance(func, str):
        func = get(func)
    if isinstance(y_score, pd.Series):
        y_score = y_score.values
    opt = -np.inf
    th = 0.
    if discretize is None:
        thresholds = np.unique(y_score)
    else:
        thresholds = get_thresholds(y_score, n_max=discretize, sample_weight=sample_weight)
    for t in thresholds:
        res = func(y_true, y_score >= t, sample_weight=sample_weight, **kwargs)
        if res > opt:
            opt = res
            th = t
    return opt, th


def argmax_score(func, y_true, y_score, **kwargs) -> float:
    return argmax_score_threshold(func, y_true, y_score, **kwargs)[0]


def argmax_threshold(func, y_true, y_score, **kwargs) -> float:
    return argmax_score_threshold(func, y_true, y_score, **kwargs)[1]


def get_thresholding_strategy(name: str):
    """
    Retrieve a thresholding strategy for binary classification, given by its name.
    :param name: The name of the thresholding strategy, like "balance", "prevalence" or "zero_one".
    :return: Thresholding strategy (callable) that can be applied to `y_true`, `y_score` and `sample_weight`, and that
    returns a single scalar threshold.
    """
    if name == 'balance':
        return balance_threshold
    elif name == 'prevalence':
        return prevalence_threshold
    elif name == 'zero_one':
        return zero_one_threshold
    elif name.startswith('zero_one(') and name[-1] == ')':
        return partial(zero_one_threshold, specificity_weight=float(name[9:-1]))
    elif name.startswith('argmax '):
        return partial(argmax_threshold, name[7:])
    else:
        raise ValueError(f'Unknown thresholding strategy "{name}".'
                         ' Choose one of "balance", "prevalence", "zero_one" or "argmax <metric>"')


def calibration_curve(y_true: np.ndarray, y_score: np.ndarray, sample_weight: Optional[np.ndarray] = None,
                      thresholds: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the calibration curve of a binary classification problem. The predicated class probabilities are binned and,
    for each bin, the fraction of positive samples is determined. These fractions can then be plotted against the
    midpoints of the respective bins. Ideally, the resulting curve will be monotonic increasing.
    :param y_true: Ground truth, array of shape `(n,)` with values among 0 and 1. Values must not be NaN.
    :param y_score: Predicated probabilities of the positive class, array of shape `(n,)` with arbitrary non-NaN values;
    in particular, the values do not necessarily need to correspond to probabilities or confidences.
    :param sample_weight: Sample weight.
    :param thresholds: The thresholds used for binning `y_score`. If None, suitable thresholds are determined
    automatically.
    :return: Pair `(fractions, thresholds)`, where `thresholds` is the array of thresholds of shape `(m,)`, and
    `fractions` is the corresponding array of fractions of positive samples in each bin, of shape `(m - 1,)`. Note that
    the `i`-th bin corresponds to the half-open interval `[thresholds[i], thresholds[i + 1])` if `i < m - 2`, and to
    the closed interval `[thresholds[i], thresholds[i + 1]]` otherwise (in other words: the last bin is closed).
    """
    assert y_true.shape == y_score.shape
    if thresholds is None:
        thresholds = get_thresholds(y_score, n_max=40, add_half_one=False, sample_weight=sample_weight)
    if len(thresholds) < 2:
        return np.empty((0,), dtype=np.float32), thresholds
    fractions = \
        np.array([y_true[(t1 <= y_score) & (y_score < t2)].mean() for t1, t2 in zip(thresholds[:-1], thresholds[1:])])
    fractions[-1] = y_true[(thresholds[-2] <= y_score) & (y_score <= thresholds[-1])].mean()
    return fractions, np.array(thresholds)


def roc_pr_curve(y_true: np.ndarray, y_score: np.ndarray, *, pos_label: Union[int, str, None] = None,
                 sample_weight: Optional[np.ndarray] = None, drop_intermediate: bool = True) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function for computing ROC- and precision-recall curves simultaneously, with only one call to
    function `_binary_clf_curve()`.
    :param y_true: Same as in `sklearn.metrics.roc_curve()` and `sklearn.metrics.precision_recall_curve()`.
    :param y_score: Same as in `sklearn.metrics.roc_curve()` and `sklearn.metrics.precision_recall_curve()`.
    :param pos_label: Same as in `sklearn.metrics.roc_curve()` and `sklearn.metrics.precision_recall_curve()`.
    :param sample_weight: Same as in `sklearn.metrics.roc_curve()` and `sklearn.metrics.precision_recall_curve()`.
    :param drop_intermediate: Same as in `sklearn.metrics.roc_curve()`.
    :return: 6-tuple `(fpr, tpr, thresholds_roc, precision, recall, thresholds_pr)`, i.e., the concatenation of the
    return values of functions `sklearn.metrics.roc_curve()` and `sklearn.metrics.precision_recall_curve()`.
    """

    fps, tps, thresholds = skl_metrics._ranking._binary_clf_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight
    )

    # ROC
    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(
            np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True]
        )[0]
        fps_roc = fps[optimal_idxs]
        tps_roc = tps[optimal_idxs]
        thresholds_roc = thresholds[optimal_idxs]
    else:
        fps_roc = fps
        tps_roc = tps
        thresholds_roc = thresholds

    # Add an extra threshold position to make sure that the curve starts at (0, 0)
    tps_roc = np.r_[0, tps_roc]
    fps_roc = np.r_[0, fps_roc]

    if fps[-1] <= 0:
        warnings.warn(
            "No negative samples in y_true, false positive value should be meaningless",
            skl_metrics._ranking.UndefinedMetricWarning,
        )
        fpr = np.repeat(np.nan, fps_roc.shape)
    else:
        fpr = fps_roc / fps[-1]

    if tps[-1] <= 0:
        warnings.warn(
            "No positive samples in y_true,"
            " true positive value should be meaningless and recall is set to 1 for all thresholds",
            skl_metrics._ranking.UndefinedMetricWarning,
        )
        tpr = np.repeat(np.nan, tps_roc.shape)
        recall = np.ones_like(tps)
    else:
        tpr = tps_roc / tps[-1]
        recall = tps / tps[-1]

    # PR
    ps = tps + fps
    precision = np.divide(tps, ps, where=(ps != 0))

    # stop when full recall attained and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    return fpr, tpr, np.r_[thresholds_roc[0] + 1, thresholds_roc], \
        np.hstack((precision[sl], 1)), np.hstack((recall[sl], 0)), thresholds[sl]


def get_thresholds(y: np.ndarray, n_max: int = 100, add_half_one: Optional[bool] = None,
                   ensure: Optional[list] = None, sample_weight: Optional[np.ndarray] = None) -> list:
    n = min(np.isfinite(y).sum(), n_max)
    if n == 0:
        thresholds = set()
    elif n == 1:
        thresholds = {np.nanmin(y)}
    elif sample_weight is None:
        thresholds = np.sort(y[np.isfinite(y)])
        thresholds = set(thresholds[np.linspace(0, len(thresholds) - 1, n).round().astype(np.int32)])
    else:
        mask = np.isfinite(y)
        y0 = y[mask]
        sample_weight = sample_weight[mask]
        indices = np.argsort(y0)
        y0 = y0[indices]
        sample_weight = sample_weight[indices]
        s = sample_weight.cumsum() / np.maximum(sample_weight.sum(), 1e-8)
        np.clip(s, 0., 1., out=s)
        s *= n
        s = np.ceil(s).astype(np.int32)
        indices = np.zeros((n,), dtype=np.int32)
        for i in range(1, n):
            aux = np.where(s == i)[0]
            if len(aux) > 0:
                indices[i] = aux[-1]
        thresholds = set(y0[indices])
    if add_half_one is True or (add_half_one is None and not (y < 0).any() and not (1 < y).any()):
        thresholds.update({0.5, 1.})
    if ensure is not None:
        thresholds.update(ensure)
    thresholds = list(thresholds)
    if len(thresholds) == 0:
        return [0, 1]
    elif len(thresholds) == 1:
        if 0 <= thresholds[0] <= 1:
            thresholds = [0, 1]
        else:
            thresholds = [np.floor(thresholds[0]), np.floor(thresholds[0]) + 1]
    else:
        thresholds.sort()
    return thresholds


def multiclass_proba_to_pred(y: np.ndarray) -> np.ndarray:
    """
    Translate multiclass class probabilities into actual predictions, by returning the class with the highest
    probability. If two or more classes have the same highest probabilities, the last one is returned. This behavior is
    consistent with binary classification problems, where the positive class is returned if both classes have equal
    probabilities and the default threshold of 0.5 is used.
    :param y: Class probabilities, of shape `(n_classes,)` or `(n_samples, n_classes)`. The values of `y` can be
    arbitrary, they don't need to be between 0 and 1. `n_classes` must be >= 1.
    :return: Predicted class indices, either single integer or array of shape `(n_samples,)`.
    """
    if y.ndim == 1:
        return len(y) - np.argmax(y[::-1]) - 1
    else:
        assert y.ndim == 2
        return y.shape[1] - np.argmax(y[:, ::-1], axis=1) - 1


class thresholded(_OperatorBase):
    """
    Convenience class for converting a classification metric that can only be applied to class predictions into a
    metric that can be applied to probabilities. This proceeds by specifying a fixed decision threshold.
    :param func: The metric to convert, e.g., `accuracy`, `balanced_accuracy`, etc.
    :param threshold: The decision threshold. In binary classification this can also be the name of a thresholding
    strategy that is accepted by function `get_thresholding_strategy()`.
    :param kwargs: Additional keyword arguments that are passed to `func` upon application.
    :return: New metric that, when applied to `y_true` and `y_score`, returns `func(y_true, y_score >= threshold)`
    in case of binary- or multilabel classification, and `func(y_true, multiclass_proba_to_pred(y_score))` in case
    of multiclass classification.
    """

    def __init__(self, func, threshold: Union[float, str] = 0.5, **kwargs):
        if isinstance(threshold, str):
            if threshold == 'argmax':
                threshold = partial(argmax_threshold, func)
            else:
                threshold = get_thresholding_strategy(threshold)
        elif not isinstance(threshold, (int, float)):
            raise ValueError(f'Threshold must be a string or float, but found {threshold}')
        super(thresholded, self).__init__(func)
        self._threshold = threshold
        self._kwargs = kwargs

    @property
    def threshold(self):
        return self._threshold

    def __call__(self, y_true, y_score, **kwargs):
        kwargs.update(self._kwargs)
        if y_score.ndim == 2 and y_score.shape[1] > 1 and (y_true.ndim == 1 or y_true.shape[1] == 1):
            # multiclass classification => `threshold` is not needed
            return self._func(y_true, multiclass_proba_to_pred(y_score), **kwargs)
        elif isinstance(self._threshold, (int, float)):
            # binary- or multilabel classification
            return self._func(y_true, y_score >= self._threshold, **kwargs)
        else:
            t = self._threshold(y_true, y_score, sample_weight=kwargs.get('sample_weight'))
            return self._func(y_true, y_score >= t, **kwargs)

    @classmethod
    def make(cls, func, threshold: Union[float, str] = 0.5, **kwargs):
        """
        Convenience function for converting a classification metric into its "thresholded" version IF NECESSARY.
        That means, if the given metric can be applied to class probabilities, it is returned unchanged. Otherwise,
        `thresholded(func, threshold)` is returned.
        :param func: The metric to convert, e.g., `accuracy`, `balanced_accuracy`, etc.
        :param threshold: The decision threshold.
        :param kwargs: Additional keyword arguments that shall be passed to `func` upon application.
        :return: Either `func` itself or `thresholded(func, threshold)`.
        """

        if not isinstance(func, cls):
            try:
                # apply `func` to see whether it can be applied to probabilities
                # apply it to one positive and one negative sample, because some metrics like `roc_auc` raise an
                # exception if only one class is present
                func(np.arange(2, dtype=np.float32), 0.3 * np.arange(1, 3, dtype=np.float32), **kwargs)
            except:  # noqa
                try:
                    # check whether `func` can be applied to multiclass problems
                    func(np.arange(3, dtype=np.float32),
                         np.array([[0.1, 0.3, 0.6], [0.9, 0.1, 0], [0.2, 0.7, 0.1]], dtype=np.float32), **kwargs)
                except:  # noqa
                    return cls(func, threshold=threshold, **kwargs)

        if kwargs:
            return partial(func, **kwargs)
        else:
            return func


maybe_thresholded = thresholded.make


# classification with thresholds
confusion_matrix = skl_metrics.confusion_matrix
precision_recall_fscore_support = skl_metrics.precision_recall_fscore_support
accuracy = skl_metrics.accuracy_score
accuracy_micro = _micro_average(accuracy)
accuracy_macro = _macro_average(accuracy)
accuracy_samples = _samples_average(accuracy)
accuracy_weighted = _weighted_average(accuracy)
balanced_accuracy = skl_metrics.balanced_accuracy_score
balanced_accuracy_micro = _micro_average(balanced_accuracy)
balanced_accuracy_macro = _macro_average(balanced_accuracy)
balanced_accuracy_samples = _samples_average(balanced_accuracy)
balanced_accuracy_weighted = _weighted_average(balanced_accuracy)
f1 = skl_metrics.f1_score
f1_micro = partial(f1, average='micro')
f1_macro = partial(f1, average='macro')
f1_samples = partial(f1, average='samples')
f1_weighted = partial(f1, average='weighted')
sensitivity = partial(skl_metrics.recall_score, zero_division=0)
sensitivity_micro = partial(sensitivity, average='micro', pos_label=None)
sensitivity_macro = partial(sensitivity, average='macro', pos_label=None)
sensitivity_samples = partial(sensitivity, average='samples', pos_label=None)
sensitivity_weighted = partial(sensitivity, average='weighted', pos_label=None)
specificity = partial(skl_metrics.recall_score, pos_label=0, zero_division=0)
specificity_micro = _micro_average(specificity)   # cannot use built-in version, because `pos_label=0` would be ignored
specificity_macro = _macro_average(specificity)
specificity_samples = _samples_average(specificity)
specificity_weighted = _weighted_average(specificity)
positive_predictive_value = partial(skl_metrics.precision_score, zero_division=1)
positive_predictive_value_micro = partial(positive_predictive_value, average='micro')
positive_predictive_value_macro = partial(positive_predictive_value, average='macro')
positive_predictive_value_samples = partial(positive_predictive_value, average='samples')
positive_predictive_value_weighted = partial(positive_predictive_value, average='weighted')
negative_predictive_value = partial(skl_metrics.precision_score, pos_label=0, zero_division=1)
negative_predictive_value_micro = _micro_average(negative_predictive_value)     # see comment at `specificity_micro`
negative_predictive_value_macro = _macro_average(negative_predictive_value)
negative_predictive_value_samples = _samples_average(negative_predictive_value)
negative_predictive_value_weighted = _weighted_average(negative_predictive_value)
cohen_kappa = skl_metrics.cohen_kappa_score
cohen_kappa_micro = _micro_average(cohen_kappa)
cohen_kappa_macro = _macro_average(cohen_kappa)
cohen_kappa_samples = _samples_average(cohen_kappa)
cohen_kappa_weighted = _weighted_average(cohen_kappa)
matthews_correlation_coefficient = skl_metrics.matthews_corrcoef
matthews_correlation_coefficient_micro = _micro_average(matthews_correlation_coefficient)
matthews_correlation_coefficient_macro = _macro_average(matthews_correlation_coefficient)
matthews_correlation_coefficient_samples = _samples_average(matthews_correlation_coefficient)
matthews_correlation_coefficient_weighted = _weighted_average(matthews_correlation_coefficient)
jaccard = partial(skl_metrics.jaccard_score, zero_division=1)
jaccard_micro = partial(jaccard, average='micro')
jaccard_macro = partial(jaccard, average='macro')
jaccard_samples = partial(jaccard, average='samples')
jaccard_weighted = partial(jaccard, average='weighted')
hamming_loss = skl_metrics.hamming_loss
hamming_loss_micro = _micro_average(hamming_loss)
hamming_loss_macro = _macro_average(hamming_loss)
hamming_loss_samples = _samples_average(hamming_loss)
hamming_loss_weighted = _weighted_average(hamming_loss)


def informedness(y_true, y_pred, *, average='binary', sample_weight=None):
    """
    Informedness (aka Youden index or Youden's J statistic) is the sum of sensitivity and specificity, minus 1.
    """
    # Don't define informedness in terms of balanced accuracy, because balanced accuracy is defined for multiclass
    # problems but informedness is not (at least, not analogous to balanced accuracy).
    sens = sensitivity(y_true, y_pred, sample_weight=sample_weight, average=average)
    spec = specificity(y_true, y_pred, sample_weight=sample_weight, average=average)
    return sens + spec - 1


def markedness(y_true, y_pred, *, average='binary', sample_weight=None):
    """
    Markedness is the sum of positive- and negative predictive value, minus 1.
    """
    ppv = positive_predictive_value(y_true, y_pred, sample_weight=sample_weight, average=average)
    npv = negative_predictive_value(y_true, y_pred, sample_weight=sample_weight, average=average)
    return ppv + npv - 1


informedness_micro = _micro_average(informedness)
informedness_macro = _macro_average(informedness)
informedness_samples = _samples_average(informedness)
informedness_weighted = _weighted_average(informedness)
markedness_micro = _micro_average(markedness)
markedness_macro = _macro_average(markedness)
markedness_samples = _samples_average(markedness)
markedness_weighted = _weighted_average(markedness)


def precision_recall_fscore_support_cm(*, tp=None, fp=None, tn=None, fn=None, beta: float = 1.0,
                                       average: Optional[str] = None, zero_division: Union[float, str] = 'warn'):
    tp, fp, tn, fn = _to_num_arrays(tp, fp, tn, fn)
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
    ATTENTION! In the multilabel case, this implementation actually corresponds to `accuracy_micro` etc.
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
    """
    Calculate accuracy from a confusion matrix.
    ATTENTION! In the multilabel case, this implementation actually corresponds to `balanced_accuracy_micro` etc.
    """
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


def informedness_cm(*, tp=None, fp=None, tn=None, fn=None, average: Optional[str] = 'binary'):
    return balanced_accuracy_cm(tp=tp, fp=fp, tn=tn, fn=fn, average=average, adjusted=True)


def markedness_cm(*, tp=None, fp=None, tn=None, fn=None, average: Optional[str] = 'binary'):
    ppv = positive_predictive_value_cm(tp=tp, fp=fp, tn=tn, fn=fn, average=average)
    npv = negative_predictive_value_cm(tp=tp, fp=fp, tn=tn, fn=fn, average=average)
    return ppv + npv - 1


def _calc_single_cm_metric(func, tp, fp, tn, fn, average=None, weighted_zero_division=1, **kwargs):
    tp, fp, tn, fn = _to_num_arrays(tp, fp, tn, fn)
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


def _to_num_array(x, rank=(0, 1)):
    if isinstance(x, pd.Series):
        assert x.dtype.kind in 'uibf'
        assert 1 in rank
        return x.values
    elif isinstance(x, (int, float, bool)):
        assert 0 in rank
        return x
    else:
        assert x.dtype.kind in 'uibf'
        assert x.ndim in rank
        return x


def _to_num_arrays(*args, rank=(0, 1)) -> tuple:
    out = tuple(_to_num_array(a, rank=rank) for a in args)
    scalar = len(out) == 0 or np.isscalar(out[0])
    assert all(np.isscalar(a) == scalar for a in out) and (scalar or all(a.shape == out[0].shape for a in out))
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
