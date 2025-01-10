#  Copyright (c) 2022-2025. RISC Software GmbH.
#  All rights reserved.

from autosklearn import metrics
from catabra_lib import metrics as um

# scorers not yet predefined in austosklearn
_EXTRA_SCORERS = dict(
    # regression
    explained_variance=metrics.make_scorer('explained_variance', um.explained_variance),
    mean_absolute_percentage_error=metrics.make_scorer(
        'mean_absolute_percentage_error',
        um.mean_absolute_percentage_error,
        greater_is_better=False,
        optimum=0,
        worst_possible_result=metrics.MAXINT
    )
)

for _m in ('brier_loss', 'hinge_loss', 'log_loss', 'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo',
           'average_precision', 'pr_auc'):
    _greater_is_better = not _m.endswith('_loss')
    for _s in ('', '_micro', '_macro', '_samples', '_weighted'):
        _n = _m + _s
        _f = getattr(um, _n, None)
        if _f is not None and getattr(metrics, _n, None) is None:
            _EXTRA_SCORERS[_n] = metrics.make_scorer(_n, _f, needs_threshold=True, greater_is_better=_greater_is_better,
                                                     worst_possible_result=0 if _greater_is_better else metrics.MAXINT,
                                                     optimum=1 if _greater_is_better else 0)

for _m in ('accuracy', 'balanced_accuracy', 'f1', 'sensitivity', 'specificity', 'positive_predictive_value',
           'negative_predictive_value', 'cohen_kappa', 'matthews_correlation_coefficient', 'jaccard', 'hamming_loss'):
    _greater_is_better = not _m.endswith('_loss')
    for _s in ('', '_micro', '_macro', '_samples', '_weighted'):
        _n = _m + _s
        _f = getattr(um, _n, None)
        if _f is not None and getattr(metrics, _n, None) is None:
            _EXTRA_SCORERS[_n] = metrics.make_scorer(_n, _f, greater_is_better=_greater_is_better,
                                                     optimum=1 if _greater_is_better else 0)


def get_scorer(name: str) -> metrics.Scorer:
    """
    Get scorer by name.

    Parameters
    ----------
    name: str
        Name of scorer.

    Returns
    -------
    metrics.Scorer
        Scorer object, for usage in autosklearn pipelines.
    """
    return _EXTRA_SCORERS.get(name) or getattr(metrics, name)
