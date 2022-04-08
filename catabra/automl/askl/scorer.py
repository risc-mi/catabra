from functools import partial
import sklearn
from autosklearn import metrics


# scorers not yet predefined in austosklearn
_EXTRA_SCORERS = dict(
    # classification
    brier=metrics.make_scorer('brier', sklearn.metrics.brier_score_loss, greater_is_better=False, needs_threshold=True),
    jaccard=metrics.make_scorer('jaccard', partial(sklearn.metrics.jaccard_score, pos_label=None, zero_division=0)),
    jaccard_micro=metrics.make_scorer('jaccard_micro', partial(sklearn.metrics.jaccard_score, pos_label=None, average='micro', zero_division=0)),
    jaccard_macro=metrics.make_scorer('jaccard_macro', partial(sklearn.metrics.jaccard_score, pos_label=None, average='macro', zero_division=0)),
    jaccard_samples=metrics.make_scorer('jaccard_samples', partial(sklearn.metrics.jaccard_score, pos_label=None, average='samples', zero_division=0)),
    jaccard_weighted=metrics.make_scorer('jaccard_weighted', partial(sklearn.metrics.jaccard_score, pos_label=None, average='weighted', zero_division=0)),
    roc_auc_ovr=metrics.make_scorer('roc_auc_ovr', partial(sklearn.metrics.roc_auc_score, multi_class='ovr'), needs_threshold=True),
    roc_auc_ovo=metrics.make_scorer('roc_auc_ovo', partial(sklearn.metrics.roc_auc_score, multi_class='ovo'), needs_threshold=True),
    roc_auc_ovr_weighted=metrics.make_scorer('roc_auc_ovr_weighted', partial(sklearn.metrics.roc_auc_score, multi_class='ovr', average='weighted'), needs_threshold=True),
    roc_auc_ovo_weighted=metrics.make_scorer('roc_auc_ovo_weighted', partial(sklearn.metrics.roc_auc_score, multi_class='ovo', average='weighted'), needs_threshold=True),
    positive_predictive_value=metrics.make_scorer('positive_predictive_value', partial(sklearn.metrics.precision_score, zero_division=0)),
    negative_predictive_value=metrics.make_scorer('negative_predictive_value', partial(sklearn.metrics.precision_score, pos_label=0, zero_division=0)),
    sensitivity=metrics.make_scorer('sensitivity', partial(sklearn.metrics.recall_score, zero_division=0)),
    specificity=metrics.make_scorer('specificity', partial(sklearn.metrics.recall_score, pos_label=0, zero_division=0)),

    # regression
    explained_variance=metrics.make_scorer('explained_variance', sklearn.metrics.explained_variance_score),
    mean_absolute_percentage_error=metrics.make_scorer('mean_absolute_percentage_error', sklearn.metrics.mean_absolute_percentage_error, greater_is_better=False, optimum=0, worst_possible_result=metrics.MAXINT)
)

# aliases with prefix "neg_"
for _n in ('mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error', 'mean_squared_log_error',
           'median_absolute_error', 'mean_absolute_percentage_error'):
    _EXTRA_SCORERS['neg_' + _n] = _EXTRA_SCORERS.get(_n) or getattr(metrics, _n)


def get_scorer(name: str) -> metrics.Scorer:
    """
    Get scorer by name.
    :param name: Name of scorer.
    :return: Scorer object, for usage in autosklearn pipelines.
    """
    return _EXTRA_SCORERS.get(name) or getattr(metrics, name)
