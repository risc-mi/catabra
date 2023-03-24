#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed

from catabra.automl.fitted_ensemble import FittedEnsemble
from catabra.explanation.base import EnsembleExplainer
from catabra.util import metrics
from catabra.util.logging import progress_bar


class PermutationEnsembleExplainer(EnsembleExplainer):

    def __init__(self, ensemble: FittedEnsemble = None, config: Optional[dict] = None,
                 feature_names: Optional[list] = None, target_names: Optional[list] = None,
                 x: Optional[pd.DataFrame] = None, y: Optional[pd.DataFrame] = None, params=None,
                 n_repetitions: int = 5, seed=None, metric=None):
        super(PermutationEnsembleExplainer, self).__init__(
            ensemble=ensemble,
            config=config,
            feature_names=feature_names,
            target_names=target_names,
            x=x,
            y=y,
            params=params
        )
        self._ensemble = ensemble
        self._n_repetitions = n_repetitions
        self._rng = np.random.RandomState(seed)

        if metric is None and config is not None:
            metric = config.get(self._ensemble.task + '_metrics')
        if metric is None:
            raise ValueError('No target metric specified.')
        elif not isinstance(metric, (list, set)):
            metric = [metric]
        self._scorers = []
        self._metric_names = []
        for m in metric:
            if isinstance(m, str):
                self._scorers.append(metrics.to_score(metrics.maybe_thresholded(metrics.get(m))))
                self._metric_names.append(m)
            else:
                self._scorers.append(metrics.to_score(metrics.maybe_thresholded(m)))
                self._metric_names.append(m.__name__)

    @property
    def name(self) -> str:
        return 'permutation'

    @property
    def behavior(self) -> dict:
        return dict(supports_local=False, requires_y=True, global_accepts_x=True,
                    global_requires_x=True, global_is_mean_of_local=False)

    @property
    def params_(self) -> dict:
        return {}

    def explain(self, x: pd.DataFrame, y: Optional[pd.DataFrame] = None, jobs: int = 1,
                batch_size: Optional[int] = None, model_id=None, mapping: Optional[Dict[str, List[str]]] = None,
                show_progress: bool = False) -> dict:
        raise RuntimeError(f'{self.__class__.__name__} does not support local explanations.')

    def explain_global(self, x: Optional[pd.DataFrame] = None, y: Optional[pd.DataFrame] = None,
                       sample_weight: Optional[np.ndarray] = None, jobs: int = 1, batch_size: Optional[int] = None,
                       model_id=None, mapping: Optional[Dict[str, List[str]]] = None,
                       show_progress: bool = False) -> dict:
        if x is None:
            raise ValueError(f'{self.__class__.__name__} requires samples for global explanations.')
        if y is None:
            raise ValueError(f'{self.__class__.__name__} requires labels for global explanations.')
        if model_id is None:
            keys = None
        elif not isinstance(model_id, (list, set)):
            keys = [model_id]
        else:
            keys = model_id
        if keys is not None and '__ensemble__' in keys:
            keys = None

        mask = np.isfinite(y).all(axis=1)
        if not mask.all():
            x = x[mask]
            y = y[mask]
            if sample_weight is not None:
                sample_weight = sample_weight[mask]

        random_seed = self._rng.randint(np.iinfo(np.int32).max + 1)
        baseline = \
            _calc_permutation_scores(self._ensemble, keys, x, y, sample_weight, [], random_seed, 1, self._scorers)

        if mapping is None:
            mapping = {c: [c] for c in x.columns}

        if jobs == 1:
            explanations = [
                _calc_permutation_scores(self._ensemble, keys, x, y, sample_weight, cs, random_seed,
                                         self._n_repetitions, self._scorers)
                for cs in progress_bar(mapping.values(), disable=not show_progress, desc='Features')
            ]
        else:
            explanations = Parallel(n_jobs=jobs)(
                delayed(_calc_permutation_scores)(
                    self._ensemble,
                    keys,
                    x,
                    y,
                    sample_weight,
                    cs,
                    random_seed,
                    self._n_repetitions,
                    self._scorers
                )
                for cs in progress_bar(mapping.values(), disable=not show_progress, desc='Features')
            )

        if keys is None:
            keys = baseline.keys()

        out = {}
        for k in keys:
            mu = pd.DataFrame(
                data=dict(zip(mapping.keys(), [(baseline[k][0] - e[k]).mean(axis=0) for e in explanations]))
            )
            sigma = pd.DataFrame(
                data=dict(zip(mapping.keys(), [(baseline[k][0] - e[k]).std(axis=0) for e in explanations]))
            )
            df = pd.concat([mu, sigma], axis=0, ignore_index=True).T
            df.columns = self._metric_names + [m + ' std' for m in self._metric_names]
            out[k] = df

        return out

    def get_versions(self) -> dict:
        return {'sklearn': sklearn.__version__, 'pandas': pd.__version__}


def _calc_permutation_scores(ensemble: FittedEnsemble, model_ids: Optional[list], X: pd.DataFrame, y: pd.DataFrame,
                             sample_weight: Optional[np.ndarray], columns: list, seed: int, n_repetitions: int,
                             scorers: list) -> dict:
    # copied and adapted from `sklearn.inspection._permutation_importance`

    # return dict mapping model-IDs to 2D-arrays with one column per given scorer and one row per repetition

    if ensemble.task in ('binary_classification', 'multiclass_classification', 'multilabel_classification'):
        if model_ids is None:
            predict_fun = ensemble.predict_proba_all
            predict_kwargs = dict(calibrated_ensemble=True)
        else:
            def predict_fun(_X, **_kwargs):
                return {_k: ensemble.predict_proba(_X, model_id=_k, **_kwargs) for _k in model_ids}

            predict_kwargs = dict(calibrated=False)
    else:
        if model_ids is None:
            predict_fun = ensemble.predict_all
            predict_kwargs = dict(calibrated_ensemble=True)
        else:
            def predict_fun(_X, **_kwargs):
                return {_k: ensemble.predict(_X, model_id=_k, **_kwargs) for _k in model_ids}

            predict_kwargs = dict(calibrated=False)

    # Work on a copy of X to ensure thread-safety in case of threading based
    # parallelism. Furthermore, making a copy is also useful when the joblib
    # backend is 'loky' (default) or the old 'multiprocessing': in those cases,
    # if X is large it will be automatically be backed by a readonly memory map.
    # X.copy() on the other hand is always guaranteed to return a writable
    # data-structure whose columns can be shuffled inplace.
    X_permuted = X.copy()

    rng = np.random.RandomState(seed)

    scores = {}
    shuffling_idx = np.arange(len(X_permuted))
    for _ in range(n_repetitions):
        if columns:
            rng.shuffle(shuffling_idx)
            col = X_permuted[columns].iloc[shuffling_idx]
            col.index = X_permuted.index
            X_permuted[columns] = col
        y_hat: dict = predict_fun(X_permuted, **predict_kwargs)
        if ensemble.task == 'binary_classification':
            # restrict to probability of positive class
            y_hat = {k: (v[:, -1] if v.ndim == 2 else v) for k, v in y_hat.items()}
        if scores is None:
            scores = {k: [] for k in y_hat.keys()}
        for k, v in y_hat.items():
            scores.setdefault(k, []).append([s(y, v, sample_weight=sample_weight) for s in scorers])

    return {k: np.array(v) for k, v in scores.items()}
