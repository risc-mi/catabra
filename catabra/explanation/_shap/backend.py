from typing import Optional
from multiprocessing import Pool
import numpy as np
import pandas as pd
import shap

from .kernel_explainer import CustomKernelExplainer
from catabra.catabra.base.logging import progress_bar
from ..base import TransformationExplainer, IdentityTransformationExplainer, EnsembleExplainer
from ...automl.fitted_ensemble import FittedEnsemble, FittedModel, get_prediction_function


class SHAPEnsembleExplainer(EnsembleExplainer):

    def __init__(self, ensemble: FittedEnsemble = None, feature_names: Optional[list] = None,
                 target_names: Optional[list] = None, x: Optional[pd.DataFrame] = None,
                 y: Optional[pd.DataFrame] = None, params=None):
        super(SHAPEnsembleExplainer, self).__init__(ensemble=ensemble, feature_names=feature_names,
                                                    target_names=target_names, x=x, y=y, params=params)
        self._ensemble = ensemble

        # dict mapping model (pipeline) IDs to pairs `(preprocessing_explainer, estimator_explainer)`
        # `preprocessing_explainer` may be None, if `estimator_explainer` explains the whole pipeline
        # `estimator_explainer` is an instance of class `SHAPExplainer`
        # not every model (pipeline) needs to occur in the dict
        self._explainers = {}

        if params is None:
            assert x is not None
            if y is None:
                n_targets = None
            else:
                y_shape = np.shape(y)
                if len(y_shape) == 1:
                    n_targets = 1
                else:
                    assert len(y_shape) == 2
                    n_targets = y_shape[1]
            if feature_names is None:
                self._feature_names = list(range(x.shape[1]))
            else:
                assert len(feature_names) == x.shape[1]
                self._feature_names = feature_names
            if target_names is None:
                assert self._ensemble.task != 'multiclass_classification'
                assert n_targets is not None
                self._target_names = ['<unknown target>'] if n_targets == 1 else list(range(n_targets))
            else:
                self._target_names = target_names
            self._params = dict(explainers={}, feature_names=self._feature_names, target_names=self._target_names)
            permutation = np.random.permutation(len(x))
            for _id, pipeline in ensemble.models_.items():
                try:
                    preprocessing_explainer = TransformationExplainer.make(pipeline.preprocessing)
                    x_pp = preprocessing_explainer.fit_forward(x, y)
                    estimator_explainer = SHAPExplainer(
                        pipeline.estimator,
                        task=ensemble.task,
                        n_targets=len(self._target_names),
                        data=x_pp,
                        permutation=permutation
                    )
                    self._explainers[_id] = (preprocessing_explainer, estimator_explainer)
                    self._params['explainers'][_id] = dict(
                        preprocessing=preprocessing_explainer.params_,
                        estimator=estimator_explainer.params_
                    )
                except:     # noqa
                    preprocessing_explainer = IdentityTransformationExplainer()
                    estimator_explainer = SHAPExplainer(pipeline, task=ensemble.task, n_targets=len(self._target_names),
                                                        data=x, permutation=permutation)
                    self._explainers[_id] = (preprocessing_explainer, estimator_explainer)
                    self._params['explainers'][_id] = dict(
                        preprocessing=preprocessing_explainer.params_,
                        estimator=estimator_explainer.params_
                    )
        else:
            self._params = params
            unknown_ids = [_id for _id in self._params['explainers'] if _id not in ensemble.model_ids_]
            if unknown_ids:
                raise ValueError('The following model IDs do not appear in the given ensemble: ' + str(unknown_ids))
            self._target_names = self._params['target_names']
            self._feature_names = self._params['feature_names']
            for _id, pipeline in ensemble.models_.items():
                current_params = self._params['explainers'].get(_id)
                if current_params is not None:
                    if not current_params.get('estimator_is_whole_pipeline', False):
                        preprocessing_explainer = TransformationExplainer.make(pipeline.preprocessing,
                                                                               params=current_params['preprocessing'])
                        estimator_explainer = SHAPExplainer(
                            pipeline.estimator,
                            task=ensemble.task,
                            n_targets=len(self._target_names),
                            params=current_params['estimator']
                        )
                    else:
                        preprocessing_explainer = IdentityTransformationExplainer()
                        estimator_explainer = SHAPExplainer(
                            pipeline,
                            task=ensemble.task,
                            n_targets=len(self._target_names),
                            params=current_params['estimator']
                        )
                    self._explainers[_id] = (preprocessing_explainer, estimator_explainer)

    @classmethod
    def name(cls) -> str:
        return 'shap'

    @classmethod
    def global_behavior(cls) -> dict:
        return SHAPExplainer.global_behavior()

    @property
    def params_(self) -> dict:
        return self._params

    def explain(self, x: pd.DataFrame, jobs: int = 1, batch_size: Optional[int] = None, model_id=None,
                show_progress: bool = False) -> dict:
        return self._explain_multi(model_id, x, None, jobs, batch_size, False, not show_progress)

    def explain_global(self, x: Optional[pd.DataFrame] = None, sample_weight: Optional[np.ndarray] = None,
                       jobs: int = 1, batch_size: Optional[int] = None, model_id=None,
                       show_progress: bool = False) -> dict:
        if x is None:
            raise ValueError(f'{self.__class__.__name__} requires samples for global explanations.')
        return self._explain_multi(model_id, x, sample_weight, jobs, batch_size, True, not show_progress)

    @classmethod
    def get_versions(cls) -> dict:
        return {'shap': shap.__version__, 'pandas': pd.__version__}

    def _explain_single(self, model_id, x: pd.DataFrame, jobs: int, batch_size: int, glob: bool,
                        silent: bool) -> np.ndarray:
        func = _explain_single_global if glob else _explain_single
        preprocessing, estimator = self._explainers[model_id]
        if len(x) <= batch_size or glob:
            return func(preprocessing, estimator, x)
        else:
            # joblib with loky backend may lead to segmentation faults, when existing explainers are "shared" among
            # processes. Built-in multiprocessing module works just fine.
            # https://github.com/slundberg/shap/issues/1204
            with Pool(processes=jobs) as pool:
                async_results = [
                    pool.apply_async(
                        func,
                        args=(preprocessing, estimator, x.iloc[i * batch_size:(i + 1) * batch_size]),
                        kwds=dict(copy=True)
                    )
                    for i in range((len(x) + batch_size - 1) // batch_size)
                ]
                explanations = [async_result.get() for async_result in progress_bar(async_results, disable=silent,
                                                                                    desc='Sample batches')]
            return np.concatenate(explanations, axis=-2)    # sample axis is last-but-one

    def _explain_multi(self, model_id: Optional[list], x: pd.DataFrame, sample_weight: Optional[np.ndarray], jobs: int,
                       batch_size: Optional[int], glob: bool, silent: bool) -> dict:
        if batch_size is None:
            batch_size = min(32, len(x))
        if model_id is None:
            model_id = self._ensemble.meta_input_
        elif not isinstance(model_id, (list, set)):
            model_id = [model_id]
        keys = [k for k in model_id if k in self._explainers]

        if len(x) <= batch_size:
            func = _explain_single      # explain locally, average at very end if `glob` is True
            if len(keys) <= 1:
                all_explanations = [func(*self._explainers[key], x) for key in keys]
            else:
                # joblib with loky backend may lead to segmentation faults, when existing explainers are "shared" among
                # processes. Built-in multiprocessing module works just fine.
                # https://github.com/slundberg/shap/issues/1204
                with Pool(processes=jobs) as pool:
                    async_results = [
                        pool.apply_async(
                            func,
                            args=(*self._explainers[key], x),
                            kwds=dict(copy=True)
                        )
                        for key in keys
                    ]
                    all_explanations = [async_result.get()
                                        for async_result in progress_bar(async_results, disable=silent, desc='Models')]
        else:
            all_explanations = []
            for i, key in enumerate(keys):
                if not silent and len(keys) > 1:
                    print(f'Model {key} ({i + 1} of {len(keys)}):')
                # explain locally, average at very end if `glob` is True
                all_explanations.append(self._explain_single(key, x, jobs, batch_size, False, silent))

        out = {k: s for k, s in zip(keys, all_explanations)}
        if isinstance(self._ensemble.meta_estimator_, (list, tuple)) \
                and len(all_explanations) == len(self._ensemble.meta_estimator_) \
                and (len(all_explanations) <= 1 or self._ensemble.task == 'regression'):
            # in a classification setting, shap might explain class probabilities in some cases and log-odds ratios
            # in other cases => cannot be compared, and in particular cannot be averaged
            # https://github.com/slundberg/shap/issues/112
            out['__ensemble__'] = sum(w * s for w, s in zip(self._ensemble.meta_estimator_, all_explanations))

        if glob:
            # reason for explaining everything locally and averaging at end is that we want to distinguish between
            # positive and negative contributions, and back-propagating through preprocessing steps could change
            # the polarity of scores
            out = {k: _local_to_global(v, sample_weight) for k, v in out.items()}

        return {k: self._finalize_output(v, x.index, glob) for k, v in out.items()}

    def _finalize_output(self, s: np.ndarray, index, glob: bool):
        if glob:
            if s.ndim == 2:
                # (2, n_features)
                # don't call columns "pos" and "neg", as these might be confused with positive and negative class
                return pd.DataFrame(data=s.T, index=self._feature_names, columns=['>0', '<0'])
            else:
                # (2, n_targets, n_features)
                return pd.DataFrame(data=s.transpose((2, 1, 0)).reshape(s.shape[-1], -1), index=self._feature_names,
                                    columns=[n + suffix for n in self._target_names for suffix in ('_>0', '_<0')])
        else:
            if s.ndim == 2:
                return pd.DataFrame(data=s, index=index, columns=self._feature_names)
            else:
                return {n: self._finalize_output(s[i], index, glob) for i, n in enumerate(self._target_names)}


class SHAPExplainer:

    def __init__(self, estimator, task=None, n_targets: int = 1, data=None, permutation=None, params=None):
        self._task = task
        self._n_targets = n_targets
        if params is None:
            assert data is not None
            if isinstance(estimator, FittedModel):
                kwargs = dict(data=_sample(data, permutation=permutation), keep_index=True)
                self._explainer = CustomKernelExplainer(
                    get_prediction_function(estimator, proba=self._task != 'regression'),
                    **kwargs
                )
                self._params = dict(explainer_class='CustomKernelExplainer', init_kwargs=kwargs,
                                    pre='prediction_function', call_kwargs=dict(silent=True))
            else:
                self._explainer, self._params = _get_explainer(estimator, data, permutation, self._task != 'regression')
        else:
            assert data is None
            self._params = params
            self._explainer = _make_explainer(estimator, self._params, self._task != 'regression')

    @classmethod
    def global_behavior(cls) -> dict:
        return dict(accepts_x=True, requires_x=True, mean_of_local=True)

    @property
    def params_(self) -> dict:
        return self._params

    def explain(self, x, jobs: int = 1, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Explain the estimator on a given set of samples.
        :param x: Samples to explain, array-like of shape `(n_samples, n_features)`.
        :param jobs: The number of jobs to use.
        :param batch_size: The batch size to use.
        :return: SHAP feature importance scores, numerical array whose shape and meaning depends on the prediction
        task:
        * Regression: Shape is `([n_targets], n_samples, n_features)`, where the first axis is only present if there
            are multiple targets. Features with positive scores contribute to increasing the model output, features
            with negative scores contribute to decreasing it. The overall contribution of a feature is proportional to
            its magnitude.
        * Binary classification: Shape is `(n_samples, n_features)`. Features with positive scores are indicators for
            the positive class, features with negative scores are indicators for the negative class. Note, however,
            that the overall contribution of a feature might _not_ be proportional to its magnitude, i.e., a feature
            with score -1 can be as important as a feature with score +2 (in the same sample).
            Reason: non-linear logistic function applied to the raw model output when explaining class probabilities.
        * Multiclass classification: Shape is `(n_classes, n_samples, n_features)`. Features with positive scores for
            some class are indicators in favor of that class, features with negative scores are indicators against the
            class. Analogous to binary classification, the overall importance of a feature might _not_ be proportional
            to its magnitude.
        * Multilabel classification: Shape is `(n_classes, n_samples, n_features)`, with the same meaning as in
            binary classification for each class individually.
        """

        # Defining property of Shapley values ("local accuracy"): sample-wise scores sum to the difference between the
        # model output for the current sample and the average/expected model output. What exactly "model output" means
        # depends on the task, the model and the explanation method, but in any case it must be array-like of shape
        # `(n_samples,)` or `(n_samples, n_outputs)`:
        #   * KernelExplainer: model output is whatever passed function returns, e.g., `predict_proba()`.
        #   * regression tasks: model output is predicted value, i.e., as returned by `predict()`.
        #   * TreeExplainer with `model_output="raw"`: model output depends on model.
        # In case of multiple outputs, shap returns a list of scores, one for each output -- regardless of the
        # prediction task!

        # From the definition of Shapley values another important property can be derived: Let `g_1`, ..., `g_n` be
        # functions that map from R^m into R, and let `w_1`, ..., `w_n` be weights.
        # If `w_1 * g_1(x) + ... + w_n * g_n(x)` is constant, then the Shapley values `phi_i_j(x)` (for `1 <= i <= n`
        # and `1 <= j <= m`) satisfy
        # `w_1 * phi_1_j(x) + ... + w_n * phi_n_j(x) = 0` for all `1 <= j <= m` and all `x`.
        # In particular, if `g_1`, ..., `g_n` correspond to the individual class probabilities in a binary- or
        # multiclass problem, and hence satisfy `g_1(x) + ... + g_n(x) = 1`, then `phi_1_j(x) + ... + phi_n_j(x) = 0`.

        s = self._explainer.shap_values(x, **self._params.get('call_kwargs', {}))
        if isinstance(s, list):
            if self._task == 'binary_classification':
                assert len(s) <= 2
                s = s[-1]
            elif self._task == 'multilabel_classification' and len(s) != self._n_targets:
                # in multilabel case, this could be list of length 2 * `n_classes`, where first two elements
                # correspond to first class, next two elements to second class, etc.
                assert len(s) == 2 * self._n_targets
                s = np.stack(s[::2], axis=0)
            else:
                s = np.stack(s, axis=0)

        assert 2 <= s.ndim <= 3
        assert s.shape[-2:] == np.shape(x)
        if self._task == 'regression':
            if s.ndim == 2:
                assert self._n_targets == 1
            else:
                assert s.shape[0] == self._n_targets
                if self._n_targets == 1:
                    s = s[0]
        elif self._task == 'binary_classification':
            if s.ndim == 3:
                assert s.shape[0] == 2
                s = s[1]
        else:
            assert s.ndim == 3
            if self._task == 'multilabel_classification':
                assert s.shape[0] == self._n_targets
        return s

    def explain_global(self, x=None, sample_weight: Optional[np.ndarray] = None, jobs: int = 1,
                       batch_size: Optional[int] = None) -> np.ndarray:
        """
        Explain the estimator globally w.r.t. a given set of samples. This amounts to explaining the given samples
        and then averaging the obtained SHAP values.
        :param x: Samples to explain, array-like of shape `(n_samples, n_features)`.
        :param sample_weight: Sample weights. None or array of shape `(n_samples,)`.
        :param jobs: The number of jobs to use.
        :param batch_size: The batch size to use.
        :return: SHAP feature importance scores, numerical array whose shape and meaning depends on the prediction
        task:
        * Regression: Shape is `(2, [n_targets], n_features)`, where the second axis is only present if there are
            multiple targets.
        * Binary classification: Shape is `(2, n_features)`.
        * Multiclass- and multilabel classification: Shape is `(2, n_classes, n_features)`.
        In any case, the first axis distinguishes between positive (index 0) and negative (index 1) contributions.
        Averaging happens by dividing through the total number of samples, such that `result[0] + result[1]` is the
        average feature importance and `result[0] - result[1]` is the average _absolute_ feature importance.

        See method `explain()` for details.
        """
        if x is None:
            raise ValueError(f'{self.__class__.__name__} requires samples for global explanations.')
        return _local_to_global(self.explain(x, jobs=jobs, batch_size=batch_size), sample_weight)


class MultiOutputExplainer(shap.Explainer):

    def __init__(self, model, data=None, permutation=None, proba=False, params=None, **kwargs):
        self.explainers = []
        if params is None:
            self.params = []
            for e in model.estimators_:
                expl, p = _get_explainer(e, data, permutation, proba)
                self.explainers.append(expl)
                self.params.append(p)
        else:
            assert len(params) == len(model.estimators_)
            self.params = params
            for e, p in zip(model.estimators_, self.params):
                self.explainers.append(_make_explainer(e, p, proba))

    def shap_values(self, x, **kwargs):
        out = []
        for e, p in zip(self.explainers, self.params):
            s = e.shap_values(x, **p.get('call_kwargs', {}), **kwargs)
            if isinstance(s, list):
                out.extend(s)
            else:
                out.append(s)
        if len(out) == 1:
            return out[0]
        return out


class OneVsRestExplainer(MultiOutputExplainer):
    """Explainer for OneVsRestClassifier models, can be used for multiclass classification only."""

    def __init__(self, model, **kwargs):
        kwargs['proba'] = True
        super(OneVsRestExplainer, self).__init__(model, **kwargs)

    def shap_values(self, x, **kwargs):
        out = []
        for e, p in zip(self.explainers, self.params):
            s = e.shap_values(x, **p.get('call_kwargs', {}), **kwargs)
            if isinstance(s, list):
                # binary classification
                assert 1 <= len(s) <= 2
                s = s[-1]
            out.append(s)
        if len(out) == 1:
            return out[0]
        return out


def _pre_adaboost(estimator):
    # special treatment of AdaBoost if base estimator is tree-based
    # https://stackoverflow.com/questions/60433389/how-to-calculate-shap-values-for-adaboost-model
    # https://github.com/slundberg/shap/issues/335
    estimators = getattr(estimator, 'estimators_', None)
    if estimators is not None:
        tree = getattr(estimators[0], 'tree_', None)
        if tree is not None:
            scaling = 1.0 / len(estimators)     # output is average of trees
            if estimator.base_estimator_.criterion in ('mse', 'variance', 'friedman_mse', 'reg:linear',
                                                       'reg:squarederror', 'regression', 'regression_l2'):
                objective = 'squared_error'
            elif estimator.base_estimator_.criterion in ('mae',):
                objective = 'absolute_error'
            elif estimator.base_estimator_.criterion in ('gini', 'entropy', 'reg:logistic', 'binary:logistic',
                                                         'binary_logloss', 'binary'):
                objective = 'binary_crossentropy'
            else:
                objective = None
            return dict(
                internal_dtype=tree.value.dtype.type,
                input_dtype=np.float32,
                objective=objective,
                tree_output='raw_value' if estimator.__class__.__name__.endswith('Regressor') else 'probability',
                trees=[_tree_to_dict(e.tree_, normalize=True, scaling=scaling) for e in estimators]
            )


def _tree_to_dict(tree, **kwargs) -> dict:
    st = shap.explainers._tree.SingleTree(tree, **kwargs)
    return dict(
        children_left=st.children_left,
        children_right=st.children_right,
        children_default=st.children_left,
        features=st.features,
        thresholds=st.thresholds,
        values=st.values,
        node_sample_weight=st.node_sample_weight
    )


def _get_explainer(estimator, data, permutation, proba):
    if shap.utils.safe_isinstance(estimator, ('sklearn.ensemble._weight_boosting.AdaBoostClassifier',
                                              'sklearn.ensemble._weight_boosting.AdaBoostRegressor')):
        model_dict = _pre_adaboost(estimator)
        if model_dict is not None:
            try:
                kwargs = dict(model_output='raw')
                explainer = shap.TreeExplainer(model_dict, **kwargs)
                return explainer, dict(explainer_class='TreeExplainer', init_kwargs=kwargs, pre='adaboost')
            except:  # noqa
                pass
    elif shap.utils.safe_isinstance(estimator, ('sklearn.multiclass.OneVsRestClassifier',
                                                'sklearn.multioutput.MultiOutputClassifier',
                                                'sklearn.multioutput.MultiOutputRegressor')):
        if getattr(estimator, 'multilabel_', True):
            kwargs = dict(proba=proba)
            explainer = MultiOutputExplainer(estimator, data=data, permutation=permutation, **kwargs)
            kwargs['params'] = explainer.params
            return explainer, dict(explainer_class='MultiOutputExplainer', init_kwargs=kwargs)
        else:
            explainer = OneVsRestExplainer(estimator, data=data, permutation=permutation)
            return explainer, dict(explainer_class='OneVsRestExplainer', init_kwargs=dict(params=explainer.params))

    try:
        kwargs = dict(model_output='raw')
        explainer = shap.TreeExplainer(estimator, **kwargs)
        return explainer, dict(explainer_class='TreeExplainer', init_kwargs=kwargs)
    except:  # noqa
        pass
    try:
        # covariance matrix is not needed for interventional feature perturbation, which is the default
        kwargs = dict(masker=(data.mean(axis=0), None))
        explainer = shap.LinearExplainer(estimator, **kwargs)
        return explainer, dict(explainer_class='LinearExplainer', init_kwargs=kwargs)
    except:  # noqa
        pass

    data_sample = _sample(data, permutation)
    try:
        kwargs = dict(masker=data_sample)
        explainer = shap.AdditiveExplainer(estimator, **kwargs)
        return explainer, dict(explainer_class='AdditiveExplainer', init_kwargs=kwargs)
    except:  # noqa
        pass
    try:
        kwargs = dict(data=data_sample)
        explainer = shap.GradientExplainer(estimator, **kwargs)
        return explainer, dict(explainer_class='GradientExplainer', init_kwargs=kwargs)
    except:  # noqa
        pass

    kwargs = dict(data=data_sample, keep_index=True)
    explainer = CustomKernelExplainer(get_prediction_function(estimator, proba=proba), **kwargs)
    return explainer, dict(explainer_class='CustomKernelExplainer', init_kwargs=kwargs, pre='prediction_function',
                           call_kwargs=dict(silent=True))


def _make_explainer(estimator, params: dict, proba: bool):
    explainer_class = params['explainer_class']
    pre = params.get('pre')
    if pre == 'prediction_function':
        estimator = get_prediction_function(estimator, proba=proba)
    elif pre == 'adaboost':
        estimator = _pre_adaboost(estimator)
    if explainer_class == 'MultiOutputExplainer':
        cls = MultiOutputExplainer
    elif explainer_class == 'OneVsRestExplainer':
        cls = OneVsRestExplainer
    elif explainer_class == 'CustomKernelExplainer':
        cls = CustomKernelExplainer
    else:
        cls = getattr(shap, explainer_class)
    return cls(estimator, **params['init_kwargs'])


def _sample(x, permutation: np.ndarray = None, n: int = 100):
    if len(x) > n:
        if permutation is None:
            idx = np.random.choice(len(x), size=n, replace=False)
        else:
            idx = permutation[:n]

        if hasattr(x, 'iloc'):
            return x.iloc[idx]
        else:
            return x[idx]

    return x


def _local_to_global(s: np.ndarray, sample_weight: Optional[np.ndarray]) -> np.ndarray:
    if sample_weight is None:
        div = s.shape[-2]
        positive = (s * (s > 0)).sum(axis=-2)  # sample axis is last-but-one
        negative = (s * (s < 0)).sum(axis=-2)  # sample axis is last-but-one
    else:
        div = sample_weight.sum()
        sample_weight = sample_weight.reshape([1] * (s.ndim - 2) + [-1, 1])
        positive = ((s * (s > 0)) * sample_weight).sum(axis=-2)  # sample axis is last-but-one
        negative = ((s * (s < 0)) * sample_weight).sum(axis=-2)  # sample axis is last-but-one
    return np.stack([positive, negative], axis=0) / div


def _explain_single(preprocessing_explainer: TransformationExplainer, estimator_explainer: SHAPExplainer,
                    x, copy: bool = False) -> np.ndarray:
    x = preprocessing_explainer.forward(x.copy() if copy else x)
    s = estimator_explainer.explain(x)
    return preprocessing_explainer.backward(s)


def _explain_single_global(preprocessing_explainer: TransformationExplainer, estimator_explainer: SHAPExplainer,
                           x, copy: bool = False) -> np.ndarray:
    if x is not None:
        x = preprocessing_explainer.transform(x.copy() if copy else x)
    s = estimator_explainer.explain_global(x=x)
    return preprocessing_explainer.backward_global(s)
