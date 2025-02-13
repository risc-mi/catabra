#  Copyright (c) 2022-2025. RISC Software GmbH.
#  All rights reserved.

from functools import partial
from pathlib import Path
from typing import Callable, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from catabra_lib import metrics
from catabra_lib.bootstrapping import Bootstrapping

from catabra.core import CaTabRaBase, CaTabRaPaths, Invocation
from catabra.ood.base import OverallOODDetector
from catabra.util import io, logging, plotting, statistics
from catabra.util import table as tu


def evaluate(*table: Union[str, Path, pd.DataFrame], folder: Union[str, Path] = None, model_id=None, explain=None,
             glob: Optional[bool] = False, split: Optional[str] = None, sample_weight: Optional[str] = None,
             create_stats: Optional[bool] = None, check_ood: Optional[bool] = None, out: Union[str, Path, None] = None,
             jobs: Optional[int] = None, batch_size: Optional[int] = None, threshold: Union[float, str, None] = None,
             bootstrapping_repetitions: Optional[int] = None, bootstrapping_metrics: Optional[list] = None,
             from_invocation: Union[str, Path, dict, None] = None):
    """
    Evaluate an existing CaTabRa object (OOD-detector, prediction model, ...) on held-out test data.

    Parameters
    ----------
    *table:
        The table(s) to evaluate the CaTabRa object on. If multiple are given, their columns are merged into a single
        table. Must have the same format as the table(s) initially passed to function `analyze()`.
    folder: str | PAth
        The folder containing the CaTabRa object to evaluate.
    model_id: optional
        ID of the prediction model to evaluate. If None or "__ensemble__", the sole trained model or the entire ensemble
        are evaluated.
    explain: list | str, optional
        Explain prediction model(s). If `"__all__"`, all models specified by `model_id` are explained; otherwise, must
        be a list of the model ID(s) to explain, which must be a subset of the models that are evaluated.
    glob: bool, default=False
        Whether to create global instead of local explanations.
    split: str, optional
        Column used for splitting the data into disjoint subsets. If specified and not `""`, each subset is evaluated
        individually. In contrast to function `analyze()`, the name/values of the column do not need to carry any
        semantic information about training and test sets.
    sample_weight: str, optional
        Column with sample weights. If specified and not `""`, must have numeric data type. Sample weights are used both
        for training and evaluating prediction models.
    create_stats: bool, optional
        Whether to generate and save descriptive statistics of the given data table.
    check_ood: bool, optional
        Whether to apply the OOD-detector (if any) to the given data.
    out: str | Path, optional
        Directory where to save all generated artifacts. Defaults to a directory located in `folder`, with a name
        following a fixed naming pattern. If `out` already exists, the user is prompted to specify whether it should be
        replaced; otherwise, it is automatically created.
    jobs: int, optional
        Number of jobs to use. Overwrites the "jobs" config param.
    batch_size: int, optional
        Batch size used for applying the prediction model.
    threshold: float | str, optional
        Decision threshold for binary- and multilabel classification tasks. Confusion matrix plots and bootstrapped
        performance results are reported for this particular threshold. In binary classification this can also be the
        name of a built-in thresholding strategy, possibly followed by "on" and the split on which to calculate the
        threshold. Splits must be specified by the name of the subdirectory containing the corresponding evaluation
        results. See /doc/metrics.md for a list of built-in thresholding strategies.
    bootstrapping_repetitions: int, optional
        Number of bootstrapping repetitions. Overwrites the `"bootstrapping_repetitions"` config param.
    bootstrapping_metrics: list, optional
        Names of metrics for which bootstrapped scores are computed, if. Defaults to the list of main metrics specified
        in the default config. Can also be `"__all__"`, in which case all standard metrics for the current prediction
        task are computed. Ignored if bootstrapping is disabled.
    from_invocation: str | Path | dict, optional
        Dict or path to an invocation.json file. All arguments of this function not explicitly specified are taken from
        this dict; this also includes the table on which to evaluate the CaTabRa object.
    """
    evaluator = CaTabRaEvaluation(invocation=from_invocation)
    evaluator(
        *table,
        folder=folder,
        model_id=model_id,
        explain=explain,
        glob=glob,
        split=split,
        sample_weight=sample_weight,
        create_stats=create_stats,
        check_ood=check_ood,
        out=out,
        jobs=jobs,
        batch_size=batch_size,
        threshold=threshold,
        bootstrapping_repetitions=bootstrapping_repetitions,
        bootstrapping_metrics=bootstrapping_metrics
    )


class CaTabRaEvaluation(CaTabRaBase):

    @property
    def invocation_class(self) -> Type['EvaluationInvocation']:
        return EvaluationInvocation

    def _call(self):
        loader = io.CaTabRaLoader(self._invocation.folder, check_exists=True)
        self._config = loader.get_config()

        out_ok = self._invocation.resolve_output_dir(
            prompt=f'Evaluation folder "{self._invocation.out.as_posix()}" already exists. Delete?'
        )
        if not out_ok:
            logging.log('### Aborting')
            return

        with logging.LogMirror((self._invocation.out / CaTabRaPaths.ConsoleLogs).as_posix()):
            logging.log(f'### Evaluation started at {self._invocation.start}')
            io.dump(io.to_json(self._invocation), self._invocation.out / CaTabRaPaths.Invocation)

            # merge tables
            df, _ = tu.merge_tables(self._invocation.table)
            if df.columns.nlevels != 1:
                raise ValueError(f'Table must have 1 column level, but found {df.columns.nlevels}.')

            # copy test data
            copy_data = self._config.get('copy_evaluation_data', False)
            if isinstance(copy_data, (int, float)):
                copy_data = df.memory_usage(index=True, deep=True).sum() <= copy_data * 1000000
            if copy_data:
                io.write_df(df, self._invocation.out / CaTabRaPaths.TestData)

            # split
            _iter_splits = self._get_split_iterator(df)
            sample_weights = self._invocation.get_sample_weights(df)
            encoder = loader.get_encoder()
            x_test, y_test = encoder.transform(data=df)

            model = loader.get_model_or_fitted_ensemble()
            ood = loader.get_ood()

            # descriptive statistics and OOD detection for each train/test split
            if encoder.task_ is not None:
                target = list(y_test.columns)
                for mask, directory in _iter_splits():
                    if self._invocation.create_stats:
                        statistics.save_descriptive_statistics(df=df.loc[mask, list(x_test.columns) + target],
                                                               target=target,
                                                               classify=encoder.task_ != 'regression',
                                                               fn=directory / CaTabRaPaths.Statistics)
                    if self._invocation.check_ood and ood is not None:
                        # OOD detection must be applied to each split separately, because some detectors (e.g., KS)
                        # evaluate distribution of whole dataset
                        ood_proba = ood.predict_proba(x_test[mask])
                        if isinstance(ood_proba, (pd.Series, np.ndarray)):
                            ood_predictions = pd.DataFrame(ood_proba, columns=['proba'])
                        else:
                            assert isinstance(ood, OverallOODDetector)
                            ood_predictions = pd.DataFrame(index=['overall'], data={'proba': [ood_proba]})
                        ood_predictions['decision'] = ood.predict(x_test[mask])
                        io.write_df(ood_predictions, directory / CaTabRaPaths.OODStats)

            if encoder.task_ is None or model is None:
                self._invocation.explain = []
            else:
                self._invocation.set_models_to_explain(model)
                # `explain` is now either None or a list: None => explain all models; list => explain only these models

                main_metrics = self._config.get(encoder.task_ + '_metrics', [])

                if self._invocation.bootstrapping_repetitions is None:
                    self._invocation.bootstrapping_repetitions = self._config.get('bootstrapping_repetitions', 0)
                if encoder.task_ == 'regression':
                    self._evaluate_regression_task(_iter_splits, encoder, main_metrics, model,
                                                   sample_weights, x_test, y_test)
                else:
                    self._evaluate_classification_task(_iter_splits, encoder, main_metrics, model,
                                                       sample_weights, x_test, y_test)

            end = pd.Timestamp.now()
            logging.log(f'### Evaluation finished at {end}')
            logging.log(f'### Elapsed time: {end - self._invocation.start}')
            logging.log(f'### Output saved in {self._invocation.out.as_posix()}')

        if self._invocation.explain is None or len(self._invocation.explain) > 0:
            from ..explanation import explain as explain_fn
            explain_fn(df, folder=self._invocation.folder, split=self._invocation.split,
                       sample_weight=self._invocation.sample_weight, model_id=self._invocation.explain,
                       out=self._invocation.out / 'explanations', glob=self._invocation.glob,
                       batch_size=self._invocation.batch_size, jobs=self._invocation.jobs)

    def _evaluate_classification_task(self, _iter_splits, encoder, main_metrics, model,
                                      sample_weights, x_test, y_test):
        y_hat = model.predict_proba(x_test, jobs=self._invocation.jobs, batch_size=self._invocation.batch_size,
                                    model_id=self._invocation.model_id)

        if encoder.task_ == 'multilabel_classification':
            # decoded ground truth and predictions for each target
            y_test_decoded = encoder.inverse_transform(y=y_test, inplace=False)
            y_hat_decoded = encoder.inverse_transform(y=y_hat, inplace=True)
            y_hat_decoded.index = y_test_decoded.index
            y_hat_decoded.columns = [f'{c}_proba' for c in y_hat_decoded.columns]
            detailed = pd.concat([y_test_decoded, y_hat_decoded], axis=1, sort=False)   # don't use `join()` here
            detailed = detailed.reindex(
                [n for c in zip(y_test_decoded.columns, y_hat_decoded.columns) for n in c],
                axis=1
            )
            del y_test_decoded
            del y_hat_decoded

            if sample_weights is not None:
                detailed['__sample_weight'] = sample_weights

            if isinstance(self._invocation.threshold, float):
                threshold = self._invocation.threshold
            else:
                threshold = 0.5
                logging.warn('Thresholding strategies can only be specified in binary classification.'
                             ' Using default decision threshold of 0.5.')

            for mask, directory in _iter_splits():
                directory.mkdir(exist_ok=True, parents=True)
                io.write_df(detailed[mask], directory / CaTabRaPaths.Predictions)
                evaluate_split(y_test[mask], y_hat[mask], encoder, directory=directory,
                               main_metrics=main_metrics,
                               sample_weight=None if sample_weights is None else sample_weights[mask],
                               threshold=threshold,
                               static_plots=self._config.get('static_plots', True),
                               interactive_plots=self._config.get('interactive_plots', False),
                               bootstrapping_repetitions=self._invocation.bootstrapping_repetitions,
                               bootstrapping_metrics=self._invocation.bootstrapping_metrics,
                               split=(None if directory == self._invocation.out else directory.stem), verbose=True)
        else:
            # decoded ground truth and predictions for each target
            detailed = encoder.inverse_transform(y=y_test, inplace=False)
            if encoder.task_ == 'multiclass_classification':
                detailed[detailed.columns[0] + '_pred'] = \
                    encoder.inverse_transform(y=metrics.multiclass_proba_to_pred(y_hat)).values[:, 0]
            y_hat_decoded = encoder.inverse_transform(y=y_hat, inplace=True)
            y_hat_decoded.index = detailed.index
            y_hat_decoded.columns = [f'{c}_proba' for c in y_hat_decoded.columns]
            detailed = pd.concat([detailed, y_hat_decoded], axis=1, sort=False)     # don't use `join()` here
            del y_hat_decoded

            if encoder.task_ == 'multiclass_classification':
                mask = np.isfinite(y_test.values[:, 0])
                detailed['__true_proba'] = np.nan
                detailed.loc[mask, '__true_proba'] = \
                    y_hat[mask][np.arange(mask.sum()), y_test.values[mask, 0].astype(np.int32)]
                detailed['__true_rank'] = \
                    (y_hat > detailed['__true_proba'].values[..., np.newaxis]).sum(axis=1) + 1 + \
                    ((y_hat == detailed['__true_proba'].values[..., np.newaxis]) &
                     (np.arange(y_hat.shape[1])[np.newaxis] > y_test.values)).sum(axis=1)
                detailed.loc[~mask, '__true_rank'] = -1
            if sample_weights is not None:
                detailed['__sample_weight'] = sample_weights

            threshold = self._invocation.threshold
            if isinstance(self._invocation.threshold, float) or self._invocation.threshold_on is None \
                    or encoder.task_ != 'binary_classification':
                pass
            elif self._invocation.split is None:
                logging.warn(f'Unable to calculate decision threshold on split "{self._invocation.threshold_on}"'
                             ' (no split specified).')
            else:
                for mask, directory in _iter_splits():
                    if self._invocation.threshold_on == directory.stem:
                        threshold = threshold(y_test[mask].values[:, 0], y_hat[mask][:, -1],
                                              sample_weight=None if sample_weights is None else sample_weights[mask])
                        logging.log(
                            f'Calculated decision threshold on split "{self._invocation.threshold_on}": {threshold}'
                        )
                        break
                if not isinstance(threshold, float):
                    logging.warn(f'Unable to calculate decision threshold on split "{self._invocation.threshold_on}"'
                                 ' (split not found). Calculating it for each split individually.')

            for mask, directory in _iter_splits():
                directory.mkdir(exist_ok=True, parents=True)
                io.write_df(detailed[mask], directory / CaTabRaPaths.Predictions)
                evaluate_split(y_test[mask], y_hat[mask], encoder, directory=directory,
                               main_metrics=main_metrics,
                               sample_weight=None if sample_weights is None else sample_weights[mask],
                               threshold=threshold,
                               static_plots=self._config.get('static_plots', True),
                               interactive_plots=self._config.get('interactive_plots', False),
                               bootstrapping_repetitions=self._invocation.bootstrapping_repetitions,
                               bootstrapping_metrics=self._invocation.bootstrapping_metrics,
                               split=(None if directory == self._invocation.out else directory.stem), verbose=True)

    def _evaluate_regression_task(self, _iter_splits, encoder, main_metrics, model, sample_weights, x_test, y_test):
        y_hat = model.predict(x_test, jobs=self._invocation.jobs, batch_size=self._invocation.batch_size,
                              model_id=self._invocation.model_id)
        if y_hat.ndim == 1:
            y_hat = y_hat.reshape(-1, 1)
        # decoded ground truth and predictions for each target
        y_test_decoded = encoder.inverse_transform(y=y_test, inplace=False)
        y_hat_decoded = encoder.inverse_transform(y=y_hat, inplace=True)
        y_hat_decoded.index = y_test_decoded.index
        for mask, directory in _iter_splits():
            evaluate_split(y_test[mask], y_hat[mask], encoder, directory=directory,
                           y_hat_decoded=y_hat_decoded[mask], y_true_decoded=y_test_decoded[mask],
                           main_metrics=main_metrics,
                           sample_weight=None if sample_weights is None else sample_weights[mask],
                           static_plots=self._config.get('static_plots', True),
                           interactive_plots=self._config.get('interactive_plots', False),
                           bootstrapping_repetitions=self._invocation.bootstrapping_repetitions,
                           bootstrapping_metrics=self._invocation.bootstrapping_metrics,
                           split=(None if directory == self._invocation.out else directory.stem), verbose=True)
        y_hat_decoded.columns = [f'{c}_pred' for c in y_hat_decoded.columns]
        detailed = pd.concat([y_test_decoded, y_hat_decoded], axis=1, sort=False)       # don't use `join()` here
        detailed = detailed.reindex(
            [n for c in zip(y_test_decoded.columns, y_hat_decoded.columns) for n in c],
            axis=1
        )
        del y_test_decoded
        del y_hat_decoded
        if sample_weights is not None:
            detailed['__sample_weight'] = sample_weights
        for mask, directory in _iter_splits():
            io.write_df(detailed[mask], directory / CaTabRaPaths.Predictions)

    def _get_split_iterator(self, df):
        if self._invocation.split is None:
            def _iter_splits():
                yield np.ones((len(df),), dtype=bool), self._invocation.out
        else:
            split_masks, _ = tu.train_test_split(df, self._invocation.split)

            def _iter_splits():
                for _k, _m in split_masks.items():
                    yield _m, self._invocation.out / _k

        return _iter_splits


class EvaluationInvocation(Invocation):

    @property
    def folder(self) -> Union[str, Path]:
        return self._folder

    @property
    def model_id(self) -> Optional[str]:
        return self._model_id

    @property
    def explain(self):
        return self._explain

    @explain.setter
    def explain(self, value):
        self._explain = value

    @property
    def glob(self) -> Optional[bool]:
        return self._glob

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    @property
    def threshold(self) -> Union[float, Callable]:
        return self._threshold

    @property
    def threshold_on(self) -> Optional[str]:
        return self._threshold_on

    @property
    def bootstrapping_repetitions(self) -> Optional[int]:
        return self._bootstrapping_repetitions

    @bootstrapping_repetitions.setter
    def bootstrapping_repetitions(self, value: int):
        self._bootstrapping_repetitions = value

    @property
    def bootstrapping_metrics(self) -> Optional[list]:
        return self._bootstrapping_metrics

    @property
    def create_stats(self) -> Optional[bool]:
        return self._create_stats

    @property
    def check_ood(self) -> Optional[bool]:
        return self._check_ood

    def __init__(
        self,
        *table,
        split: Optional[str] = None,
        sample_weight: Optional[str] = None,
        create_stats: Optional[bool] = None,
        check_ood: Optional[bool] = None,
        out: Union[str, Path, None] = None,
        jobs: Optional[int] = None,
        folder: Union[str, Path] = None,
        model_id=None,
        explain=None,
        glob: Optional[bool] = False,
        batch_size: Optional[int] = None,
        threshold: Union[float, str, None] = None,
        bootstrapping_repetitions: Optional[int] = None,
        bootstrapping_metrics: Optional[list] = None,
    ):

        super().__init__(*table, split=split, sample_weight=sample_weight, out=out, jobs=jobs)
        self._folder = folder
        self._model_id = model_id
        self._create_stats = create_stats
        self._check_ood = check_ood
        self._explain = explain
        self._glob = glob
        self._batch_size = batch_size
        self._threshold_raw = threshold
        self._threshold = 0.5       # default value, overwritten in `resolve()`
        self._threshold_on = None   # default value, overwritten in `resolve()`
        self._bootstrapping_repetitions = bootstrapping_repetitions
        self._bootstrapping_metrics = bootstrapping_metrics

    def update(self, src: dict = None):
        super().update(src)
        if src:
            if self._folder is None:
                self._folder = src.get('folder')
            if self._model_id is None:
                self._model_id = src.get('model_id')
            if self._explain is None:
                self._explain = src.get('explain')
            if self._glob is None:
                self._glob = src.get('glob')
            if self._threshold_raw is None:
                self._threshold_raw = src.get('threshold')
            if self._bootstrapping_repetitions is None:
                self._bootstrapping_repetitions = src.get('bootstrapping_repetitions')
            if self._bootstrapping_metrics is None:
                self._bootstrapping_metrics = src.get('bootstrapping_metrics')
            if self._batch_size is None:
                self._batch_size = src.get('batch_size')
            if self._create_stats is None:
                self._create_stats = src.get('create_stats')
            if self._check_ood is None:
                self._check_ood = src.get('check_ood')

    def resolve(self):
        super().resolve()
        if isinstance(self._threshold_raw, str):
            try:
                self._threshold = float(self._threshold_raw)
            except ValueError:
                s = self._threshold_raw.split(' on ', 1)
                self._threshold = metrics.get_thresholding_strategy(s[0].strip())
                if len(s) == 1:
                    if self._split is not None:
                        logging.warn(
                            'Calculating thresholds for each split individually.'
                            ' Normally, a single threshold should be calculated on one target split and then used for'
                            ' all other splits. This can be achieved by adding " on " and the name of the target split'
                            ' to the thresholding strategy.'
                        )
                else:
                    self._threshold_on = s[1].strip()
        elif isinstance(self._threshold_raw, (int, float)):
            self._threshold = self._threshold_raw

        if self._create_stats is None:
            self._create_stats = True
        if self._check_ood is None:
            self._check_ood = True

        if self._folder is None:
            raise ValueError('No CaTabRa directory specified.')
        else:
            self._folder = io.make_path(self._folder, absolute=True)

        if self._out is None:
            self._out = self._table[0]
            if isinstance(self._out, pd.DataFrame):
                self._out = self._folder / ('eval_' + self._start.strftime('%Y-%m-%d_%H-%M-%S'))
            else:
                self._out = self._folder / ('eval_' + self._out.stem + '_' + self._start.strftime('%Y-%m-%d_%H-%M-%S'))
        else:
            self._out = io.make_path(self._out, absolute=True)
        if self._out == self._folder:
            raise ValueError(
                f'Output directory must differ from CaTabRa directory, but both are "{self._out.as_posix()}".'
            )

    def to_dict(self) -> dict:
        dic = super().to_dict()
        dic.update(dict(
            table=['<DataFrame>' if isinstance(tbl, pd.DataFrame) else tbl for tbl in self._table],
            folder=self._folder,
            model_id=self._model_id,
            explain=self._explain,
            glob=self._glob,
            threshold=self._threshold_raw,
            bootstrapping_repetitions=self._bootstrapping_repetitions,
            bootstrapping_metrics=self._bootstrapping_metrics,
            batch_size=self._batch_size,
            create_stats=self._create_stats,
            check_ood=self._check_ood
        ))
        return dic

    def set_models_to_explain(self, model):
        if self._model_id == '__ensemble__':
            self._model_id = None

        if self._explain is None:
            self._explain = set()
        elif isinstance(self._explain, list):

            self._explain = set(self._explain)
        elif not isinstance(self._explain, set):
            self._explain = {self._explain}
        if '__ensemble__' in self._explain:
            if self._model_id is None:
                self._explain = None
            else:
                self._explain.remove('__ensemble__')
                self._explain.update(model.model_ids_)
        elif '__all__' in self._explain:
            if self._model_id is None:
                self._explain = None
            else:
                self._explain.remove('__all__')
                self._explain.add(self._model_id)
        if isinstance(self._explain, set):
            if self._model_id is not None and any(e != self._model_id for e in self._explain):
                raise ValueError('Cannot explain models that are not being evaluated.')
            self._explain = list(self._explain)

    @staticmethod
    def requires_table() -> bool:
        return True


def evaluate_split(y_true: pd.DataFrame, y_hat: np.ndarray, encoder, directory=None, main_metrics: list = None,
                   y_true_decoded=None, y_hat_decoded=None, sample_weight: Optional[np.ndarray] = None,
                   threshold: Union[float, Callable] = 0.5, static_plots: bool = True, interactive_plots: bool = False,
                   bootstrapping_repetitions: int = 0, bootstrapping_metrics: Optional[list] = None,
                   split: Optional[str] = None, verbose: bool = False) -> Optional[dict]:
    """
    Evaluate a single split, given by ground truth and predictions.

    Parameters
    ----------
    y_true: DataFrame
        Ground truth, encoded DataFrame.
    y_hat: ndarray
        Predictions array.
    encoder:
        Encoder used for encoding and decoding.
    directory: str | Path
        Directory where to save the evaluation results. If `None`, results are returned in a dict.
    main_metrics: list
        Main evaluation metrics. `None` defaults to the metrics specified in the default config.
    y_true_decoded: optinal
        Decoded ground truth for creating regression plots. If None, `encoder` is applied to decode `y_true`.
    y_hat_decoded: optional
        Decoded predictions for creating regression plots. If None, `encoder` is applied to decode`y_hat`.
    sample_weight: ndarray, optional
        Sample weights, optional. If `None`, uniform weights are used.
    threshold:
        Decision threshold for binary- and multilabel classification problems.
    static_plots:
        Whether to create static plots.
    interactive_plots:
        Whether to create interactive plots.
    bootstrapping_repetitions:
        Number of bootstrapping repetitions.
    bootstrapping_metrics:
        Names of metrics for which bootstrapped scores are computed, if `bootstrapping_repetitions` is > 0. Defaults to
        `main_metrics`. Can also be `"__all__"`, in which case all standard metrics for the current prediction task are
        computed. Ignored if `bootstrapping_repetitions` is 0.
    split:
        Name of the current split, or None. Only used for logging.
    verbose:
        Whether to log key performance metrics.

    Returns
    -------
    dict | None
        `None` if `directory` is given, else dict with evaluation results.
    """

    if directory is None:
        out = {}

        def _save(_obj, _name: str):
            out[_name] = _obj

        _save_plots = _save
    else:
        out = None
        directory = io.make_path(directory)
        directory.mkdir(exist_ok=True, parents=True)

        def _save(_obj, _name: str):
            if isinstance(_obj, pd.DataFrame):
                io.write_df(_obj, directory / (_name + '.xlsx'))
            elif isinstance(_obj, dict):
                io.write_dfs(_obj, directory / (_name + '.xlsx'))

        def _save_plots(_obj, _name: str):
            plotting.save(_obj, directory / _name)

    if main_metrics is None:
        from ..core.config import DEFAULT_CONFIG
        main_metrics = DEFAULT_CONFIG.get(encoder.task_ + '_metrics', [])

    na_mask = ~np.isnan(y_hat).any(axis=1) & y_true.notna().all(axis=1)
    if bootstrapping_repetitions > 0:
        if bootstrapping_metrics is None:
            bootstrapping_metrics = main_metrics
        elif bootstrapping_metrics == '__all__':
            bootstrapping_metrics = _get_default_metrics(encoder.task_)
    else:
        bootstrapping_metrics = None

    if encoder.task_ == 'regression':
        met = calc_regression_metrics(y_true, y_hat, sample_weight=sample_weight)
        _save(met, 'metrics')
        if verbose and '__overall__' in met.index:
            msg = ['Evaluation results' + (':' if split is None else ' for {}:'.format(split))]
            msg += ['    {}: {}'.format(m, met.loc['__overall__', m])
                    for m in main_metrics if m in met.columns]
            logging.log('\n'.join(msg))
        if static_plots:
            if y_true_decoded is None:
                y_true_decoded = encoder.inverse_transform(y=y_true, inplace=False)
            if y_hat_decoded is None:
                y_hat_decoded = encoder.inverse_transform(y=y_hat, inplace=True)
                y_hat_decoded.index = y_true_decoded.index
            _save_plots(plot_regression(y_true_decoded, y_hat_decoded, sample_weight=sample_weight, interactive=False),
                        'static_plots')
        if interactive_plots:
            if y_true_decoded is None:
                y_true_decoded = encoder.inverse_transform(y=y_true, inplace=False)
            if y_hat_decoded is None:
                y_hat_decoded = encoder.inverse_transform(y=y_hat, inplace=True)
                y_hat_decoded.index = y_true_decoded.index
            _save_plots(plot_regression(y_true_decoded, y_hat_decoded, sample_weight=sample_weight, interactive=True),
                        'interactive_plots')
        bs_summary, bs_details, _, _, _ = _bootstrap(
            bootstrapping_repetitions, y_true[na_mask], y_hat[na_mask],
            sample_weight=None if sample_weight is None else sample_weight[na_mask],
            task=encoder.task_, metric_list=bootstrapping_metrics
        )
        if bs_summary is not None:
            _save(dict(summary=bs_summary, details=bs_details), 'bootstrapping')
    elif encoder.task_ == 'multilabel_classification':
        labels = pd.DataFrame({t: encoder.get_dtype(t)['categories'] for t in encoder.target_names_})
        overall, thresh, thresh_per_class, roc_curves, pr_curves = \
            calc_multilabel_metrics(y_true, y_hat, sample_weight=sample_weight, ensure_thresholds=[threshold])
        if verbose:
            msg = ['Evaluation results' + (':' if split is None else ' for {}:'.format(split))]
            for m in main_metrics:
                if m in thresh.columns:
                    if 'threshold' in thresh.columns:
                        i = np.argmin((thresh['threshold'] - threshold).abs().values)
                        msg.append('    {} @ {}: {}'.format(m, thresh['threshold'].iloc[i], thresh[m].iloc[i]))
                elif m.endswith('_micro'):
                    if '__micro__' in overall.index and m[:-6] in overall.columns:
                        msg.append('    {}: {}'.format(m, overall.loc['__micro__', m[:-6]]))
                elif m.endswith('_macro'):
                    if '__macro__' in overall.index and m[:-6] in overall.columns:
                        msg.append('    {}: {}'.format(m, overall.loc['__macro__', m[:-6]]))
                elif m.endswith('_weighted'):
                    if '__weighted__' in overall.index and m[:-9] in overall.columns:
                        msg.append('    {}: {}'.format(m, overall.loc['__weighted__', m[:-9]]))
            if len(msg) > 1:
                logging.log('\n'.join(msg))
        if static_plots:
            _save_plots(
                plot_multilabel(overall, thresh_per_class, threshold=threshold, roc_curves=roc_curves,
                                pr_curves=pr_curves, interactive=False, labels=labels),
                'static_plots'
            )
        if interactive_plots:
            _save_plots(
                plot_multilabel(overall, thresh_per_class, threshold=threshold, roc_curves=roc_curves,
                                pr_curves=pr_curves, interactive=True, labels=labels),
                'interactive_plots'
            )
        overall.insert(0, 'pos_label', list(labels.iloc[1]) + [None] * 3)
        _save(dict(overall=overall, thresholded=thresh, **thresh_per_class), 'metrics')
        bs_summary, bs_details, _, _, _ = _bootstrap(
            bootstrapping_repetitions, y_true[na_mask], y_hat[na_mask],
            sample_weight=None if sample_weight is None else sample_weight[na_mask],
            task=encoder.task_, metric_list=bootstrapping_metrics, threshold=threshold
        )
        if bs_summary is not None:
            _save(dict(summary=bs_summary, details=bs_details), 'bootstrapping')
    else:
        labels = encoder.get_dtype(encoder.target_names_[0])['categories']
        if encoder.task_ == 'binary_classification':
            if not isinstance(threshold, float):
                threshold = threshold(y_true[na_mask].values[:, 0], y_hat[na_mask][:, -1],
                                      sample_weight=sample_weight)

            overall, thresh, calib, roc_curve, pr_curve = \
                calc_binary_classification_metrics(y_true, y_hat, sample_weight=sample_weight,
                                                   ensure_thresholds=[threshold])
            if verbose:
                msg = ['Evaluation results' + (':' if split is None else ' for {}:'.format(split))]
                for m in main_metrics:
                    if m in thresh.columns:
                        if 'threshold' in thresh.columns:
                            i = np.argmin((thresh['threshold'] - threshold).abs().values)
                            msg.append('    {} @ {}: {}'.format(m, thresh['threshold'].iloc[i], thresh[m].iloc[i]))
                    else:
                        v = overall.get(m)
                        if v is not None:
                            msg.append('    {}: {}'.format(m, v))
                if len(msg) > 1:
                    logging.log('\n'.join(msg))
            overall_df = pd.DataFrame(data=overall, index=[y_true.columns[0]])
            overall_df.insert(0, 'pos_label', labels[1])
            _save(dict(overall=overall_df, thresholded=thresh, calibration=calib), 'metrics')
            bs_summary, bs_details, roc_curve_bs, pr_curve_bs, calibration_curve_bs = _bootstrap(
                bootstrapping_repetitions,
                y_true[na_mask].iloc[:, 0],
                y_hat[na_mask, -1],
                sample_weight=None if sample_weight is None else sample_weight[na_mask],
                task=encoder.task_,
                calib=calib,
                metric_list=bootstrapping_metrics,
                threshold=threshold,
                calc_roc_pr_calibration=static_plots or interactive_plots
            )
            if bs_summary is not None:
                _save(dict(summary=bs_summary, details=bs_details), 'bootstrapping')
            if static_plots:
                _save_plots(
                    plot_binary_classification(overall, thresh, threshold=threshold, calibration=calib,
                                               interactive=False,
                                               neg_label=str(labels[0]), pos_label=str(labels[1]),
                                               roc_curve=roc_curve, pr_curve=pr_curve,
                                               roc_curve_bs=roc_curve_bs, pr_curve_bs=pr_curve_bs,
                                               calibration_curve_bs=calibration_curve_bs),
                    'static_plots'
                )
            if interactive_plots:
                _save_plots(
                    plot_binary_classification(overall, thresh, threshold=threshold, calibration=calib,
                                               interactive=True,
                                               neg_label=str(labels[0]), pos_label=str(labels[1]),
                                               roc_curve=roc_curve, pr_curve=pr_curve,
                                               roc_curve_bs=roc_curve_bs, pr_curve_bs=pr_curve_bs,
                                               calibration_curve_bs=calibration_curve_bs),
                    'interactive_plots'
                )
        else:
            overall, conf_mat, per_class = \
                calc_multiclass_metrics(y_true, y_hat, sample_weight=sample_weight, labels=labels)
            if verbose:
                msg = ['Evaluation results' + (':' if split is None else ' for {}:'.format(split))]
                for m in main_metrics:
                    v = overall.get(m)
                    if v is not None:
                        msg.append('    {}: {}'.format(m, v))
                if len(msg) > 1:
                    logging.log('\n'.join(msg))
            overall = pd.DataFrame(data=overall, index=[y_true.columns[0]])
            _save(dict(overall=overall, confusion_matrix=conf_mat, per_class=per_class), 'metrics')
            if static_plots:
                _save_plots(plot_multiclass(conf_mat, interactive=False), 'static_plots')
            if interactive_plots:
                _save_plots(plot_multiclass(conf_mat, interactive=True), 'interactive_plots')
            bs_summary, bs_details, _, _, _ = _bootstrap(
                bootstrapping_repetitions, y_true[na_mask].iloc[:, 0], y_hat[na_mask],
                sample_weight=None if sample_weight is None else sample_weight[na_mask],
                task=encoder.task_, metric_list=bootstrapping_metrics
            )
            if bs_summary is not None:
                _save(dict(summary=bs_summary, details=bs_details), 'bootstrapping')

    return out


def calc_regression_metrics(y_true: pd.DataFrame, y_hat: Union[pd.DataFrame, np.ndarray],
                            sample_weight: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Calculate all suitable regression metrics for all targets individually, and for their combination.

    Parameters
    ----------
    y_true: DataFrame
        Ground truth. All columns must have numerical data type. Entries may be NaN, in which case only
    non-NaN entries are considered.
    y_hat: DataFrame, ndarray
        Predictions. Must have the same shape as `y_true`. Entries may be NaN, in which case only non-NaN entries are
        considered.
    sample_weight: ndarray
        Sample weights. If given, must have shape `(len(y_true),)`.

    Returns
    -------
    DataFrame
        DataFrame with one column for each calculated metric, and one row for each column of `y_true` plus an extra row
        "__overall__". Note that "__overall__" is added even if `y_true` has only one column, in which case the metrics
        for that column and "__overall__" coincide.
    """
    assert y_true.shape == y_hat.shape
    if isinstance(y_hat, pd.DataFrame):
        assert (y_hat.columns == y_true.columns).all()
        y_hat = y_hat.values
    elif isinstance(y_hat, np.ndarray) and y_hat.ndim == 1:
        y_hat = y_hat.reshape((-1, 1))
    targets = y_true.columns
    y_true = y_true.values

    mask = np.isfinite(y_true) & np.isfinite(y_hat)
    out = pd.DataFrame(index=list(targets) + ['__overall__'], data=dict(n=0))
    out['n'].values[:-1] = mask.sum(axis=0)
    out.loc['__overall__', 'n'] = mask.all(axis=1).sum()

    if sample_weight is not None:
        out['n_weighted'] = (mask.all(axis=1) * sample_weight).sum()
        out['n_weighted'].values[:-1] = (mask * sample_weight[..., np.newaxis]).sum(axis=0)

    for name in _REGRESSION_METRICS:
        out[name] = np.nan
        func = metrics.get(name)
        for i, c in enumerate(targets):
            try:
                # some metrics cannot be computed if `y_true` or `y_hat` contain certain values,
                # e.g., "mean_squared_log_error" cannot be applied to negative values => skip
                out.loc[c, name] = func(y_true[mask[:, i], i], y_hat[mask[:, i], i],
                                        sample_weight=None if sample_weight is None else sample_weight[mask[:, i]])
            except:     # noqa
                pass
        try:
            out.loc['__overall__', name] = \
                func(y_true[mask.all(axis=1)], y_hat[mask.all(axis=1)],
                     sample_weight=None if sample_weight is None else sample_weight[mask.all(axis=1)])
        except:     # noqa
            pass

    # drop all-NaN columns
    out.dropna(axis=1, how='all', inplace=True)
    return out


def plot_regression(y_true: pd.DataFrame, y_hat: pd.DataFrame, sample_weight: Optional[np.ndarray] = None,
                    interactive: bool = False) -> dict:
    """
    Plot evaluation results of regression tasks.
    :param y_true: Ground truth. May be encoded or decoded, and may contain NaN values.
    :param y_hat: Predictions, with same shape, column names and data types as `y_true`.
    :param sample_weight: Sample weights.
    :param interactive: Whether to create interactive plots using the plotly backend, or static plots using the
    Matplotlib backend.
    :return: Dict mapping names to figures.
    """
    assert y_true.shape == y_hat.shape
    assert (y_true.columns == y_hat.columns).all()

    if interactive:
        if plotting.plotly_backend is None:
            logging.warn(plotting.PLOTLY_WARNING)
            return {}
        else:
            backend = plotting.plotly_backend
    else:
        backend = plotting.mpl_backend

    return {c: backend.regression_scatter(y_true[c], y_hat[c], sample_weight=sample_weight, name=c)
            for c in y_true.columns}


def calc_binary_classification_metrics(
        y_true: pd.DataFrame,
        y_hat: Union[pd.DataFrame, np.ndarray],
        sample_weight: Optional[np.ndarray] = None,
        thresholds: Optional[list] = None,
        ensure_thresholds: Optional[list] = None,
        calibration_thresholds: Optional[np.ndarray] = None) \
        -> Tuple[dict, pd.DataFrame, pd.DataFrame, Tuple[np.ndarray, np.ndarray, np.ndarray],
                 Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Calculate all metrics suitable for binary classification tasks.

    Parameters
    ----------
    y_true: DataFrame
        Ground truth. Must have 1 column with float data type and values among 0, 1 and NaN.
    y_hat: DataFrame | ndarray
        Predictions. Must have the same number of rows as `y_true` and either 1 or 2 columns.
    sample_weight: ndarray, optional
        Sample weights. If given, must have shape `(len(y_true),)`.
    thresholds: list, optional
        List of thresholds to use for thresholded metrics. If `None`, a default list of thresholds depending on the
        values of `y_hat` is constructed.
    ensure_thresholds: list, optional
        List of thresholds that must appear among the used thresholds, if `thresholds` is set to `None`. Ignored if
        `thresholds` is a list.
    calibration_thresholds: ndarray, optional
        Thresholds to use for calibration curves. If None, a default list depending on the values of `y_hat` is
        constructed.

    Returns
    -------
    Tuple
        5-tuple `(overall, threshold, calibration, roc_curve, pr_curve)`:

        * `overall` is a dict containing the scores of threshold-independent metrics (e.g., ROC-AUC).
        * `threshold` is a DataFrame with one column for each threshold-dependent metric, and one row for each decision
          threshold.
        * `calibration` is a DataFrame with one row for each threshold-bin and three columns with information about the
          corresponding bin ranges and fraction of positive samples.
        * `roc_curve` is the receiver operating characteristic curve, as returned by `sklearn.metrics.roc_curve()`.
          Although similar information is already contained in `threshold["specificity"]` and
          `threshold["sensitivity"]`, `roc_curve` is more fine-grained and better suited for plotting.
        * `pr_curve` is the precision-recall curve, as returned by `sklearn.metrics.precision_recall_curve()`.
          Although similar information is already contained in `threshold["sensitivity"]` and
          `threshold["positive_predictive_value"]`, `pr_curve` is more fine-grained and better suited for plotting.
    """
    assert y_true.shape[1] == 1
    assert len(y_true) == len(y_hat)
    if isinstance(y_hat, pd.DataFrame):
        y_hat = y_hat.values
    if y_hat.ndim == 2:
        assert y_hat.shape[1] in (1, 2)
        y_hat = y_hat[:, -1]
    else:
        assert y_hat.ndim == 1
    y_true = y_true.values[:, 0]

    mask = np.isfinite(y_true) & np.isfinite(y_hat)
    y_true = y_true[mask]
    y_hat = y_hat[mask]

    n_positive = (y_true > 0).sum()
    n_negative = (y_true < 1).sum()
    dct = dict(n=mask.sum(), n_pos=n_positive)

    if sample_weight is None:
        n_positive_weighted = n_negative_weighted = None
    else:
        sample_weight = sample_weight[mask]
        n_positive_weighted = ((y_true > 0) * sample_weight).sum()
        n_negative_weighted = ((y_true < 1) * sample_weight).sum()
        dct.update(n_weighted=sample_weight.sum(), n_pos_weighted=n_positive_weighted)

    for m in _BINARY_PROBA_METRICS:
        try:
            dct[m] = metrics.get(m)(y_true, y_hat, sample_weight=sample_weight)
        except:     # noqa
            pass

    if thresholds is None:
        thresholds = metrics.get_thresholds(y_hat, n_max=100, add_half_one=None, ensure=ensure_thresholds,
                                            sample_weight=sample_weight)
    out = pd.DataFrame(data=dict(threshold=thresholds, true_positive=0, true_negative=0))
    for i, t in enumerate(thresholds):
        out['true_positive'].values[i] = ((y_true > 0) & (y_hat >= t)).sum()
        out['true_negative'].values[i] = ((y_true < 1) & (y_hat < t)).sum()
    out['false_positive'] = n_negative - out['true_negative']
    out['false_negative'] = n_positive - out['true_positive']

    if sample_weight is not None:
        out['true_positive_weighted'] = np.zeros((len(out),), dtype=np.float32)
        out['true_negative_weighted'] = out['true_positive_weighted']
        for i, t in enumerate(thresholds):
            out['true_positive_weighted'].values[i] = (((y_true > 0) & (y_hat >= t)) * sample_weight).sum()
            out['true_negative_weighted'].values[i] = (((y_true < 1) & (y_hat < t)) * sample_weight).sum()
        out['false_positive_weighted'] = n_negative_weighted - out['true_negative_weighted']
        out['false_negative_weighted'] = n_positive_weighted - out['true_positive_weighted']

    for m in reversed(_BINARY_CLASS_METRICS):
        func = getattr(metrics, m + '_cm', None)
        if func is None:
            s = np.full((len(out),), np.nan)
            func = metrics.get(m)
            for i, t in enumerate(thresholds):
                try:
                    s[i] = func(y_true, y_hat >= t, sample_weight=sample_weight)
                except:     # noqa
                    pass
        elif sample_weight is None:
            s = func(tp=out['true_positive'], tn=out['true_negative'], fp=out['false_positive'],
                     fn=out['false_negative'], average=None)
        else:
            s = func(tp=out['true_positive_weighted'], tn=out['true_negative_weighted'],
                     fp=out['false_positive_weighted'], fn=out['false_negative_weighted'], average=None)
        out.insert(1, m, s)

    fractions, th = \
        metrics.calibration_curve(y_true, y_hat, thresholds=calibration_thresholds, sample_weight=sample_weight)
    calibration = pd.DataFrame(data=dict(threshold_lower=th[:-1], threshold_upper=th[1:], pos_fraction=fractions))

    roc_pr_curve = metrics.roc_pr_curve(y_true, y_hat, sample_weight=sample_weight)

    # drop all-NaN columns
    out.dropna(axis=1, how='all', inplace=True)
    return dct, out, calibration, roc_pr_curve[:3], roc_pr_curve[3:]


def plot_binary_classification(overall: dict, thresholded: pd.DataFrame, threshold: float = 0.5,
                               calibration: Optional[pd.DataFrame] = None, name: Optional[str] = None,
                               neg_label: str = 'negative', pos_label: str = 'positive', roc_curve=None, pr_curve=None,
                               roc_curve_bs=None, pr_curve_bs=None, calibration_curve_bs=None,
                               interactive: bool = False) -> dict:
    """
    Plot evaluation results of binary classification tasks.

    Parameters
    ----------
    overall: dict
        Overall, non-thresholded performance metrics, as returned by function `calc_binary_classification_metrics()`.
    thresholded: DataFrame
        Thresholded performance metrics, as returned by function `calc_binary_classification_metrics()`.
    threshold: float, default=0.5
        Decision threshold.
    calibration: DataFrame, optional
        Calibration curve, as returned by function `calc_binary_classification_metrics()`.
    name: str, optional
        Name of the classified variable.
    neg_label: str, default='negative'
        Name of the negative class.
    pos_label: str, default='positive'
        Name of the positive class.
    roc_curve: optional
        ROC-curve, triple `(fpr, tpr, thresholds)` or None.
    pr_curve: optional
        Precision-recall-curve, triple `(precision, recall, thresholds)` or None.
    roc_curve_bs: optional
        ROC-curves obtained via bootstrapping. None or a pair `(fpr, tpr)`, where both components are equal-length lists
        of arrays of shape `(n_thresholds,)`.
    pr_curve_bs: optional
        Precision-recall-curves obtained via bootstrapping. None or a pair `(precision, recall)`, where both components
        are equal-length lists of arrays of shape `(n_thresholds,)`.
    calibration_curve_bs: optional
        Calibration curves obtained via bootstrapping. None or a single array of shape `(n_thresholds, n_repetitions)`;
        the thresholds must agree with those in `calibration`.
    interactive: bool, default=False
        Whether to create interactive plots using the plotly backend, or static plots using the Matplotlib backend.

    Returns
    -------
    dict
        Dict mapping names to figures.
    """
    if 'n_pos_weighted' in overall:
        pos_prevalence = overall.get('n_pos_weighted', 0) / overall.get('n_weighted', 1)
    else:
        pos_prevalence = overall.get('n_pos', 0) / overall.get('n', 1)
    th_metrics = [m for m in ('accuracy', 'balanced_accuracy', 'f1', 'sensitivity', 'specificity',
                              'positive_predictive_value', 'negative_predictive_value') if m in thresholded.columns]
    cm, th = _get_confusion_matrix_from_thresholds(thresholded, threshold=threshold,
                                                   neg_label=neg_label, pos_label=pos_label)
    if interactive:
        if plotting.plotly_backend is None:
            logging.warn(plotting.PLOTLY_WARNING)
            return {}
        else:
            backend = plotting.plotly_backend
    else:
        backend = plotting.mpl_backend

    if roc_curve is None:
        roc_curve = (1 - thresholded['specificity'].values, thresholded['sensitivity'].values)
    else:
        roc_curve = roc_curve[:2]
    if roc_curve_bs is None:
        deviation_roc = None
        deviation_legend_roc = None
    else:
        auc = np.nanquantile([metrics.skl_metrics.auc(x, y) for x, y in zip(*roc_curve_bs)], q=[0.025, 0.975])
        deviation_roc = np.stack([np.interp(roc_curve[0], x, y) for x, y in zip(*roc_curve_bs)])
        np.clip(deviation_roc, 0, 1, out=deviation_roc)
        deviation_roc = np.nanquantile(deviation_roc, q=[0.025, 0.975], axis=0)
        deviation_legend_roc = '95% CI=[{:.4f}, {:.4f}]'.format(auc[0], auc[1])

    if pr_curve is None:
        pr_curve = (thresholded['sensitivity'].values, thresholded['positive_predictive_value'].values)
    else:
        pr_curve = pr_curve[1::-1]
    if pr_curve_bs is None:
        deviation_pr = None
        deviation_legend_pr = None
    else:
        auc = np.nanquantile([metrics.skl_metrics.auc(x, y) for y, x in zip(*pr_curve_bs)], q=[0.025, 0.975])
        deviation_pr = np.stack([np.interp(pr_curve[0], x[::-1], y[::-1]) for y, x in zip(*pr_curve_bs)])
        np.clip(deviation_pr, 0, 1, out=deviation_pr)
        deviation_pr = np.nanquantile(deviation_pr, q=[0.025, 0.975], axis=0)
        deviation_legend_pr = '95% CI=[{:.4f}, {:.4f}]'.format(auc[0], auc[1])

    out = dict(
        roc_curve=backend.roc_pr_curve(*roc_curve, deviation=deviation_roc, roc=True,
                                       legend='AUC={:.4f}'.format(overall.get('roc_auc', 0.5)),
                                       deviation_legend=deviation_legend_roc, name=name),
        pr_curve=backend.roc_pr_curve(*pr_curve, deviation=deviation_pr, roc=False, name=name,
                                      legend='AUC={:.4f}'.format(overall.get('pr_auc', pos_prevalence)),
                                      deviation_legend=deviation_legend_pr, positive_prevalence=pos_prevalence),
        threshold=backend.threshold_metric_curve(thresholded['threshold'], [thresholded[m] for m in th_metrics],
                                                 threshold=threshold, legend=th_metrics, name=name),
        confusion_matrix=backend.confusion_matrix(cm, title=f'Confusion Matrix @ {th}', name=name)
    )
    if calibration is not None:
        if calibration_curve_bs is None:
            deviation = None
            legend = None
            deviation_legend = None
        else:
            deviation = np.nanquantile(calibration_curve_bs, q=[0.025, 0.975], axis=0)
            legend = 'curve' if interactive else None
            deviation_legend = '95% CI'
        out['calibration'] = backend.calibration_curve(calibration['threshold_lower'], calibration['threshold_upper'],
                                                       calibration['pos_fraction'], name=name, deviation=deviation,
                                                       legend=legend, deviation_legend=deviation_legend)
    return out


def calc_multiclass_metrics(y_true: pd.DataFrame, y_hat: Union[pd.DataFrame, np.ndarray],
                            sample_weight: Optional[np.ndarray] = None,
                            labels: Optional[list] = None) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Calculate all metrics suitable for multiclass classification.

    Parameters
    ----------
    y_true: DataFrame
        Ground truth. Must have 1 column with float data type and values among NaN, 0, 1, ..., `n_classes` - 1.
    y_hat: DataFrame | ndarray
        Predicted class probabilities. Must have shape `(len(y_true), n_classes)` and values between 0 and 1.
    sample_weight: ndarray
        Sample weights.
    labels: list
        Class names.


    Returns
    --------
    Tuple
        Triple `(overall, conf_mat, per_class)`, where `overall` is a dict with overall performance metrics (accuracy,
        F1, etc.), `conf_mat` is the confusion matrix, and `per_class` is a DataFrame with per-class metrics (one row
        per class, one column per metric).
    """
    assert y_true.shape[1] == 1
    assert len(y_true) == len(y_hat)
    if isinstance(y_hat, pd.DataFrame):
        y_hat = y_hat.values
    assert y_hat.ndim == 2
    y_true = y_true.values[:, 0]
    assert not (y_true < 0).any()
    assert not (y_true >= y_hat.shape[1]).any()
    if labels is None:
        labels = list(range(y_hat.shape[1]))
    else:
        assert len(labels) == y_hat.shape[1]
        assert '__total__' not in labels
        labels = list(labels)

    mask = np.isfinite(y_true) & np.isfinite(y_hat).all(axis=1)
    y_true = y_true[mask]
    y_hat = y_hat[mask]
    y_pred = metrics.multiclass_proba_to_pred(y_hat).astype(y_true.dtype)
    if sample_weight is not None:
        sample_weight = sample_weight[mask]

    # confusion matrix
    conf_mat = metrics.confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    conf_mat = np.concatenate([conf_mat, conf_mat.sum(axis=0, keepdims=True)])
    conf_mat = pd.DataFrame(data=conf_mat, columns=labels, index=labels + ['__total__'])
    conf_mat['__total__'] = conf_mat.sum(axis=1)
    conf_mat.index.name = 'true \\ pred'        # suitable for saving as Excel file

    precision, recall, f1, support = \
        metrics.precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0,
                                                sample_weight=sample_weight)
    jaccard = metrics.jaccard(y_true, y_pred, average=None, zero_division=0, sample_weight=sample_weight)
    n = support.sum()   # weighted support

    # per-class metrics
    per_class = pd.DataFrame(
        index=labels,
        columns=_BINARY_PROBA_METRICS + _BINARY_CLASS_METRICS
    )
    for i, lbl in enumerate(labels):
        for m in _BINARY_PROBA_METRICS:
            try:
                per_class[m].values[i] = metrics.get(m)(y_true == i, y_hat[:, i], sample_weight=sample_weight)
            except:     # noqa
                pass
        for m in _BINARY_CLASS_METRICS:
            if m == 'sensitivity':
                per_class[m].values[i] = recall[i]
            elif m == 'positive_predictive_value':
                per_class[m].values[i] = precision[i]
            elif m == 'f1':
                per_class[m].values[i] = f1[i]
            elif m == 'jaccard':
                per_class[m].values[i] = jaccard[i]
            else:
                try:
                    per_class[m].values[i] = metrics.get(m)(y_true == i, y_pred == i, sample_weight=sample_weight)
                except:     # noqa
                    pass
    if sample_weight is None:
        per_class.insert(0, 'n', conf_mat['__total__'].iloc[:-1])
    else:
        per_class.insert(0, 'n_weighted', conf_mat['__total__'].iloc[:-1])
        per_class.insert(0, 'n', np.array([(y_true == i).sum() for i in range(len(labels))]))

    # overall metrics
    if sample_weight is None:
        dct = dict(n=n)
    else:
        dct = dict(n=mask.sum(), n_weighted=n)
    for name in ['accuracy', 'balanced_accuracy', 'cohen_kappa', 'matthews_correlation_coefficient']:
        try:
            dct[name] = metrics.get(name)(y_true, y_pred, sample_weight=sample_weight)
        except:  # noqa
            pass
    for name in ['roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']:
        try:
            dct[name] = metrics.get(name)(y_true, y_hat, sample_weight=sample_weight)
        except:  # noqa
            pass
    precision_micro, recall_micro, f1_micro, _ = \
        metrics.precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0,
                                                sample_weight=sample_weight)
    dct.update(
        precision_micro=precision_micro,
        precision_macro=precision.mean(),
        precision_weighted=precision.dot(support) / n,
        recall_micro=recall_micro,
        recall_macro=recall.mean(),
        recall_weighted=recall.dot(support) / n,
        f1_micro=f1_micro,
        f1_macro=f1.mean(),
        f1_weighted=f1.dot(support) / n,
        jaccard_micro=metrics.jaccard(y_true, y_pred, average='micro', zero_division=0),
        jaccard_macro=jaccard.mean(),
        jaccard_weighted=jaccard.dot(support) / n,
        mean_average_precision=per_class['average_precision'].mean()
    )

    # drop all-NaN columns
    per_class.dropna(axis=1, how='all', inplace=True)
    return dct, conf_mat, per_class


def plot_multiclass(confusion_matrix: pd.DataFrame, interactive: bool = False) -> dict:
    """
    Plot evaluation results of multiclass classification tasks.

    Parameters
    ----------
    confusion_matrix: DataFrame
        Confusion matrix, as returned by function `calc_multiclass_metrics()`.
    interactive: bool, default=False
        Whether to create interactive plots using the plotly backend, or static plots using the Matplotlib backend.

    Returns
    -------
    dict
        Dict mapping names to figures.
    """
    if interactive:
        if plotting.plotly_backend is None:
            logging.warn(plotting.PLOTLY_WARNING)
            return {}
        else:
            return dict(confusion_matrix=plotting.plotly_backend.confusion_matrix(confusion_matrix))
    else:
        return dict(confusion_matrix=plotting.mpl_backend.confusion_matrix(confusion_matrix))


def calc_multilabel_metrics(y_true: pd.DataFrame, y_hat: Union[pd.DataFrame, np.ndarray],
                            sample_weight: Optional[np.ndarray] = None, thresholds: Optional[list] = None,
                            ensure_thresholds: Optional[list] = None) \
        -> Tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict]:
    """
    Calculate all metrics suitable for multilabel classification.

    Parameters
    ----------
    y_true: DataFrame
        Ground truth. Must have `n_classes` columns with float data type and values among 0, 1 and NaN.
    y_hat: DataFrame | ndarray
        Predicted class probabilities. Must have shape `(len(y_true), n_classes)` and values between 0 and 1.
    sample_weight: ndarray, optional
        Sample weights. If given, must have shape `(len(y_true),)`.
    thresholds: list, optional
        List of thresholds to use for thresholded metrics. If None, a default list of thresholds depending on the values
        of `y_hat` is constructed.
    ensure_thresholds: list, optional
        List of thresholds that must appear among the used thresholds, if `thresholds` is set to None. Ignored if
        `thresholds` is a list.

    Returns
    -------
    Tuple
        5-tuple `(overall, threshold, threshold_per_class, roc_curves, pr_curves)`:

        * `overall` is a DataFrame containing non-thresholded metrics per class and for all classes combined
          ("__micro__", "__macro__" and "__weighted__"). Weights are the number of positive samples per class.
        * `threshold` is a DataFrame containing thresholded metrics for different thresholds for all classes combined.
        * `threshold_per_class` is a dict mapping classes to per-class thresholded metrics.
        * `roc_curves` is a dict mapping classes to receiver operating characteristic curves, as returned by
          `sklearn.metrics.roc_curve()`. Although similar information is already contained in `threshold_per_class`,
          `roc_curves` is more fine-grained and better suited for plotting.
        * `pr_curve` is a dict mapping classes to precision-recall curves, as returned by
          `sklearn.metrics.precision_recall_curve()`. Although similar information is already contained in
          `threshold_per_class`, `pr_curves` is more fine-grained and better suited for plotting.
    """
    assert len(y_true) == len(y_hat)
    if isinstance(y_hat, pd.DataFrame):
        y_hat = y_hat.values
    assert y_hat.ndim == 2
    labels = list(y_true.columns)
    y_true = y_true.values

    mask = np.isfinite(y_true) & np.isfinite(y_hat)

    n_positive = (y_true > 0).sum(axis=0)
    n_negative = (y_true < 1).sum(axis=0)
    if sample_weight is None:
        n_positive_weighted = n_negative_weighted = None
    else:
        n_positive_weighted = ((y_true > 0) * sample_weight[..., np.newaxis]).sum(axis=0)
        n_negative_weighted = ((y_true < 1) * sample_weight[..., np.newaxis]).sum(axis=0)

    dct = {}
    for i, lbl in enumerate(labels):
        dct_i = dict(n=mask[:, i].sum(), n_pos=n_positive[i])
        y_true_i = y_true[mask[:, i], i]
        y_hat_i = y_hat[mask[:, i], i]
        if sample_weight is None:
            sample_weight_i = None
        else:
            sample_weight_i = sample_weight[mask[:, i]]
            dct_i.update(n_weighted=sample_weight_i.sum(), n_pos_weighted=n_positive_weighted[i])
        for m in _BINARY_PROBA_METRICS:
            try:
                dct_i[m] = metrics.get(m)(y_true_i, y_hat_i, sample_weight=sample_weight_i)
            except:  # noqa
                pass
        dct[lbl] = dct_i

    # micro average
    mask_flat = mask.ravel()
    y_true_flat = y_true.ravel()[mask_flat]
    y_hat_flat = y_hat.ravel()[mask_flat]
    if sample_weight is None:
        sample_weight_flat = None
    else:
        sample_weight_flat = np.repeat(sample_weight, y_true.shape[1])[mask_flat]
    dct_i = dict(n=mask.sum(), n_pos=n_positive.sum())
    if sample_weight is not None:
        dct_i.update(n_weighted=(mask.sum(axis=1) * sample_weight).sum(), n_pos_weighted=n_positive_weighted.sum())
    for m in _BINARY_PROBA_METRICS:
        try:
            dct_i[m] = metrics.get(m)(y_true_flat, y_hat_flat, sample_weight=sample_weight_flat)
        except:  # noqa
            pass
    dct['__micro__'] = dct_i

    # macro and weighted average
    dct_i = dict(n=mask.any(axis=1).sum(), n_pos=(y_true > 0).any(axis=1).sum())
    if sample_weight is None:
        n_pos_col = 'n_pos'
    else:
        n_pos_col = 'n_pos_weighted'
        dct_i.update(n_weighted=(mask.any(axis=1) * sample_weight).sum(),
                     n_pos_weighted=((y_true > 0).any(axis=1) * sample_weight).sum())
    dct['__macro__'] = dct_i
    dct['__weighted__'] = dct_i
    overall = pd.DataFrame.from_dict(dct, orient='index')
    div = overall[n_pos_col].sum() - overall.loc[['__micro__', '__macro__', '__weighted__'], n_pos_col].sum()
    for c in overall.columns:
        if c not in ('n', 'n_pos', 'n_weighted', 'n_pos_weighted', 'balance_score', 'balance_threshold'):
            overall.loc['__macro__', c] = overall[c].iloc[:-3].mean()
            overall.loc['__weighted__', c] = (overall[n_pos_col].iloc[:-3] * overall[c].iloc[:-3]).sum() / div

    if thresholds is None:
        thresholds = metrics.get_thresholds(y_hat_flat, n_max=100, add_half_one=None, ensure=ensure_thresholds,
                                            sample_weight=sample_weight_flat)

    per_class = {}
    roc_curves = {}
    pr_curves = {}
    for j, lbl in enumerate(labels):
        out = pd.DataFrame(data=dict(threshold=thresholds, true_negative=0, true_positive=0))
        y_true_j = y_true[mask[:, j], j]
        y_hat_j = y_hat[mask[:, j], j]
        for i, t in enumerate(thresholds):
            out['true_positive'].values[i] = ((y_true_j > 0) & (y_hat_j >= t)).sum()
            out['true_negative'].values[i] = ((y_true_j < 1) & (y_hat_j < t)).sum()
        out['false_positive'] = n_negative[j] - out['true_negative']
        out['false_negative'] = n_positive[j] - out['true_positive']

        if sample_weight is None:
            sample_weight_j = None
        else:
            sample_weight_j = sample_weight[mask[:, j]]
            out['true_negative_weighted'] = np.zeros((len(out),), dtype=np.float32)
            out['true_positive_weighted'] = out['true_negative_weighted']
            for i, t in enumerate(thresholds):
                out['true_positive_weighted'].values[i] = (((y_true_j > 0) & (y_hat_j >= t)) * sample_weight_j).sum()
                out['true_negative_weighted'].values[i] = (((y_true_j < 1) & (y_hat_j < t)) * sample_weight_j).sum()
            out['false_positive_weighted'] = n_negative_weighted[j] - out['true_negative_weighted']
            out['false_negative_weighted'] = n_positive_weighted[j] - out['true_positive_weighted']

        for m in reversed(_BINARY_CLASS_METRICS):
            func = getattr(metrics, m + '_cm', None)
            if func is None:
                s = np.full((len(out),), np.nan)
                func = metrics.get(m)
                for i, t in enumerate(thresholds):
                    try:
                        s[i] = func(y_true_j > 0, y_hat_j >= t, sample_weight=sample_weight_j)
                    except:     # noqa
                        pass
            elif sample_weight is None:
                s = func(tp=out['true_positive'], tn=out['true_negative'], fp=out['false_positive'],
                         fn=out['false_negative'], average=None)
            else:
                s = func(tp=out['true_positive_weighted'], tn=out['true_negative_weighted'],
                         fp=out['false_positive_weighted'], fn=out['false_negative_weighted'], average=None)
            out.insert(1, m, s)

        roc_pr_curve = metrics.roc_pr_curve(y_true_j, y_hat_j, sample_weight=sample_weight_j)
        per_class[lbl] = out
        roc_curves[lbl] = roc_pr_curve[:3]
        pr_curves[lbl] = roc_pr_curve[3:]

    out = pd.DataFrame(data=dict(threshold=thresholds))
    tp = np.zeros((len(thresholds),), dtype=np.int32)
    tn = np.zeros_like(tp)
    if sample_weight is None:
        weight = 1
        fp = n_negative.sum()
        fn = n_positive.sum()
    else:
        weight = sample_weight_flat
        fp = n_negative_weighted.sum()
        fn = n_positive_weighted.sum()
    for i, t in enumerate(thresholds):
        tp[i] = (((y_true_flat > 0) & (y_hat_flat >= t)) * weight).sum()
        tn[i] = (((y_true_flat < 1) & (y_hat_flat < t)) * weight).sum()
    fp -= tn
    fn -= tp
    for m in _BINARY_CLASS_METRICS:
        func = getattr(metrics, m + '_cm', None)
        if func is None:
            out[m + '_micro'] = np.nan
            func = metrics.get(m)
            for i, t in enumerate(thresholds):
                try:
                    out[m + '_micro'].values[i] = \
                        func(y_true_flat > 0, y_hat_flat >= t, sample_weight=sample_weight_flat)
                except:  # noqa
                    pass
        else:
            out[m + '_micro'] = func(tp=tp, tn=tn, fp=fp, fn=fn, average=None)
        out[m + '_macro'] = sum(v[m] for v in per_class.values()) / len(per_class)
        out[m + '_weighted'] = sum(v[m] * overall.loc[k, n_pos_col] for k, v in per_class.items()) / div

    # drop all-NaN columns
    out.dropna(axis=1, how='all', inplace=True)
    return overall, out, per_class, roc_curves, pr_curves


def plot_multilabel(overall: pd.DataFrame, thresholded: dict, threshold: float = 0.5, labels=None, roc_curves=None,
                    pr_curves=None, interactive: bool = False) -> dict:
    """
    Plot evaluation results of multilabel classification tasks.

    Parameters
    ----------
    overall: DataFrame
        Overall, non-thresholded performance metrics, as returned by function `calc_multilabel_metrics()`.
    thresholded: dict
        Thresholded performance metrics, as returned by function `calc_multilabel_metrics()`.
    threshold: float, default=0.5
        Decision threshold.
    labels: optional
        Class names. None or a DataFrame with `n_class` columns and 2 rows.
    roc_curves: optional
        ROC-curves, dict mapping classes to triples `(fpr, tpr, thresholds)` or None.
    pr_curves: optional
        Precision-recall-curves, dict mapping classes to triples `(precision, recall, thresholds)` or None.
    interactive: bool, default=False
        Whether to create interactive plots using the plotly backend, or static plots using the Matplotlib backend.

    Returns
    -------
    dict
        Dict mapping names to figures.
    """
    out = {}
    kwargs = {}
    if roc_curves is None:
        roc_curves = {}
    if pr_curves is None:
        pr_curves = {}
    for name, th in thresholded.items():
        if labels is not None:
            kwargs = dict(neg_label=str(labels[name].iloc[0]), pos_label=str(labels[name].iloc[1]))
        out[name] = \
            plot_binary_classification(overall.loc[name].to_dict(), th, threshold=threshold, name=name,
                                       roc_curve=roc_curves.get(name), pr_curve=pr_curves.get(name),
                                       interactive=interactive, **kwargs)
    return out


def calc_metrics(predictions: Union[str, Path, pd.DataFrame], encoder: 'Encoder', threshold: float = 0.5, # noqa F821
                 bootstrapping_repetitions: int = 0, bootstrapping_metrics: Optional[list] = None,
                 sample_weight: Union[str, np.ndarray] = 'from_predictions') -> Tuple[dict, dict]:
    """
    Calculate performance metrics from raw sample-wise predictions and corresponding ground-truth.

    Parameters
    ----------
    predictions: str | Path | DataFrame
        Sample-wise predictions, as saved in "predictions.xlsx".
    encoder: Encoder
        Encoder, as saved in "encoder.json". Can be conveniently loaded by instantiating a CaTabRaLoader oject and
        calling its `get_encoder()` method.
    threshold: float, default=0.5
        Decision threshold for binary- and multilabel classification problems. In binary classification this can also be
        the name of a built-in thresholding strategy. See /doc/metrics.md for a list of built-in thresholding
        strategies.
    bootstrapping_repetitions: int, default=0
        Number of bootstrapping repetitions to perform.
    bootstrapping_metrics: list, optional
        Names of metrics for which bootstrapped scores are computed, if `bootstrapping_repetitions` is > 0. Defaults to
        the list of main metrics specified in the default config. Can also be "__all__", in which case all standard
        metrics for the current prediction task are computed. Ignored if `bootstrapping_repetitions` is 0.
    sample_weight: str | ndarray, optional
        Sample weights, one of "from_predictions" (to use sample weights stored in `predictions`), "uniform"
        (to use uniform sample weights) or an array.

    Returns
    -------
    Tuple
        Pair `(metrics, bootstrapping)`, where `metrics` is a dict whose values are DataFrames and corresponds exactly
        to what is by default saved as "metrics.xlsx" when invoking function `evaluate()`, and `bootstrapping` is a dict
        mapping keys "summary" and "details" to DataFrames or None.
    """

    y_true, y_hat, weight = _get_y_true_hat_weight_from_predictions(predictions, encoder)
    y_true = encoder.transform(y=y_true, inplace=False)
    na_mask = y_hat.notna().all(axis=1) & y_true.notna().all(axis=1)

    if isinstance(sample_weight, str):
        if sample_weight == 'from_predictions':
            sample_weight = weight
        elif sample_weight == 'uniform':
            sample_weight = None
        else:
            raise ValueError(
                f'`sample_weight` must be "from_predictions", "uniform" or an array, but found "{sample_weight}".'
            )
    elif sample_weight is None:
        raise ValueError('`sample_weight` must be "from_predictions", "uniform" or an array, but found None.')
    elif len(sample_weight) != len(y_true):
        raise ValueError(
            'Length of `sample_weight` differs from number of samples: %r' % [len(y_true), len(sample_weight)] # noqa F507
        )

    if encoder.task_ == 'regression':
        y_hat = encoder.transform(y=y_hat, inplace=False)
        met = dict(metrics=calc_regression_metrics(y_true, y_hat, sample_weight=sample_weight))
    elif encoder.task_ == 'multilabel_classification':
        overall, thresh, thresh_per_class, _, _ = \
            calc_multilabel_metrics(y_true, y_hat, ensure_thresholds=[threshold], sample_weight=sample_weight)
        overall.insert(0, 'pos_label',
                       [encoder.get_dtype(t)['categories'][1] for t in encoder.target_names_] + [None] * 3)
        met = dict(overall=overall, thresholded=thresh, **thresh_per_class)
    elif encoder.task_ == 'binary_classification':
        pos_label = encoder.get_dtype(encoder.target_names_[0])['categories'][1]
        if isinstance(threshold, str):
            threshold = metrics.get_thresholding_strategy(threshold)(y_true.iloc[:, 0], y_hat.iloc[:, -1],
                                                                     sample_weight=sample_weight)
        overall, thresh, calib, roc_curve, pr_curve = \
            calc_binary_classification_metrics(y_true, y_hat, ensure_thresholds=[threshold],
                                               sample_weight=sample_weight)
        overall_df = pd.DataFrame(data=overall, index=[y_true.columns[0]])
        overall_df.insert(0, 'pos_label', pos_label)
        met = dict(overall=overall_df, thresholded=thresh, calibration=calib)
        y_true = y_true.iloc[:, 0]
        y_hat = y_hat[pos_label]
    elif encoder.task_ == 'multiclass_classification':
        overall, conf_mat, per_class = calc_multiclass_metrics(
            y_true,
            y_hat,
            sample_weight=sample_weight,
            labels=encoder.get_dtype(encoder.target_names_[0])['categories']
        )
        overall = pd.DataFrame(data=overall, index=encoder.target_names_)
        met = dict(overall=overall, confusion_matrix=conf_mat, per_class=per_class)
        y_true = y_true.iloc[:, 0]
    else:
        raise ValueError(f'Unknown prediction task: {encoder.task_}')

    if bootstrapping_repetitions > 0:
        if bootstrapping_metrics == '__all__':
            bootstrapping_metrics = _get_default_metrics(encoder.task_)
        elif bootstrapping_metrics is None:
            from ..core.config import DEFAULT_CONFIG
            bootstrapping_metrics = DEFAULT_CONFIG.get(encoder.task_ + '_metrics', [])
        bs_summary, bs_details, _, _, _ = _bootstrap(bootstrapping_repetitions, y_true[na_mask].values,
                                                     y_hat[na_mask].values, sample_weight=sample_weight,
                                                     task=encoder.task_, metric_list=bootstrapping_metrics,
                                                     threshold=threshold)
    else:
        bs_summary = bs_details = None

    return met, dict(summary=bs_summary, details=bs_details)


def plot_results(predictions: Union[str, Path, pd.DataFrame], metrics_: Union[str, Path, pd.DataFrame, dict],
                 encoder: 'Encoder', interactive: bool = False, threshold: float = 0.5, # noqa F821
                 bootstrapping_repetitions: int = 0) -> dict:
    """
    Plot the results of an evaluation. This happens automatically if config params "static_plots" or
    "interactive_plots" are set to True. This function does not save the resulting plots to disk, but instead returns
    them in a (nested) dict. This allows one to further modify / fine-tune them before displaying or saving.

    Parameters
    ----------
    predictions: str | Path | DataFrame
        Sample-wise predictions, as saved in "predictions.xlsx".
    metrics_: str | Path | DataFrame | dict
        Performance metrics, as saved in "metrics.xlsx".
    encoder: Encoder
        Encoder, as saved in "encoder.json". Can be conveniently loaded by instantiating a CaTabRaLoader object and
        calling its `get_encoder()` method.
    interactive: bool, default=False
        Whether to create static Matplotlib plots or interactive plotly plots.
    threshold: float, default=0.5
        Decision threshold for binary- and multilabel classification problems.
    bootstrapping_repetitions: int, default=0
        Number of bootstrapping repetitions to perform for adding confidence intervals to ROC-, PR- and calibration
        curves in binary classification tasks.

    Returns
    -------
    dict
        (Nested) dict of Matplotlib or plotly figure objects, depending on the value of `interactive`.
    """
    if encoder.task_ == 'regression':
        assert predictions is not None
        y_true, y_hat, sample_weight = _get_y_true_hat_weight_from_predictions(predictions, encoder)
        return plot_regression(y_true, y_hat, sample_weight=sample_weight, interactive=interactive)
    else:
        inplace = False
        if isinstance(metrics_, (str, Path)):
            metrics_ = io.read_dfs(metrics_)
            inplace = True
        if not isinstance(metrics_, dict):
            raise ValueError('metrics must be a dict, but found type ' + str(type(metrics_)))

        if encoder.task_ == 'binary_classification':
            target_name = encoder.target_names_[0]
            labels = encoder.get_dtype(target_name)['categories']
            calib = metrics_['calibration']
            if predictions is None:
                roc_curve = pr_curve = roc_curve_bs = pr_curve_bs = calibration_bs = None
            else:
                y_true, y_hat, sample_weight = _get_y_true_hat_weight_from_predictions(predictions, encoder)
                y_true = encoder.transform(y=y_true, inplace=False).iloc[:, 0]
                y_hat = y_hat[labels[1]]
                roc_pr_curve = metrics.roc_pr_curve(y_true, y_hat, sample_weight=sample_weight)
                roc_curve = roc_pr_curve[:3]
                pr_curve = roc_pr_curve[3:]
                na_mask = y_hat.notna() & y_true.notna()
                _, _, roc_curve_bs, pr_curve_bs, calibration_bs = \
                    _bootstrap(bootstrapping_repetitions, y_true[na_mask], y_hat[na_mask],
                               calib=calib, calc_roc_pr_calibration=True,
                               sample_weight=None if sample_weight is None else sample_weight[na_mask])
            return plot_binary_classification(
                metrics_['overall'].iloc[0].to_dict(),
                metrics_['thresholded'],
                threshold=threshold,
                calibration=calib,
                name=target_name,
                neg_label=labels[0],
                pos_label=labels[1],
                roc_curve=roc_curve,
                pr_curve=pr_curve,
                roc_curve_bs=roc_curve_bs,
                pr_curve_bs=pr_curve_bs,
                calibration_curve_bs=calibration_bs,
                interactive=interactive
            )
        elif encoder.task_ == 'multiclass_classification':
            conf_mat = metrics_['confusion_matrix']
            if conf_mat.index.name is None and conf_mat.columns[0] == 'true \\ pred':
                if inplace:
                    conf_mat.set_index(conf_mat.columns[0], inplace=True)
                else:
                    conf_mat = conf_mat.set_index(conf_mat.columns[0], inplace=False)
            return plot_multiclass(conf_mat, interactive=interactive)
        elif encoder.task_ == 'multilabel_classification':
            labels = pd.DataFrame({t: encoder.get_dtype(t)['categories'] for t in encoder.target_names_})
            overall = metrics_['overall']
            if overall.index.name is None and overall.index.dtype.kind != 'O':
                if inplace:
                    overall.set_index(overall.columns[0], inplace=True)
                else:
                    overall = overall.set_index(overall.columns[0], inplace=False)
            if predictions is None:
                roc_pr_curves = {}
            else:
                y_true, y_hat, sample_weight = _get_y_true_hat_weight_from_predictions(predictions, encoder)
                y_true = encoder.transform(y=y_true, inplace=False)
                roc_pr_curves = {k: metrics.roc_pr_curve(y_true[k], y_hat[k], sample_weight=sample_weight)
                                 for k in labels.columns}
            return plot_multilabel(
                overall,
                {k: metrics_[k] for k in labels.columns},
                threshold=threshold,
                roc_curves={k: v[:3] for k, v in roc_pr_curves.items()},
                pr_curves={k: v[3:] for k, v in roc_pr_curves.items()},
                labels=labels,
                interactive=interactive
            )
        else:
            raise ValueError(f'Unknown prediction task: {encoder.task_}')


def performance_summary(*args, sample_weight: Optional[np.ndarray] = None, task: str = None, metric_list=None,
                        threshold: float = 0.5, add_na_results: bool = True) \
        -> Union[dict, Callable[[np.ndarray, np.ndarray], dict]]:
    """
    Summarize the quality of predictions by evaluating a given set of performance metrics. In contrast to functions
    `calc_regression_metrics()` etc., this function is more "light-weight", for instance in the sense that it only
    computes binary classification metrics for one particular threshold and only returns aggregate results over all
    classes in case of multiclass and multilabel classification. It can also be efficiently combined with bootstrapping.

    Parameters
    ----------
    *args:
        Ground truth and predictions, array-like of shape `(n_samples,)` or `(n_samples, n_targets)`. Should not contain
        NA values. Predictions must contain class probabilities rather than classes in case of classification tasks.
        Either none or both must be specified.
    sample_weight: ndarray, optional
        Sample weight, None or array-like of shape `(n_samples,)`.
    task: str, optional
        Prediction task.
    metric_list: list, optional
        List of metrics to evaluate. Metrics that do not fit to the given prediction task are tacitly skipped.
    threshold: float, default=0.5
        Decision threshold for binary- and multilabel classification problems.
    add_na_results: bool, default=True
        Whether to add N/A results in the output, e.g., if one of the requested metrics cannot be calculated. If False,
        these metrics are tacitly skipped.

    Returns
    -------
    dict | Callable
        If `args` is pair `(y_true, y_hat)`: dict whose keys are names of metrics and whose values are the results of
        the respective metrics evaluated on `y_true` and `y_hat`. Otherwise, if `args` is empty, callable which can be
        applied to `y_true` and `y_hat` (and optionally `sample_weight`).
    """

    if len(args) == 0:
        assert sample_weight is None
    else:
        assert len(args) == 2

    default_mapping = []
    cm_mapping = []
    cm_kwargs = dict(average=None) if task == 'multilabel_classification' else {}
    for m in metric_list:
        if not isinstance(m, str):
            default_mapping.append((m.__name__, m))
        elif m.endswith('_cm'):
            cm_mapping.append((m, metrics.get(m), cm_kwargs))
        elif any(m.endswith(suffix) for suffix in ('_micro', '_macro', '_samples', '_weighted')):
            if task in ('binary_classification', 'multilabel_classification'):
                i = m.rfind('_')
                try:
                    func = metrics.get(m[:i] + '_cm')
                    cm_mapping.append((m, func, dict(average=m[i + 1:])))
                except:     # noqa
                    default_mapping.append((m, metrics.maybe_thresholded(metrics.get(m), threshold=threshold)))
            elif task == 'multiclass_classification':
                default_mapping.append((m, metrics.maybe_thresholded(metrics.get(m))))
            else:
                default_mapping.append((m, metrics.get(m)))
        elif task in ('binary_classification', 'multilabel_classification'):
            try:
                cm_mapping.append((m, metrics.get(m + '_cm'), cm_kwargs))
            except:     # noqa
                default_mapping.append((m, metrics.maybe_thresholded(metrics.get(m), threshold=threshold)))
        elif task == 'multiclass_classification':
            default_mapping.append((m, metrics.maybe_thresholded(metrics.get(m))))
        else:
            default_mapping.append((m, metrics.get(m)))

    def _evaluator(y_true, y_hat, sample_weight=None) -> dict:        # noqa
        out = {}
        for n, f in default_mapping:
            try:
                # some metrics cannot be computed if `y_true` or `y_hat` contain certain values,
                # e.g., "mean_squared_log_error" cannot be applied to negative values => skip
                out[n] = f(y_true, y_hat, sample_weight=sample_weight)
            except:     # noqa
                if add_na_results:
                    out[n] = np.nan

        if cm_mapping:
            if sample_weight is None:
                w = 1
            else:
                w = sample_weight.reshape((-1, *([1] * (y_hat.ndim - 1))))      # make `sample_weight` broadcast-able
            tp = (((y_hat >= threshold) & (y_true > 0)) * w).sum(axis=0)
            fp = (((y_hat >= threshold) & (y_true < 1)) * w).sum(axis=0)
            tn = (((y_hat < threshold) & (y_true < 1)) * w).sum(axis=0)
            fn = (((y_hat < threshold) & (y_true > 0)) * w).sum(axis=0)

            for n, f, kwargs in cm_mapping:
                try:
                    out[n] = f(tp=tp, fp=fp, tn=tn, fn=fn, **kwargs)
                except:  # noqa
                    if add_na_results:
                        out[n] = np.nan

        return out

    if len(args) == 2:
        return _evaluator(*args, sample_weight=sample_weight)
    else:
        return _evaluator


def _get_confusion_matrix_from_thresholds(thresholds: pd.DataFrame, threshold: float = 0.5, neg_label: str = 'negative',
                                          pos_label: str = 'positive') -> Tuple[pd.DataFrame, float]:
    i = (thresholds['threshold'] - threshold).abs().idxmin()
    if 'true_negative_weighted' in thresholds.columns:
        cm = pd.DataFrame(
            data={
                neg_label: [thresholds.loc[i, 'true_negative_weighted'], thresholds.loc[i, 'false_negative_weighted']],
                pos_label: [thresholds.loc[i, 'false_positive_weighted'], thresholds.loc[i, 'true_positive_weighted']]
            },
            index=[neg_label, pos_label]
        )
    else:
        cm = pd.DataFrame(
            data={neg_label: [thresholds.loc[i, 'true_negative'], thresholds.loc[i, 'false_negative']],
                  pos_label: [thresholds.loc[i, 'false_positive'], thresholds.loc[i, 'true_positive']]},
            index=[neg_label, pos_label]
        )
    return cm, thresholds.loc[i, 'threshold']


def _bootstrap(n_repetitions: int, y_true, y_hat, sample_weight: Optional[np.ndarray] = None,
               task: str = None, metric_list: Optional[list] = None, threshold: float = 0.5,
               calib: Optional[pd.DataFrame] = None, calc_roc_pr_calibration: bool = False):
    if n_repetitions > 0:
        if isinstance(metric_list, str):
            metric_list = [metric_list]
        if metric_list:
            bootstrapping_fn = dict(main=performance_summary(task=task, metric_list=metric_list, threshold=threshold))
        else:
            bootstrapping_fn = {}
        if calc_roc_pr_calibration:
            bootstrapping_fn['roc_pr_curve'] = metrics.roc_pr_curve
            if calib is not None:
                bootstrapping_fn['calibration_curve'] = partial(
                    metrics.calibration_curve,
                    thresholds=np.r_[calib['threshold_lower'].values[0], calib['threshold_upper'].values]
                )
        bs = Bootstrapping(y_true, y_hat, fn=bootstrapping_fn,
                           kwargs=None if sample_weight is None else dict(sample_weight=sample_weight))
        bs.run(n_repetitions)
        if calc_roc_pr_calibration:
            aux = bs.results.pop('roc_pr_curve', None)
        else:
            aux = None
        if aux is None:
            roc_curve_bs = pr_curve_bs = None
        else:
            roc_curve_bs = aux[:2]
            pr_curve_bs = aux[3:5]
        if calc_roc_pr_calibration:
            aux = bs.results.pop('calibration_curve', None)
        else:
            aux = None
        if aux is None:
            calibration_bs = None
        else:
            calibration_bs = np.stack(aux[0])
        if 'main' in bs.results:
            df = pd.DataFrame(bs.results['main'])
            summary = df.describe()
            if task in ('binary_classification', 'multilabel_classification'):
                summary['__threshold'] = threshold
                summary.loc['count', '__threshold'] = n_repetitions
                summary.loc['std', '__threshold'] = 0
        else:
            df = bs.dataframe()
            if df is None:
                summary = None
            else:
                summary = df.describe()
        if df is not None:
            df['__seed'] = bs.seeds
        return summary, df, roc_curve_bs, pr_curve_bs, calibration_bs
    else:
        return None, None, None, None, None


def _get_y_true_hat_weight_from_predictions(
        predictions: Union[pd.DataFrame, str, Path],
        encoder: 'Encoder' # noqa F821
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[np.ndarray]]:
    # return _decoded_ `y_true` and `y_hat`

    inplace = False
    if isinstance(predictions, (str, Path)):
        from ..util import table
        predictions = io.read_df(predictions)
        predictions = table.convert_object_dtypes(predictions, inplace=True)
        inplace = True

    target_names = encoder.target_names_
    if encoder.task_ == 'regression':
        if predictions.columns[0] not in (target_names + [f'{t}_pred' for t in target_names]) \
                and predictions[predictions.columns[0]].dtype.kind in 'iuO':
            if inplace:
                predictions.set_index(predictions.columns[0], inplace=True)
            else:
                predictions = predictions.set_index(predictions.columns[0], inplace=False)
            if predictions.index.name == 'Unnamed: 0':
                predictions.index.name = None
        y_true = predictions[target_names]
        y_hat = predictions[[f'{t}_pred' for t in target_names]]
        y_hat.columns = target_names
    elif encoder.task_ == 'multilabel_classification':
        y_true = predictions[target_names]
        y_hat = predictions[[f'{t}_proba' for t in target_names]]
        y_hat.columns = target_names
    else:
        assert len(target_names) == 1
        labels = encoder.get_dtype(target_names[0])['categories']
        y_true = predictions[target_names]
        y_hat = predictions[[f'{lbl}_proba' for lbl in labels]]
        y_hat.columns = labels

    if '__sample_weight' in predictions.columns:
        sample_weight = predictions['__sample_weight'].values
    else:
        sample_weight = None

    return y_true, y_hat, sample_weight


def _get_default_metrics(task: str, average: str = 'macro') -> list:
    if task == 'regression':
        return _REGRESSION_METRICS
    elif task == 'binary_classification':
        return _BINARY_PROBA_METRICS + _BINARY_CLASS_METRICS
    elif task == 'multiclass_classification':
        return ['roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted', 'accuracy',
                'balanced_accuracy', 'cohen_kappa', 'matthews_correlation_coefficient'] + \
               [m + '_' + average for m in ('precision', 'recall', 'f1', 'jaccard')]
    elif task == 'multilabel_classification':
        return [m + '_' + average for m in _BINARY_PROBA_METRICS + _BINARY_CLASS_METRICS]
    return []


_REGRESSION_METRICS = ['r2', 'mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error',
                       'mean_squared_log_error', 'median_absolute_error', 'mean_absolute_percentage_error',
                       'max_error', 'explained_variance', 'mean_poisson_deviance', 'mean_gamma_deviance']
# metrics for binary classification, which require probabilities of positive class
_BINARY_PROBA_METRICS = ['roc_auc', 'average_precision', 'pr_auc', 'brier_loss', 'hinge_loss', 'log_loss']
# metrics for binary classification, which require predicted classes
_BINARY_CLASS_METRICS = ['accuracy', 'balanced_accuracy', 'f1', 'sensitivity', 'specificity',
                         'positive_predictive_value', 'negative_predictive_value', 'cohen_kappa', 'hamming_loss',
                         'jaccard']
