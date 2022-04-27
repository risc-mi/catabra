from typing import Union, Optional, Tuple
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

from ..util import table as tu
from ..util import io
from ..util import logging
from ..util import plotting
from ..util.encoding import Encoder
from ..util import metrics
from ..util import statistics


def evaluate(*table: Union[str, Path, pd.DataFrame], folder: Union[str, Path] = None, model_id=None,
             split: Optional[str] = None, out: Union[str, Path, None] = None, jobs: Optional[int] = None,
             batch_size: Optional[int] = None, from_invocation: Union[str, Path, dict, None] = None):
    """
    Evaluate an existing CaTabRa object (OOD-detector, prediction model, ...) on held-out test data.
    :param table: The table(s) to evaluate the CaTabRa object on. If multiple are given, their columns are merged into
    a single table. Must have the same format as the table(s) initially passed to function `analyze()`.
    :param folder: The folder containing the CaTabRa object to evaluate.
    :param model_id: Optional, ID of the prediction model to evaluate. If None or "__ensemble__", the sole trained
    model or the entire ensemble are evaluated.
    :param split: Optional, column used for splitting the data into disjoint subsets. If specified and not "", each
    subset is evaluated individually. In contrast to function `analyze()`, the name/values of the column do not need to
    carry any semantic information about training and test sets.
    :param out: Optional, directory where to save all generated artifacts. Defaults to a directory located in `folder`,
    with a name following a fixed naming pattern. If `out` already exists, the user is prompted to specify whether it
    should be replaced; otherwise, it is automatically created.
    :param jobs: Optional, number of jobs to use. Overwrites the "jobs" config param.
    :param batch_size: Optional, batch size used for applying the prediction model.
    :param from_invocation: Optional, dict or path to an invocation.json file. All arguments of this function not
    explicitly specified are taken from this dict; this also includes the table to analyze.
    """

    if isinstance(from_invocation, (str, Path)):
        from_invocation = io.load(from_invocation)
    if isinstance(from_invocation, dict):
        if len(table) == 0:
            table = from_invocation.get('table') or []
            if '<DataFrame>' in table:
                raise ValueError('Invocations must not contain "<DataFrame>" tables.')
        if folder is None:
            folder = from_invocation.get('folder')
        if model_id is None:
            model_id = from_invocation.get('model_id')
        if split is None:
            split = from_invocation.get('split')
        if out is None:
            out = from_invocation.get('out')
        if jobs is None:
            jobs = from_invocation.get('jobs')

    if len(table) == 0:
        raise ValueError('No table specified.')
    if folder is None:
        raise ValueError('No folder specified.')
    else:
        folder = io.make_path(folder, absolute=True)
    if not folder.exists():
        raise ValueError(f'Folder "{folder.as_posix()}" does not exist.')

    config = io.load(folder / 'config.json')

    start = pd.Timestamp.now()
    table = [io.make_path(tbl, absolute=True) if isinstance(tbl, (str, Path)) else tbl for tbl in table]

    if out is None:
        out = table[0]
        if isinstance(out, pd.DataFrame):
            out = folder / ('eval_' + start.strftime('%Y-%m-%d_%H-%M-%S'))
        else:
            out = folder / ('eval_' + out.stem + '_' + start.strftime('%Y-%m-%d_%H-%M-%S'))
    else:
        out = io.make_path(out, absolute=True)
    if out == folder:
        raise ValueError(f'Output directory must differ from CaTabRa directory, but both are "{out.as_posix()}".')
    elif out.exists():
        if logging.prompt(f'Evaluation folder "{out.as_posix()}" already exists. Delete?',
                          accepted=['y', 'n'], allow_headless=False) == 'y':
            if out.is_dir():
                shutil.rmtree(out.as_posix())
            else:
                out.unlink()
        else:
            logging.log('### Aborting')
            return
    out.mkdir(parents=True)

    if split == '':
        split = None

    with logging.LogMirror((out / 'console.txt').as_posix()):
        logging.log(f'### Evaluation started at {start}')
        invocation = dict(
            table=['<DataFrame>' if isinstance(tbl, pd.DataFrame) else tbl for tbl in table],
            folder=folder,
            model_id=model_id,
            split=split,
            out=out,
            jobs=jobs,
            timestamp=start
        )
        io.dump(io.to_json(invocation), out / 'invocation.json')

        # merge tables
        df, _ = tu.merge_tables(table)
        if df.columns.nlevels != 1:
            raise ValueError(f'Table must have 1 column level, but found {df.columns.nlevels}.')

        # copy test data
        copy_data = config.get('copy_evaluation_data', False)
        if isinstance(copy_data, (int, float)):
            copy_data = df.memory_usage(index=True, deep=True).sum() <= copy_data * 1000000
        if copy_data:
            io.write_df(df, out / 'test_data.h5')

        # split
        if split is None:
            def _iter_splits():
                yield np.ones((len(df),), dtype=np.bool), out
        else:
            split_masks, _ = tu.train_test_split(df, split)

            def _iter_splits():
                for _k, _m in split_masks.items():
                    yield _m, out / _k

        encoder = Encoder.load(folder / 'encoder.json')
        x_test, y_test = encoder.transform(data=df)

        static_plots = config.get('static_plots', True)
        interactive_plots = config.get('interactive_plots', False)
        if interactive_plots and plotting.plotly_backend is None:
            logging.warn(plotting.PLOTLY_WARNING)
            interactive_plots = False

        # TODO: Perform OOD checks.

        # descriptive statistics for each train/test split
        if encoder.task_ is not None and (folder / 'invocation.json').exists():
            group_ = io.load(folder / 'invocation.json').get('group')
            group_ = [] if group_ is None else [group_]
            ignore_ = io.load(folder / 'invocation.json').get('ignore')
            ignore_ = [] if ignore_ is None else ignore_
            for mask, directory in _iter_splits():
                statistics.save_descriptive_statistics(df=df[mask].drop(columns=[split]+group_+ignore_).copy(),
                                                       target=list(y_test.columns),
                                                       classify=encoder.task_ != 'regression',
                                                       fn=folder / 'statistics' / directory.name)

        if encoder.task_ is not None and (folder / 'model.joblib').exists():
            model = io.load(folder / 'model.joblib')
            if model_id == '__ensemble__':
                model_id = None
            if encoder.task_ == 'regression':
                y_hat = model.predict(x_test, jobs=jobs, batch_size=batch_size, model_id=model_id)

                # decoded ground truth and predictions for each target
                y_test_decoded = encoder.inverse_transform(y=y_test, inplace=False)
                y_hat_decoded = encoder.inverse_transform(y=y_hat, inplace=True)
                y_hat_decoded.index = y_test_decoded.index

                for mask, directory in _iter_splits():
                    directory.mkdir(exist_ok=True, parents=True)
                    io.write_df(calc_regression_metrics(y_test[mask], y_hat[mask]), directory / 'metrics.xlsx')
                    if static_plots:
                        plotting.save(plot_regression(y_test_decoded[mask], y_hat_decoded[mask], interactive=False),
                                      directory / 'static_plots')
                    if interactive_plots:
                        plotting.save(plot_regression(y_test_decoded[mask], y_hat_decoded[mask], interactive=True),
                                      directory / 'interactive_plots')

                y_hat_decoded.columns = [f'{c}_pred' for c in y_hat_decoded.columns]
                detailed = y_test_decoded.join(y_hat_decoded)
                detailed = detailed.reindex(
                    [n for c in zip(y_test_decoded.columns, y_hat_decoded.columns) for n in c],
                    axis=1
                )
                del y_test_decoded
                del y_hat_decoded

                for mask, directory in _iter_splits():
                    io.write_df(detailed[mask], directory / 'predictions.xlsx')
            else:
                y_hat = model.predict_proba(x_test, jobs=jobs, batch_size=batch_size, model_id=model_id)
                if encoder.task_ == 'multilabel_classification':
                    # decoded ground truth and predictions for each target
                    y_test_decoded = encoder.inverse_transform(y=y_test, inplace=False)
                    y_hat_decoded = encoder.inverse_transform(y=y_hat, inplace=True)
                    y_hat_decoded.index = y_test_decoded.index
                    y_hat_decoded.columns = [f'{c}_proba' for c in y_hat_decoded.columns]
                    detailed = y_test_decoded.join(y_hat_decoded)
                    detailed = detailed.reindex(
                        [n for c in zip(y_test_decoded.columns, y_hat_decoded.columns) for n in c],
                        axis=1
                    )
                    del y_test_decoded
                    del y_hat_decoded

                    labels = encoder.inverse_transform(
                        y=np.vstack([np.zeros((len(y_test.columns),)), np.ones((len(y_test.columns),))])
                    )
                    for mask, directory in _iter_splits():
                        directory.mkdir(exist_ok=True, parents=True)
                        io.write_df(detailed[mask], directory / 'predictions.xlsx')
                        overall, thresh, thresh_per_class = calc_multilabel_metrics(y_test[mask], y_hat[mask])
                        if static_plots:
                            plotting.save(
                                plot_multilabel(overall, thresh_per_class, interactive=False, labels=labels),
                                directory / 'static_plots'
                            )
                        if interactive_plots:
                            plotting.save(
                                plot_multilabel(overall, thresh_per_class, interactive=True, labels=labels),
                                directory / 'interactive_plots'
                            )
                        overall.insert(0, 'pos_label', list(labels.iloc[1]) + [None] * 3)
                        io.write_dfs(dict(overall=overall, thresholded=thresh, **thresh_per_class),
                                     directory / 'metrics.xlsx')
                else:
                    # decoded ground truth and predictions for each target
                    detailed = encoder.inverse_transform(y=y_test, inplace=False)
                    detailed[detailed.columns[0] + '_pred'] = \
                        encoder.inverse_transform(y=np.argmax(y_hat, axis=1)).iloc[:, 0]
                    y_hat_decoded = encoder.inverse_transform(y=y_hat, inplace=True)
                    y_hat_decoded.index = detailed.index
                    y_hat_decoded.columns = [f'{c}_proba' for c in y_hat_decoded.columns]
                    detailed = detailed.join(y_hat_decoded)
                    del y_hat_decoded
                    mask = np.isfinite(y_test.values[:, 0])
                    detailed['__true_proba'] = np.nan
                    detailed.loc[mask, '__true_proba'] = \
                        y_hat[mask][np.arange(mask.sum()), y_test.values[mask, 0].astype(np.int32)]
                    detailed['__true_rank'] = (y_hat > detailed['__true_proba'].values[..., np.newaxis]).sum(axis=1) + 1
                    detailed.loc[~mask, '__true_rank'] = -1

                    labels = list(encoder.inverse_transform(y=np.arange(y_hat.shape[1])).iloc[:, 0])
                    if encoder.task_ == 'binary_classification':
                        for mask, directory in _iter_splits():
                            directory.mkdir(exist_ok=True, parents=True)
                            io.write_df(detailed[mask], directory / 'predictions.xlsx')
                            overall, thresh, calib = calc_binary_classification_metrics(y_test[mask], y_hat[mask])
                            if static_plots:
                                plotting.save(
                                    plot_binary_classification(overall, thresh, calibration=calib, interactive=False,
                                                               neg_label=str(labels[0]), pos_label=str(labels[1])),
                                    directory / 'static_plots'
                                )
                            if interactive_plots:
                                plotting.save(
                                    plot_binary_classification(overall, thresh, calibration=calib, interactive=True,
                                                               neg_label=str(labels[0]), pos_label=str(labels[1])),
                                    directory / 'interactive_plots'
                                )
                            overall = pd.DataFrame(data=overall, index=[y_test.columns[0]])
                            overall.insert(0, 'pos_label', labels[1])
                            io.write_dfs(dict(overall=overall, thresholded=thresh, calibration=calib),
                                         directory / 'metrics.xlsx')
                    else:
                        for mask, directory in _iter_splits():
                            directory.mkdir(exist_ok=True, parents=True)
                            io.write_df(detailed[mask], directory / 'predictions.xlsx')
                            overall, conf_mat, per_class = calc_multiclass_metrics(
                                y_test[mask],
                                y_hat[mask],
                                labels=labels
                            )
                            overall = pd.DataFrame(data=overall, index=[y_test.columns[0]])
                            io.write_dfs(dict(overall=overall, confusion_matrix=conf_mat, per_class=per_class),
                                         directory / 'metrics.xlsx')
                            if static_plots:
                                plotting.save(plot_multiclass(conf_mat, interactive=False),
                                              directory / 'static_plots')
                            if interactive_plots:
                                plotting.save(plot_multiclass(conf_mat, interactive=True),
                                              directory / 'interactive_plots')

        end = pd.Timestamp.now()
        logging.log(f'### Evaluation finished at {end}')
        logging.log(f'### Elapsed time: {end - start}')
        logging.log(f'### Output saved in {out.as_posix()}')


def calc_regression_metrics(y_true: pd.DataFrame, y_hat: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
    """
    Calculate all suitable regression metrics for all targets individually, and for their combination.
    :param y_true: Ground truth. All columns must have numerical data type. Entries may be NaN, in which case only
    non-NaN entries are considered.
    :param y_hat: Predictions. Must have the same shape as `y_true`. Entries may be NaN, in which case only non-NaN
    entries are considered.
    :return: DataFrame with one column for each calculated metric, and one row for each column of `y_true` plus an
    extra row "__overall__". Note that "__overall__" is added even if `y_true` has only one column, in which case the
    metrics for that column and "__overall__" coincide.
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

    for name in ['r2', 'mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error', 'mean_squared_log_error',
                 'median_absolute_error', 'mean_absolute_percentage_error', 'max_error', 'explained_variance',
                 'mean_poisson_deviance', 'mean_gamma_deviance']:
        out[name] = np.nan
        func = getattr(metrics, name)
        for i, c in enumerate(targets):
            try:
                # some metrics cannot be computed if `y_true` or `y_hat` contain certain values,
                # e.g., "mean_squared_log_error" cannot be applied to negative values => skip
                out.loc[c, name] = func(y_true[mask[:, i], i], y_hat[mask[:, i], i])
            except:     # noqa
                pass
        try:
            out.loc['__overall__', name] = func(y_true[mask.all(axis=1)], y_hat[mask.all(axis=1)])
        except:     # noqa
            pass

    # drop all-NaN columns
    out.dropna(axis=1, how='all', inplace=True)
    return out


def plot_regression(y_true: pd.DataFrame, y_hat: pd.DataFrame, interactive: bool = False) -> dict:
    """
    Plot evaluation results of regression tasks.
    :param y_true: Ground truth. May be encoded or decoded, and may contain NaN values.
    :param y_hat: Predictions, with same shape, column names and data types as `y_true`.
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
            return {c: plotting.plotly_backend.regression_scatter(y_true[c], y_hat[c], name=c) for c in y_true.columns}
    else:
        return {c: plotting.mpl_backend.regression_scatter(y_true[c], y_hat[c], name=c) for c in y_true.columns}


def calc_binary_classification_metrics(
        y_true: pd.DataFrame,
        y_hat: Union[pd.DataFrame, np.ndarray],
        thresholds: Optional[list] = None,
        calibration_thresholds: Optional[np.ndarray] = None) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Calculate all metrics suitable for binary classification tasks.
    :param y_true: Ground truth. Must have 1 column with float data type and values among 0, 1 and NaN.
    :param y_hat: Predictions. Must have the same number of rows as `y_true` and either 1 or 2 columns.
    :param thresholds: List of thresholds to use for thresholded metrics. If None, a default list of thresholds
    depending on the values of `y_hat` is constructed.
    :param calibration_thresholds: Thresholds to use for calibration curves. If None, a default list depending on the
    values of `y_hat` is constructed.
    :return: Triple `(overall, threshold, calibration)`:
    * `overall` is a dict containing the scores of threshold-independent metrics (e.g., ROC-AUC).
    * `threshold` is a DataFrame with one column for each threshold-dependent metric, and one row for each decision
        threshold.
    * `calibration` is a DataFrame with one row for each threshold-bin and three columns with information about the
        corresponding bin ranges and fraction of positive samples.
    """
    assert y_true.shape[1] == 1
    assert len(y_true) == len(y_hat)
    if isinstance(y_hat, pd.DataFrame):
        y_hat = y_hat.values
    if y_hat.ndim == 2:
        assert y_hat.shape[1] in (1, 2)
        y_hat = y_hat[:, 1]
    else:
        assert y_hat.ndim == 1
    y_true = y_true.values[:, 0]

    mask = np.isfinite(y_true) & np.isfinite(y_hat)
    y_true = y_true[mask]
    y_hat = y_hat[mask]

    n_positive = (y_true > 0).sum()
    n_negative = (y_true < 1).sum()
    dct = dict(n=mask.sum(), n_pos=n_positive)
    for m in _BINARY_PROBA_METRICS:
        try:
            dct[m] = getattr(metrics, m)(y_true, y_hat)
        except:     # noqa
            pass

    if thresholds is None:
        thresholds = metrics.get_thresholds(y_hat, n_max=100, add_half_one=None)
    out = pd.DataFrame(data=dict(threshold=thresholds, true_positive=0, true_negative=0))
    for i, t in enumerate(thresholds):
        out['true_positive'].values[i] = ((y_true > 0) & (y_hat >= t)).sum()
        out['true_negative'].values[i] = ((y_true < 1) & (y_hat < t)).sum()
    out['false_positive'] = n_negative - out['true_negative']
    out['false_negative'] = n_positive - out['true_positive']

    for m in reversed(_BINARY_CLASS_METRICS):
        func = getattr(metrics, m + '_cm', None)
        if func is None:
            s = np.full((len(out),), np.nan)
            func = getattr(metrics, m)
            for i, t in enumerate(thresholds):
                try:
                    s[i] = func(y_true, y_hat >= t)
                except:     # noqa
                    pass
        else:
            s = func(tp=out['true_positive'], tn=out['true_negative'], fp=out['false_positive'],
                     fn=out['false_negative'], average=None)
        out.insert(1, m, s)

    if 'sensitivity' in out.columns and 'specificity' in out.columns:
        i = (out['sensitivity'] - out['specificity']).abs().idxmin()
        dct['balance_threshold'] = out.loc[i, 'threshold']
        dct['balance_score'] = (out.loc[i, 'sensitivity'] + out.loc[i, 'specificity']) * 0.5

    fractions, th = metrics.calibration_curve(y_true, y_hat, thresholds=calibration_thresholds)
    calibration = pd.DataFrame(data=dict(threshold_lower=th[:-1], threshold_upper=th[1:], pos_fraction=fractions))

    # drop all-NaN columns
    out.dropna(axis=1, how='all', inplace=True)
    return dct, out, calibration


def plot_binary_classification(overall: dict, threshold: pd.DataFrame, calibration: Optional[pd.DataFrame] = None,
                               name: Optional[str] = None, neg_label: str = 'negative', pos_label: str = 'positive',
                               interactive: bool = False) -> dict:
    """
    Plot evaluation results of binary classification tasks.
    :param overall: Overall, non-thresholded performance metrics, as returned by function
    `calc_binary_classification_metrics()`.
    :param threshold: Thresholded performance metrics, as returned by function `calc_binary_classification_metrics()`.
    :param calibration: Calibration curve, as returned by function `calc_binary_classification_metrics()`.
    :param name: Name of the classified variable.
    :param neg_label: Name of the negative class.
    :param pos_label: Name of the positive class.
    :param interactive: Whether to create interactive plots using the plotly backend, or static plots using the
    Matplotlib backend.
    :return: Dict mapping names to figures.
    """
    pos_prevalence = overall.get('n_pos', 0) / overall.get('n', 1)
    metrics = [m for m in ('accuracy', 'balanced_accuracy', 'f1', 'sensitivity', 'specificity',
                           'positive_predictive_value', 'negative_predictive_value') if m in threshold.columns]
    cm, th = _get_confusion_matrix_from_thresholds(threshold, neg_label=neg_label, pos_label=pos_label)
    if interactive:
        if plotting.plotly_backend is None:
            logging.warn(plotting.PLOTLY_WARNING)
            return {}
        else:
            backend = plotting.plotly_backend
    else:
        backend = plotting.mpl_backend

    out = dict(
        roc_curve=backend.roc_pr_curve(1 - threshold['specificity'], threshold['sensitivity'], roc=True,
                                       legend='AUC={:.4f}'.format(overall.get('roc_auc', 0.5)), name=name),
        pr_curve=backend.roc_pr_curve(threshold['sensitivity'], threshold['positive_predictive_value'], roc=False,
                                      legend='AUC={:.4f}'.format(overall.get('pr_auc', pos_prevalence)),
                                      positive_prevalence=pos_prevalence, name=name),
        threshold=backend.threshold_metric_curve(threshold['threshold'], [threshold[m] for m in metrics],
                                                 legend=metrics, name=name),
        confusion_matrix=backend.confusion_matrix(cm, title='Confusion Matrix @ {:.2f}'.format(th), name=name)
    )
    if calibration is not None:
        out['calibration'] = backend.calibration_curve(calibration['threshold_lower'], calibration['threshold_upper'],
                                                       calibration['pos_fraction'], name=name)
    return out


def calc_multiclass_metrics(y_true: pd.DataFrame, y_hat: Union[pd.DataFrame, np.ndarray],
                            labels: Optional[list] = None) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Calculate all metrics suitable for multiclass classification.
    :param y_true: Ground truth. Must have 1 column with float data type and values among
    NaN, 0, 1, ..., `n_classes` - 1.
    :param y_hat: Predicted class probabilities. Must have shape `(len(y_true), n_classes)` and values between 0 and 1.
    :param labels: Class names.
    :return: Triple `(overall, conf_mat, per_class)`, where `overall` is a dict with overall performance metrics
    (accuracy, F1, etc.), `conf_mat` is the confusion matrix, and `per_class` is a DataFrame with per-class metrics
    (one row per class, one column per metric).
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
    y_pred = np.argmax(y_hat, axis=1).astype(y_true.dtype)

    # confusion matrix
    conf_mat = metrics.confusion_matrix(y_true, y_pred)
    conf_mat = np.concatenate([conf_mat, conf_mat.sum(axis=0, keepdims=True)])
    conf_mat = pd.DataFrame(data=conf_mat, columns=labels, index=labels + ['__total__'])
    conf_mat['__total__'] = conf_mat.sum(axis=1)
    conf_mat.index.name = 'true \\ pred'        # suitable for saving as Excel file

    precision, recall, f1, support = \
        metrics.precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    jaccard = metrics.jaccard(y_true, y_pred, average=None, zero_division=0)
    n = support.sum()

    # per-class metrics
    per_class = pd.DataFrame(
        index=labels,
        columns=_BINARY_PROBA_METRICS + _BINARY_CLASS_METRICS
    )
    for i, lbl in enumerate(labels):
        for m in _BINARY_PROBA_METRICS:
            try:
                per_class[m].values[i] = getattr(metrics, m)(y_true == i, y_hat[:, i])
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
                    per_class[m].values[i] = getattr(metrics, m)(y_true == i, y_pred == i)
                except:     # noqa
                    pass
    per_class.insert(0, 'n', conf_mat['__total__'].iloc[:-1])

    # overall metrics
    dct = dict(n=n)
    for name in ['accuracy', 'balanced_accuracy', 'cohen_kappa', 'matthews_correlation_coefficient']:
        try:
            dct[name] = getattr(metrics, name)(y_true, y_pred)
        except:  # noqa
            pass
    for name in ['roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']:
        try:
            dct[name] = getattr(metrics, name)(y_true, y_hat)
        except:  # noqa
            pass
    precision_micro, recall_micro, f1_micro, _ = \
        metrics.precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
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
    :param confusion_matrix: Confusion matrix, as returned by function `calc_multiclass_metrics()`.
    :param interactive: Whether to create interactive plots using the plotly backend, or static plots using the
    Matplotlib backend.
    :return: Dict mapping names to figures.
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
                            thresholds: Optional[list] = None) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Calculate all metrics suitable for multilabel classification.
    :param y_true: Ground truth. Must have `n_classes` columns with float data type and values among 0, 1 and NaN.
    :param y_hat: Predicted class probabilities. Must have shape `(len(y_true), n_classes)` and values between 0 and 1.
    :param thresholds: List of thresholds to use for thresholded metrics. If None, a default list of thresholds
    depending on the values of `y_hat` is constructed.
    :return: Triple `(overall, threshold, threshold_per_class)`:
    * `overall` is a DataFrame containing non-thresholded metrics per class and for all classes combined
        ("__micro__", "__macro__" and "__weighted__"). Weights are the number of positive samples per class.
    * `threshold` is a DataFrame containing thresholded metrics for different thresholds for all classes combined.
    * `threshold_per_class` is a dict mapping classes to per-class thresholded metrics.
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

    dct = {}
    for i, lbl in enumerate(labels):
        dct_i = dict(n=mask[:, i].sum(), n_pos=n_positive[i])
        for m in _BINARY_PROBA_METRICS:
            try:
                dct_i[m] = getattr(metrics, m)(y_true[mask[:, i], i], y_hat[mask[:, i], i])
            except:  # noqa
                pass
        dct[lbl] = dct_i

    # micro average
    y_true_flat = y_true.ravel()
    y_hat_flat = y_hat.ravel()
    mask_flat = mask.ravel()
    dct_i = dict(n=mask.sum(), n_pos=(y_true > 0).sum())
    for m in _BINARY_PROBA_METRICS:
        try:
            dct_i[m] = getattr(metrics, m)(y_true_flat[mask_flat], y_hat_flat[mask_flat])
        except:  # noqa
            pass
    dct['__micro__'] = dct_i

    # macro and weighted average
    dct['__macro__'] = dict(n=mask.any(axis=1).sum(), n_pos=(y_true > 0).any(axis=1).sum())
    dct['__weighted__'] = dct['__macro__']
    overall = pd.DataFrame.from_dict(dct, orient='index')
    div = overall['n_pos'].sum() - overall.loc[['__micro__', '__macro__', '__weighted__'], 'n_pos'].sum()
    for c in overall.columns:
        if c not in ('n', 'n_pos'):
            overall.loc['__macro__', c] = overall[c].iloc[:-3].mean()
            overall.loc['__weighted__', c] = (overall['n_pos'].iloc[:-3] * overall[c].iloc[:-3]).sum() / div

    if thresholds is None:
        thresholds = metrics.get_thresholds(y_hat.reshape(-1), n_max=100, add_half_one=None)

    per_class = {}
    for j, lbl in enumerate(labels):
        out = pd.DataFrame(data=dict(threshold=thresholds, true_negative=0, true_positive=0))
        y_true_j = y_true[mask[:, j], j]
        y_hat_j = y_hat[mask[:, j], j]
        for i, t in enumerate(thresholds):
            out['true_positive'].values[i] = ((y_true_j > 0) & (y_hat_j >= t)).sum()
            out['true_negative'].values[i] = ((y_true_j < 1) & (y_hat_j < t)).sum()
        out['false_positive'] = n_negative[j] - out['true_negative']
        out['false_negative'] = n_positive[j] - out['true_positive']

        for m in reversed(_BINARY_CLASS_METRICS):
            func = getattr(metrics, m + '_cm', None)
            if func is None:
                s = np.full((len(out),), np.nan)
                func = getattr(metrics, m)
                for i, t in enumerate(thresholds):
                    try:
                        s[i] = func(y_true_j > 0, y_hat_j >= t)
                    except:     # noqa
                        pass
            else:
                s = func(tp=out['true_positive'], tn=out['true_negative'], fp=out['false_positive'],
                         fn=out['false_negative'], average=None)
            out.insert(1, m, s)

        if 'sensitivity' in out.columns and 'specificity' in out.columns:
            i = (out['sensitivity'] - out['specificity']).abs().idxmin()
            if 'balance_threshold' not in overall.columns:
                overall['balance_threshold'] = np.nan
                overall['balance_score'] = np.nan
            overall.loc[lbl, 'balance_threshold'] = out.loc[i, 'threshold']
            overall.loc[lbl, 'balance_score'] = (out.loc[i, 'sensitivity'] + out.loc[i, 'specificity']) * 0.5

        per_class[lbl] = out

    out = pd.DataFrame(data=dict(threshold=thresholds))
    tp = np.zeros((len(thresholds),), dtype=np.int32)
    tn = np.zeros_like(tp)
    for i, t in enumerate(thresholds):
        tp[i] = ((y_true_flat[mask_flat] > 0) & (y_hat_flat[mask_flat] >= t)).sum()
        tn[i] = ((y_true_flat[mask_flat] < 1) & (y_hat_flat[mask_flat] < t)).sum()
    for m in _BINARY_CLASS_METRICS:
        func = getattr(metrics, m + '_cm', None)
        if func is None:
            out[m + '_micro'] = np.nan
            func = getattr(metrics, m)
            for i, t in enumerate(thresholds):
                try:
                    out[m + '_micro'].values[i] = func(y_true_flat[mask_flat] > 0, y_hat_flat[mask_flat] >= t)
                except:  # noqa
                    pass
        else:
            out[m + '_micro'] = func(tp=tp, tn=tn, fp=n_negative.sum() - tn, fn=n_positive.sum() - tp, average=None)
        out[m + '_macro'] = sum(v[m] for v in per_class.values()) / len(per_class)
        out[m + '_weighted'] = sum(v[m] * overall.loc[k, 'n_pos'] for k, v in per_class.items()) / div

    # drop all-NaN columns
    out.dropna(axis=1, how='all', inplace=True)
    return overall, out, per_class


def plot_multilabel(overall: pd.DataFrame, threshold: dict, labels=None, interactive: bool = False) -> dict:
    """
    Plot evaluation results of binary classification tasks.
    :param overall: Overall, non-thresholded performance metrics, as returned by function `calc_multilabel_metrics()`.
    :param threshold: Thresholded performance metrics, as returned by function `calc_multilabel_metrics()`.
    :param labels: Class names. None or a DataFrame with `n_class` columns and 2 rows.
    :param interactive: Whether to create interactive plots using the plotly backend, or static plots using the
    Matplotlib backend.
    :return: Dict mapping names to figures.
    """
    out = {}
    kwargs = {}
    for name, th in threshold.items():
        if labels is not None:
            kwargs = dict(neg_label=str(labels[name].iloc[0]), pos_label=str(labels[name].iloc[1]))
        out[name] = \
            plot_binary_classification(overall.loc[name].to_dict(), th, name=name, interactive=interactive, **kwargs)
    return out


def _get_confusion_matrix_from_thresholds(thresholds: pd.DataFrame, threshold: float = 0.5, neg_label: str = 'negative',
                                          pos_label: str = 'positive') -> Tuple[pd.DataFrame, float]:
    i = (thresholds['threshold'] - threshold).abs().idxmin()
    return pd.DataFrame(
        data={neg_label: [thresholds.loc[i, 'true_negative'], thresholds.loc[i, 'false_negative']],
              pos_label: [thresholds.loc[i, 'false_positive'], thresholds.loc[i, 'true_positive']]},
        index=[neg_label, pos_label]
    ), thresholds.loc[i, 'threshold']


# metrics for binary classification, which require probabilities of positive class
_BINARY_PROBA_METRICS = ['roc_auc', 'average_precision', 'pr_auc', 'brier_loss', 'hinge_loss', 'log_loss']
# metrics for binary classification, which require predicted classes
_BINARY_CLASS_METRICS = ['accuracy', 'balanced_accuracy', 'f1', 'sensitivity', 'specificity',
                         'positive_predictive_value', 'negative_predictive_value', 'cohen_kappa', 'hamming_loss',
                         'jaccard']
