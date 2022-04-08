from typing import Union, Optional, Tuple
from functools import partial
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import sklearn

from ..util import table as tu
from ..util import io
from ..util import logging
from ..util.encoding import Encoder


def evaluate(*table: Union[str, Path, pd.DataFrame], folder: Union[str, Path] = None, model_id=None,
             split: Optional[str] = None, out: Union[str, Path, None] = None, jobs: Optional[int] = None,
             batch_size: Optional[int] = None):
    """
    Evaluate an existing CaTabRa object (OOD-detector, prediction model, ...) on held-out test data.
    :param table: The table(s) to evaluate the CaTabRa object on. If multiple are given, their columns are merged into
    a single table. Must have the same format as the table(s) initially passed to function `analyze()`.
    :param folder: The folder containing the CaTabRa object to evaluate.
    :param model_id: Optional, ID of the prediction model to evaluate. If None, the sole trained model or the entire
    ensemble are evaluated.
    :param split: Optional, column used for splitting the data into disjoint subsets. If specified, each subset is
    evaluated individually. In contrast to function `analyze()`, the name/values of the column do not need to carry any
    semantic information about training and test sets.
    :param out: Optional, directory where to save all generated artifacts. Defaults to a directory located in `folder`,
    with a name following a fixed naming pattern. If `out` already exists, the user is prompted to specify whether it
    should be replaced; otherwise, it is automatically created.
    :param jobs: Optional, number of jobs to use.
    :param batch_size: Optional, batch size used for applying the prediction model.
    """
    if len(table) == 0:
        raise ValueError('No table specified.')
    if folder is None:
        raise ValueError('No folder specified.')
    elif not isinstance(folder, Path):
        folder = Path(folder)
    if not folder.exists():
        raise ValueError(f'Folder "{folder.as_posix()}" does not exist.')

    config = io.load(folder / 'config.json')

    start = pd.Timestamp.now()
    if out is None:
        out = table[0]
        if isinstance(out, pd.DataFrame):
            out = folder / ('eval_' + start.strftime('%Y-%m-%d_%H-%M-%S'))
        else:
            if not isinstance(out, Path):
                out = Path(out)
            out = folder / ('eval_' + out.stem + '_' + start.strftime('%Y-%m-%d_%H-%M-%S'))
    elif not isinstance(out, Path):
        out = Path(out)
    if out.exists():
        if logging.prompt(f'Evaluation folder "{out.as_posix()}" already exists. Delete?',
                          accepted=['y', 'n'], allow_headless=False) == 'y':
            if out.is_dir():
                shutil.rmtree(out.as_posix())
            else:
                out.unlink()
        else:
            print('### Aborting')
            return
    out.mkdir(parents=True)

    with logging.LogMirror((out / 'console.txt').as_posix()):
        logging.log(f'### Evaluation started at {start}')
        invocation = dict(
            table=['<DataFrame>' if isinstance(tbl, pd.DataFrame) else tbl for tbl in table],
            folder=folder,
            model_id=model_id,
            split=split,
            out=out,
            timestamp=start
        )
        io.dump(io.to_json(invocation), out / 'invocation.json')

        # merge tables
        df, _ = tu.merge_tables(table)
        if df.columns.nlevels != 1:
            raise ValueError(f'Table must have 1 column level, but found {df.columns.nlevels}.')

        # copy test data
        copy_data = config.get('copy_test_data', False)
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

        # TODO: Perform OOD checks.

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
                y_hat_decoded.columns = [f'{c}_pred' for c in y_hat_decoded.columns]
                detailed = y_test_decoded.join(y_hat_decoded)
                detailed = detailed.reindex(
                    [n for c in zip(y_test_decoded.columns, y_hat_decoded.columns) for n in c],
                    axis=1
                )
                del y_test_decoded
                del y_hat_decoded

                for mask, directory in _iter_splits():
                    directory.mkdir(exist_ok=True, parents=True)
                    io.write_df(detailed[mask], directory / 'predictions.xlsx')
                    io.write_df(calc_regression_metrics(y_test[mask], y_hat[mask]), directory / 'metrics.xlsx')
                    # TODO: For each target: plot truth vs. prediction.
            else:
                y_hat = model.predict_proba(x_test, jobs=jobs, batch_size=batch_size, model_id=model_id)
                if encoder.task_ == 'multilabel_classification':
                    # TODO
                    # - per sample: true class indicators, predicted probabilities
                    # - per split: all suitable metrics for overall performance, multilabel confusion matrix (?),
                    #       binary classification output for each class individually
                    # - plots: binary classification plots for each class individually
                    pass
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

                    if encoder.task_ == 'binary_classification':
                        pos_label = \
                            encoder.inverse_transform(y=np.ones((1,), dtype=np.float32), inplace=True).iloc[0, 0]
                        for mask, directory in _iter_splits():
                            directory.mkdir(exist_ok=True, parents=True)
                            io.write_df(detailed[mask], directory / 'predictions.xlsx')
                            overall, thresh = calc_binary_classification_metrics(y_test[mask], y_hat[mask])
                            overall = pd.DataFrame(data=overall, index=[y_test.columns[0]])
                            overall.insert(0, 'pos_label', pos_label)
                            io.write_dfs(dict(overall=overall, thresholded=thresh), directory / 'metrics.xlsx')
                            # TODO: Plot
                            #   * ROC- and PR curves (both including best feature estimator);
                            #   * threshold vs. metric, with accuracy, balanced_accuracy, F1, sensitivity, specificity,
                            #       PPV, NPV in one common figure;
                            #   * color-coded confusion matrix @0.5.
                    else:
                        for mask, directory in _iter_splits():
                            directory.mkdir(exist_ok=True, parents=True)
                            io.write_df(detailed[mask], directory / 'predictions.xlsx')
                            # TODO: Generate confusion matrix and other suitable metrics. Use PyCM?
                            # TODO: Plot color-coded confusion matrix.

        end = pd.Timestamp.now()
        logging.log(f'### Evaluation finished at {end}')
        logging.log(f'### Elapsed time: {end - start}')


def calc_regression_metrics(y_true: pd.DataFrame, y_hat: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
    """
    Calculate all suitable metrics for all targets individually, and for their combination.
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
        assert y_hat.columns == y_true.columns
        y_hat = y_hat.values
    elif isinstance(y_hat, np.ndarray) and y_hat.ndim == 1:
        y_hat = y_hat.reshape((-1, 1))
    targets = y_true.columns
    y_true = y_true.values

    mask = np.isfinite(y_true) & np.isfinite(y_hat)
    out = pd.DataFrame(index=list(targets) + ['__overall__'], data=dict(n=0))
    out['n'].values[:-1] = mask.sum(axis=0)
    out.loc['__overall__', 'n'] = mask.all(axis=1).sum()

    for name, func in [('r2', sklearn.metrics.r2_score),
                       ('mean_absolute_error', sklearn.metrics.mean_absolute_error),
                       ('mean_squared_error', sklearn.metrics.mean_squared_error),
                       ('root_mean_squared_error', partial(sklearn.metrics.mean_squared_error, squared=False)),
                       ('mean_squared_log_error', sklearn.metrics.mean_squared_log_error),
                       ('median_absolute_error', sklearn.metrics.median_absolute_error),
                       ('mean_absolute_percentage_error', sklearn.metrics.mean_absolute_percentage_error),
                       ('max_error', sklearn.metrics.max_error),
                       ('explained_variance', sklearn.metrics.explained_variance_score),
                       ('mean_tweedie_deviance', sklearn.metrics.mean_tweedie_deviance),
                       ('mean_poisson_deviance', sklearn.metrics.mean_poisson_deviance),
                       ('mean_gamma_deviance', sklearn.metrics.mean_gamma_deviance)]:
        out[name] = np.nan
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


def calc_binary_classification_metrics(y_true: pd.DataFrame, y_hat: Union[pd.DataFrame, np.ndarray]) \
        -> Tuple[dict, pd.DataFrame]:
    """
    Calculate all suitable metrics.
    :param y_true: Ground truth. Must have 1 column with float data type and values among 0, 1 and NaN.
    :param y_hat: Predictions. Must have the same number of rows as `y_true` and either 1 or 2 columns.
    :return: Pair `(overall, threshold)`, where `overall` is a dict containing the scores of threshold-independent
    metrics (e.g., ROC-AUC) and `threshold` is a DataFrame with one column for each threshold-dependent metric, and one
    row for each decision threshold.
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

    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_hat)
    dct = dict(n=mask.sum())
    for name, func in [('roc_auc', sklearn.metrics.roc_auc_score),
                       ('average_precision', sklearn.metrics.average_precision_score),
                       ('brier_loss', sklearn.metrics.brier_score_loss),
                       ('hinge_loss', sklearn.metrics.hinge_loss),
                       ('log_loss', sklearn.metrics.log_loss)]:
        try:
            dct[name] = func(y_true, y_hat)
        except:     # noqa
            pass
    try:
        dct['pr_auc'] = sklearn.metrics.auc(precision, recall)
    except:  # noqa
        pass

    if len(thresholds) == 0:
        thresholds = [0.5]
    elif len(thresholds) > 110:
        thresholds = thresholds[np.linspace(0, len(thresholds) - 1, 100).round().astype(np.int32)]
    out = pd.DataFrame(data=dict(threshold=thresholds))
    n_positive = (y_true > 0).sum()
    n_negative = (y_true < 1).sum()
    for i, t in enumerate(thresholds):
        y_pred = (y_hat >= t).astype(np.float32)
        for name, func in [('accuracy', sklearn.metrics.accuracy_score),
                           ('balanced_accuracy', sklearn.metrics.balanced_accuracy_score),
                           ('f1', sklearn.metrics.f1_score),
                           ('sensitivity', partial(sklearn.metrics.recall_score, zero_division=0)),
                           ('specificity', partial(sklearn.metrics.recall_score, pos_label=0, zero_division=0)),
                           ('positive_predictive_value', partial(sklearn.metrics.precision_score, zero_division=0)),
                           ('negative_predictive_value', partial(sklearn.metrics.precision_score, pos_label=0, zero_division=0)),
                           ('cohen_kappa', sklearn.metrics.cohen_kappa_score),
                           ('hamming_loss', sklearn.metrics.hamming_loss),
                           ('jaccard_index', sklearn.metrics.jaccard_score)]:
            if i == 0:
                out[name] = np.nan
            try:
                out[name].values[i] = func(y_true, y_pred)
            except:     # noqa
                pass
        if i == 0:
            out['true_positive'] = 0
            out['true_negative'] = 0
        out['true_positive'].values[i] = ((y_true > 0) & (y_pred > 0)).sum()
        out['true_negative'].values[i] = ((y_true < 1) & (y_pred < 1)).sum()
    out['false_positive'] = n_negative - out['true_negative']
    out['false_negative'] = n_positive - out['true_positive']

    # drop all-NaN columns
    out.dropna(axis=1, how='all', inplace=True)
    return dct, out
