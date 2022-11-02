from typing import Union, Optional, Tuple
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

from ..util import table as tu
from ..core import logging, io
from ..util import plotting
from ..core.paths import CaTabRaPaths


def explain(*table: Union[str, Path, pd.DataFrame], folder: Union[str, Path] = None, model_id=None,
            split: Optional[str] = None, sample_weight: Optional[str] = None, out: Union[str, Path, None] = None,
            glob: Optional[bool] = False, jobs: Optional[int] = None, batch_size: Optional[int] = None,
            from_invocation: Union[str, Path, dict, None] = None):
    """
    Explain an existing CaTabRa object (prediction model) in terms of feature importance.
    :param table: The table(s) to explain the CaTabRa object on. If multiple are given, their columns are merged into
    a single table. Must have the same format as the table(s) initially passed to function `analyze()`, possibly
    without target column(s).
    :param folder: The folder containing the CaTabRa object to explain.
    :param model_id: Optional, ID(s) of the prediction model(s) to evaluate. If None or "__ensemble__", all models in
    the ensemble are explained, if possible. Note that due to technical restrictions not all models might be
    explainable.
    :param split: Optional, column used for splitting the data into disjoint subsets. If specified and not "", each
    subset is explained individually. In contrast to function `analyze()`, the name/values of the column do not need to
    carry any semantic information about training and test sets.
    :param sample_weight: Optional, column with sample weights. If specified and not "", must have numeric data type.
    Sample weights are used both for training and evaluating prediction models.
    :param out: Optional, directory where to save all generated artifacts. Defaults to a directory located in `folder`,
    with a name following a fixed naming pattern. If `out` already exists, the user is prompted to specify whether it
    should be replaced; otherwise, it is automatically created.
    :param glob: Whether to explain the CaTabRa object globally. If True, `table` might not have to be specified
    (depends on explanation backend).
    :param jobs: Optional, number of jobs to use. Overwrites the "jobs" config param.
    :param batch_size: Optional, batch size used for explaining the prediction model(s).
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
        if sample_weight is None:
            sample_weight = from_invocation.get('sample_weight')
        if out is None:
            out = from_invocation.get('out')
        if glob is None:
            glob = from_invocation.get('glob')
        if jobs is None:
            jobs = from_invocation.get('jobs')

    if len(table) == 0:
        raise ValueError('No table specified.')
    if folder is None:
        raise ValueError('No CaTabRa directory specified.')

    loader = io.CaTabRaLoader(folder, check_exists=True)
    config = loader.get_config()

    start = pd.Timestamp.now()
    table = [io.make_path(tbl, absolute=True) if isinstance(tbl, (str, Path)) else tbl for tbl in table]

    if out is None:
        out = table[0]
        if isinstance(out, pd.DataFrame):
            out = loader.path / ('explain_' + start.strftime('%Y-%m-%d_%H-%M-%S'))
        else:
            out = loader.path / ('explain_' + out.stem + '_' + start.strftime('%Y-%m-%d_%H-%M-%S'))
    else:
        out = io.make_path(out, absolute=True)
    if out == loader.path:
        raise ValueError(f'Output directory must differ from CaTabRa directory, but both are "{out.as_posix()}".')
    elif out.exists():
        if logging.prompt(f'Explanation folder "{out.as_posix()}" already exists. Delete?',
                          accepted=['y', 'n'], allow_headless=False) == 'y':
            if out.is_dir():
                shutil.rmtree(out.as_posix())
            else:
                out.unlink()
        else:
            logging.log('### Aborting')
            return
    out.mkdir(parents=True)

    if out is None:
        out = table[0]
        if isinstance(out, pd.DataFrame):
            raise ValueError('Output directory must be specified when passing a DataFrame.')
        out = out.parent / (out.stem + '_catabra_' + start.strftime('%Y-%m-%d_%H-%M-%S'))
    else:
        out = io.make_path(out, absolute=True)
    if out.exists():
        if logging.prompt(f'Output folder "{out.as_posix()}" already exists. Delete?',
                          accepted=['y', 'n'], allow_headless=False) == 'y':
            if out.is_dir():
                shutil.rmtree(out.as_posix())
            else:
                out.unlink()
        else:
            logging.log('### Aborting')
            return
    out.mkdir(parents=True)

    explainer = loader.get_explainer()
    if explainer is None:
        logging.log('### Aborting: no trained prediction model or no explainer params found')
        return
    global_behavior = explainer.global_behavior()

    if split == '':
        split = None
    if sample_weight == '':
        sample_weight = None
    if glob is None:
        glob = not (global_behavior.get('mean_of_local', False) or len(table) > 0)

    with logging.LogMirror((out / CaTabRaPaths.ConsoleLogs).as_posix()):
        logging.log(f'### Explanation started at {start}')
        invocation = dict(
            table=['<DataFrame>' if isinstance(tbl, pd.DataFrame) else tbl for tbl in table],
            folder=loader.path,
            model_id=model_id,
            split=split,
            sample_weight=sample_weight,
            out=out,
            glob=glob,
            jobs=jobs,
            timestamp=start
        )
        io.dump(io.to_json(invocation), out / CaTabRaPaths.Invocation)

        encoder = loader.get_encoder()

        # merge tables
        df, _ = tu.merge_tables(table)
        if df is None:
            if not glob or global_behavior.get('requires_x', False):
                raise ValueError('No table(s) to explain models on specified.')

            def _iter_splits():
                yield None, out

            x_test = None
        else:
            if df.columns.nlevels != 1:
                raise ValueError(f'Table must have 1 column level, but found {df.columns.nlevels}.')

            # copy test data
            copy_data = config.get('copy_evaluation_data', False)
            if isinstance(copy_data, (int, float)):
                copy_data = df.memory_usage(index=True, deep=True).sum() <= copy_data * 1000000
            if copy_data:
                io.write_df(df, out / 'explanation_data.h5')

            # split
            if split is None:
                def _iter_splits():
                    yield None, out
            else:
                split_masks, _ = tu.train_test_split(df, split)

                def _iter_splits():
                    for _k, _m in split_masks.items():
                        yield _m, out / _k

            x_test = encoder.transform(x=df)

        if sample_weight is None or df is None:
            sample_weights = None
        elif sample_weight in df.columns:
            if df[sample_weight].dtype.kind not in 'fiub':
                raise ValueError(f'Column "{sample_weight}" must have numeric data type,'
                                 f' but found {df[sample_weight].dtype.name}.')
            logging.log(f'Weighting samples by column "{sample_weight}"')
            sample_weights = df[sample_weight].values
            na_mask = np.isnan(sample_weights)
            if na_mask.any():
                sample_weights = sample_weights.copy()
                sample_weights[na_mask] = 1.
        else:
            raise ValueError(f'"{sample_weight}" is no column of the specified table.')

        static_plots = config.get('static_plots', True)
        interactive_plots = config.get('interactive_plots', False)
        if interactive_plots and plotting.plotly_backend is None:
            logging.warn(plotting.PLOTLY_WARNING)
            interactive_plots = False

        if model_id == '__ensemble__':
            model_id = None
        for mask, directory in _iter_splits():
            if mask is not None:
                logging.log('*** Split ' + directory.stem)
            explain_split(explainer, x=x_test if mask is None else x_test[mask],
                          sample_weight=None if sample_weights is None else sample_weights[mask],
                          directory=directory, glob=glob, model_id=model_id, batch_size=batch_size, jobs=jobs,
                          static_plots=static_plots, interactive_plots=interactive_plots, verbose=True)

        end = pd.Timestamp.now()
        logging.log(f'### Explanation finished at {end}')
        logging.log(f'### Elapsed time: {end - start}')
        logging.log(f'### Output saved in {out.as_posix()}')


def explain_split(explainer: 'EnsembleExplainer', x: Optional[pd.DataFrame] = None,
                  sample_weight: Optional[np.ndarray] = None, directory=None, glob: bool = False,
                  model_id=None, batch_size: Optional[int] = None, jobs: int = 1, static_plots: bool = True,
                  interactive_plots: bool = False, verbose: bool = False) -> Optional[dict]:
    """
    Explain a single data split.
    :param explainer: Explainer object.
    :param x: Encoded data to apply `explainer` to, optional unless `glob` is False. Only features, no labels.
    :param sample_weight: Sample weights, optional. Ignored if `x` is None or `glob` is False.
    :param directory: Directory where to save the explanations. If None, results are returned in a dict.
    :param glob: Whether to create global explanations.
    :param model_id: ID(s) of the model(s) to explain.
    :param batch_size: Batch size.
    :param jobs: Number of jobs.
    :param static_plots: Whether to create static plots.
    :param interactive_plots: Whether to create interactive plots.
    :param verbose: Whether to print intermediate results and progress bars.
    :return: None if `directory` is given, else dict with evaluation results.
    """

    if directory is None:
        out = {}

        def _save_plots(_obj, _name: str):
            out[_name] = _obj
    else:
        out = None
        directory = io.make_path(directory)
        directory.mkdir(exist_ok=True, parents=True)

        def _save_plots(_obj, _name: str):
            plotting.save(_obj, directory / _name)

    title = explainer.name() + ' Feature Importance'
    if glob:
        explanations: dict = explainer.explain_global(x=x, sample_weight=sample_weight, jobs=jobs,
                                                      batch_size=batch_size, model_id=model_id, show_progress=verbose)
    else:
        explanations: dict = explainer.explain(x, jobs=jobs, batch_size=batch_size,
                                               model_id=model_id, show_progress=verbose)
    if static_plots:
        _save_plots(plot_beeswarms(explanations, features=x, interactive=False, title=title), 'static_plots')
    if interactive_plots:
        _save_plots(plot_beeswarms(explanations, features=x, interactive=True, title=title), 'interactive_plots')

    if directory is None:
        out['explanations'] = explanations
    else:
        for k, v in explanations.items():
            fn = directory / (str(k) + '.h5')
            if isinstance(v, dict):
                io.write_dfs(v, fn)
            elif isinstance(v, pd.Series):
                io.write_df(v.to_frame(), fn)
            else:
                io.write_df(v, fn)

    return out


def plot_beeswarms(explanations: Union[dict, str, Path, pd.DataFrame], features: Optional[pd.DataFrame] = None,
                   interactive: bool = False, title: Optional[str] = None, max_features: Optional[int] = None,
                   add_sum_of_remaining: bool = True) -> Union[dict, pd.DataFrame]:
    """
    Create beeswarm plots of local explanations.
    :param explanations: Local explanations to plot, a dict as returned by `EnsembleExplainer.explain()`, i.e., 1-2
    levels of nesting, values are DataFrames with samples on row index and features on column index.
    :param features: Encoded feature values corresponding to feature importance scores, optional.
    :param interactive: Whether to create interactive or static plots.
    :param title: The title of the plots.
    :param max_features: Maximum number of features to plot, or None to determine this number automatically.
    :param add_sum_of_remaining: Whether to add the sum of remaining features, if not all features can be plotted.
    :return: Dict with plots or single plot.
    """
    if interactive:
        if plotting.plotly_backend is None:
            logging.warn(plotting.PLOTLY_WARNING)
            return {}
        else:
            backend = plotting.plotly_backend
    else:
        backend = plotting.mpl_backend

    key = None
    if isinstance(explanations, (str, Path)):
        aux = {k: v for k, v in io.read_dfs(explanations).items() if isinstance(v, pd.DataFrame)}
        if aux:
            explanations = aux
        else:
            key = 'table'
            explanations = {key: io.read_df(explanations)}
    elif isinstance(explanations, pd.DataFrame):
        key = 'table'
        explanations = {key: explanations}

    out = {}
    for k, v in explanations.items():
        if isinstance(v, (str, Path)):
            aux = {k1: v1 for k1, v1 in io.read_dfs(v).items() if isinstance(v1, pd.DataFrame)}
            if aux:
                v = aux
            else:
                v = io.read_df(v)
        if isinstance(v, dict):
            aux = plot_beeswarms(v, features=features, interactive=interactive, title=title, max_features=max_features,
                                 add_sum_of_remaining=add_sum_of_remaining)
            for k1, v1 in aux.items():
                out[str(k) + '_' + str(k1)] = v1
        elif isinstance(v, pd.DataFrame):
            out[str(k)] = \
                backend.beeswarm(_prepare_for_beeswarm(v, max_features, add_sum_of_remaining),
                                 colors=features if features is None or len(features) == len(v) else None,
                                 color_name='Feature value', title=title, x_label='Importance')

    if key is not None:
        return out[key]
    return out


def plot_bars(explanations: Union[dict, str, Path, pd.DataFrame], interactive: bool = False,
              title: Optional[str] = None, max_features: int = 10, add_sum_of_remaining: bool = True) \
        -> Union[dict, pd.DataFrame]:
    """
    Create bar plots of global explanations.
    :param explanations: Global explanations to plot, a dict as returned by `EnsembleExplainer.explain_global()`, i.e.,
    values are Series or DataFrames with features on row index and arbitrary column index.
    :param interactive: Whether to create interactive or static plots.
    :param title: The title of the plots.
    :param max_features: Maximum number of features to plot.
    :param add_sum_of_remaining: Whether to add the sum of remaining features, if not all features can be plotted.
    :return: Dict with plots or single plot.
    """
    if interactive:
        if plotting.plotly_backend is None:
            logging.warn(plotting.PLOTLY_WARNING)
            return {}
        else:
            backend = plotting.plotly_backend
    else:
        backend = plotting.mpl_backend

    key = None
    if isinstance(explanations, (str, Path)):
        aux = {k: v for k, v in io.read_dfs(explanations).items() if isinstance(v, pd.DataFrame)}
        if aux:
            explanations = aux
        else:
            key = 'table'
            explanations = {key: io.read_df(explanations)}
    elif isinstance(explanations, pd.DataFrame):
        key = 'table'
        explanations = {key: explanations}

    out = {}
    for k, v in explanations.items():
        if isinstance(v, (str, Path)):
            v = io.read_df(v)
        df, groups = _prepare_for_bar(v, max_features, add_sum_of_remaining)
        out[str(k)] = backend.horizontal_bar(df, groups=groups, title=title, x_label='Importance')

    if key is not None:
        return out[key]
    return out


def average_local_explanations(explanations: Union[pd.DataFrame, dict], sample_weight: Optional[np.ndarray] = None,
                               **kwargs) -> Union[np.ndarray, pd.DataFrame, dict]:
    """
    Average local explanations to get a global overview of feature importance.
    :param explanations: Local explanations to average, DataFrame of shape `(*dim, n_samples, n_features)` or a
    (nested) dict thereof with at most two levels of nesting.
    :param sample_weight: Sample weights, optional.
    :return: Averaged explanations, with the same format as what would be returned by method
    `EnsembleExplainer.explain_global()`. That is, either a single DataFrame, or a dict whose values are DataFrames.
    """
    if isinstance(explanations, dict):
        if kwargs.get('_require_df', False):
            raise ValueError('Expected DataFrame, got dict.')
        elif kwargs.get('_nest', True):
            return {k: average_local_explanations(v, sample_weight=sample_weight, _nest=False)
                    for k, v in explanations.items()}
        else:
            dfs = []
            for k, v in explanations.items():
                df = average_local_explanations(v, sample_weight=sample_weight, _require_df=True)
                df.columns = [f'{k}_{c}' for c in df.columns]
                dfs.append(df)
            if dfs:
                return pd.concat(dfs, axis=1, sort=False)
            else:
                return pd.DataFrame()
    else:
        if sample_weight is None:
            w = 1
            div = len(explanations)
        else:
            w = sample_weight[..., np.newaxis]
            div = sample_weight.sum()
        positive = ((explanations * (explanations > 0)) * w).sum(axis=0) / div
        negative = ((explanations * (explanations < 0)) * w).sum(axis=0) / div
        positive_std = np.sqrt((np.square(((explanations * (explanations > 0)) - positive)) * w).sum(axis=0) / div)
        negative_std = np.sqrt((np.square(((explanations * (explanations < 0)) - negative)) * w).sum(axis=0) / div)
        return pd.concat(
            [positive.to_frame('>0'), negative.to_frame('<0'),
             positive_std.to_frame('>0 std'), negative_std.to_frame('<0 std')],
            axis=1,
            sort=False
        )


def _prepare_for_beeswarm(df: pd.DataFrame, max_features: Optional[int], add_sum_of_remaining: bool) -> pd.DataFrame:
    cols = df.abs().mean(axis=0).sort_values(ascending=False).index
    if max_features is None:
        n = max(6, int(50000 / max(len(df), 1)))    # number of features to show
    else:
        n = max_features
    if n < len(cols) and add_sum_of_remaining:
        df0 = df.reindex(cols[:n - 1], axis=1)
        df0[f'Sum of {len(cols) + 1 - n} remaining features'] = df[cols[n - 1:]].sum(axis=1)
        return df0
    else:
        return df.reindex(cols[:n], axis=1)


def _prepare_for_bar(df: Union[pd.DataFrame, pd.Series], max_features: int,
                     add_sum_of_remaining: bool) -> Tuple[pd.DataFrame, dict]:
    if isinstance(df, pd.Series):
        idx = df.abs().sort_values(ascending=False).index
        groups = {str(df.name): [df.name]}
        df = df.to_frame()
    else:
        if df.shape[1] == 2 and '<0' in df.columns and '>0' in df.columns:
            groups = {'': list(df.columns)}
        elif df.shape[1] == 4 and '<0' in df.columns and '>0' in df.columns \
                and '<0 std' in df.columns and '>0 std' in df.columns:
            groups = {'': ['<0', '>0']}
        else:
            cols = {c[:-3] if isinstance(c, str) and (c.endswith('_>0') or c.endswith('_<0')) else c
                    for c in df.columns}
            groups = {str(c): [c0 for c0 in (c, f'{c}_>0', f'{c}_<0') if c0 in df.columns] for c in cols}
        idx = pd.Series(0., index=df.index)
        for columns in groups.values():
            idx = np.maximum(idx, np.maximum(0, df[columns].max(axis=1)) - np.minimum(0, df[columns].min(axis=1)))
        idx = idx.sort_values(ascending=False).index

    if max_features < len(idx) and add_sum_of_remaining:
        df0 = df.reindex(idx[:max_features - 1])
        df0.loc[f'Sum of {len(idx) + 1 -max_features} remaining features'] = \
            df.loc[idx[max_features - 1:]].sum(axis=0)
    else:
        df0 = df.reindex(idx[:max_features])
    return df0, groups
