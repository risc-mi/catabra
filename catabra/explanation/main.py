#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

from catabra.core import CaTabRaBase, Invocation
from catabra.core.paths import CaTabRaPaths
from catabra.util import io, logging, plotting
from catabra.util import table as tu


def explain(*table: Union[str, Path, pd.DataFrame], folder: Union[str, Path] = None, model_id=None,
            explainer: Optional[str] = None, split: Optional[str] = None, sample_weight: Optional[str] = None,
            out: Union[str, Path, None] = None, glob: Optional[bool] = None, jobs: Optional[int] = None,
            batch_size: Optional[int] = None, aggregation_mapping: Optional[Dict[str, List[str]]] = None,
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
    :param explainer: Optional, name of the explainer to use. Defaults to the first explainer specified in config param
    "explainer". Note that only explainers that were fitted to training data during "analyze" can be used, as well as
    explainers that do not need to be fit to training data (e.g., "permutation").
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
    :param aggregation_mapping: Optional, mapping from target column names to lists of source column names in `table`,
    whose explanations will be aggregated by the explainer's aggregation function. Can be either a dict or a JSON file
    containing such a dict.
    Useful for generating joint explanations of certain features, e.g., corresponding to the same variable observed at
    different times.
    :param from_invocation: Optional, dict or path to an invocation.json file. All arguments of this function not
    explicitly specified are taken from this dict; this also includes the table on which to explain the CaTabRa object.
    """
    expl = CaTabRaExplanation(invocation=from_invocation)
    expl(
        *table,
        folder=folder,
        model_id=model_id,
        explainer=explainer,
        glob=glob,
        split=split,
        sample_weight=sample_weight,
        out=out,
        jobs=jobs,
        batch_size=batch_size,
        aggregation_mapping=aggregation_mapping
    )


class CaTabRaExplanation(CaTabRaBase):

    @property
    def invocation_class(self) -> Type['ExplanationInvocation']:
        return ExplanationInvocation

    def _call(self):
        loader = io.CaTabRaLoader(self._invocation.folder, check_exists=True)
        self._config = loader.get_config()

        out_ok = self._invocation.resolve_output_dir(
            prompt=f'Explanation folder "{self._invocation.out.as_posix()}" already exists. Delete?'
        )
        if not out_ok:
            logging.log('### Aborting')
            return

        explainer = loader.get_explainer(explainer=self._invocation.explainer)
        if explainer is None:
            if self._invocation.explainer is not None:
                from .base import EnsembleExplainer
                lst = EnsembleExplainer.list_explainers()
                if self._invocation.explainer not in lst:
                    from ..util.common import repr_list
                    logging.log(f'### Aborting: unknown explanation backend {self._invocation.explainer};'
                                ' choose among ' + repr_list(lst))
                    return
            logging.log('### Aborting: no trained prediction model or no explainer params found')
            return
        behavior = explainer.behavior

        glob = self._invocation.glob
        if glob is None:
            glob = not (behavior.get('supports_local') and (behavior.get('global_is_mean_of_local', False) or
                                                            len(self._invocation.table) > 0))

        with logging.LogMirror((self._invocation.out / CaTabRaPaths.ConsoleLogs).as_posix()):
            logging.log(f'### Explanation started at {self._invocation.start}')
            io.dump(io.to_json(self._invocation), self._invocation.out / CaTabRaPaths.Invocation)

            encoder = loader.get_encoder()

            # merge tables
            df, _ = tu.merge_tables(self._invocation.table)
            if df is None:
                if not glob or behavior.get('global_requires_x', False):
                    raise ValueError('No table(s) to explain models on specified.')
                x_test = None
                y_test = None
            else:
                if df.columns.nlevels != 1:
                    raise ValueError(f'Table must have 1 column level, but found {df.columns.nlevels}.')

                # copy test data
                copy_data = self._config.get('copy_evaluation_data', False)
                if isinstance(copy_data, (int, float)):
                    copy_data = df.memory_usage(index=True, deep=True).sum() <= copy_data * 1000000
                if copy_data:
                    io.write_df(df, self._invocation.out / CaTabRaPaths.ExplanationData)

                x_test = encoder.transform(x=df)
                try:
                    y_test = encoder.transform(y=df)
                except ValueError:
                    y_test = None

            _iter_splits = self._get_split_iterator(df)
            sample_weights = self._invocation.get_sample_weights(df)

            static_plots = self._config.get('static_plots', True)
            interactive_plots = self._config.get('interactive_plots', False)
            if interactive_plots and plotting.plotly_backend is None:
                logging.warn(plotting.PLOTLY_WARNING)
                interactive_plots = False

            model_id = self._invocation.model_id
            if model_id == '__ensemble__':
                model_id = None
            for mask, directory in _iter_splits():
                if mask is not None:
                    logging.log('*** Split ' + directory.stem)
                explain_split(explainer, x=x_test if x_test is None or mask is None else x_test[mask],
                              y=y_test if y_test is None or mask is None else y_test[mask],
                              sample_weight=sample_weights if sample_weights is None or mask is None else
                              sample_weights[mask],
                              directory=directory, glob=glob, model_id=model_id,
                              batch_size=self._invocation.batch_size, jobs=self._invocation.jobs,
                              static_plots=static_plots, interactive_plots=interactive_plots,
                              aggregation_mapping=self._invocation.aggregation_mapping, verbose=True)

            end = pd.Timestamp.now()
            logging.log(f'### Explanation finished at {end}')
            logging.log(f'### Elapsed time: {end - self._invocation.start}')
            logging.log(f'### Output saved in {self._invocation.out.as_posix()}')

    def _get_split_iterator(self, df):
        if df is None:
            def _iter_splits():
                yield None, self._invocation.out
        elif self._invocation.split is None:
            def _iter_splits():
                yield None, self._invocation.out
        else:
            split_masks, _ = tu.train_test_split(df, self._invocation.split)

            def _iter_splits():
                for _k, _m in split_masks.items():
                    yield _m, self._invocation.out / _k

        return _iter_splits


class ExplanationInvocation(Invocation):

    @property
    def folder(self) -> Union[str, Path]:
        return self._folder

    @property
    def model_id(self) -> Optional[str]:
        return self._model_id

    @property
    def explainer(self) -> Optional[str]:
        return self._explainer

    @property
    def glob(self) -> Optional[bool]:
        return self._glob

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    @property
    def aggregation_mapping(self) -> Optional[Dict]:
        return self._aggregation_mapping

    def __init__(
            self,
            *table,
            split: Optional[str] = None,
            sample_weight: Optional[str] = None,
            out: Union[str, Path, None] = None,
            jobs: Optional[int] = None,
            folder: Union[str, Path] = None,
            model_id=None,
            explainer=None,
            glob: Optional[bool] = None,
            batch_size: Optional[int] = None,
            aggregation_mapping: Union[str, Path, Dict, None] = None
    ):
        super().__init__(*table, split=split, sample_weight=sample_weight, out=out, jobs=jobs)
        self._folder = folder
        self._model_id = model_id
        self._explainer = explainer
        self._glob = glob
        self._batch_size = batch_size
        self._aggregation_mapping = aggregation_mapping

    def update(self, src: dict = None):
        super().update(src)
        if src:
            if self._folder is None:
                self._folder = src.get('folder')
            if self._model_id is None:
                self._model_id = src.get('model_id')
            if self._explainer is None:
                self._explainer = src.get('explainer')
            if self._glob is None:
                self._glob = src.get('glob')
            if self._batch_size is None:
                self._batch_size = src.get('batch_size')
            if self._aggregation_mapping is None:
                self._aggregation_mapping = src.get('aggregation_mapping')

    def resolve(self):
        super().resolve()

        if self._folder is None:
            raise ValueError('No CaTabRa directory specified.')
        else:
            self._folder = io.make_path(self._folder, absolute=True)

        if self._out is None:
            self._out = self._table[0] if self._table else None
            if self._out is None or isinstance(self._out, pd.DataFrame):
                self._out = self._folder / ('explain_' + self._start.strftime('%Y-%m-%d_%H-%M-%S'))
            else:
                self._out = \
                    self._folder / ('explain_' + self._out.stem + '_' + self._start.strftime('%Y-%m-%d_%H-%M-%S'))
        else:
            self._out = io.make_path(self._out, absolute=True)
        if self._out == self._folder:
            raise ValueError(
                f'Output directory must differ from CaTabRa directory, but both are "{self._out.as_posix()}".'
            )

        if isinstance(self._aggregation_mapping, str) or isinstance(self._aggregation_mapping, Path):
            with open(self._aggregation_mapping, 'r') as file:
                self._aggregation_mapping = json.load(file)

    def to_dict(self) -> dict:
        dct = super().to_dict()
        dct.update(dict(
            table=['<DataFrame>' if isinstance(tbl, pd.DataFrame) else tbl for tbl in self._table],
            folder=self._folder,
            model_id=self._model_id,
            explainer=self._explainer,
            glob=self._glob,
            batch_size=self._batch_size,
            aggregation_mapping=self._aggregation_mapping
        ))
        return dct

    @staticmethod
    def requires_table() -> bool:
        return False


def explain_split(explainer: 'EnsembleExplainer', x: Optional[pd.DataFrame] = None, # noqa F821
                  y: Optional[pd.DataFrame] = None,  sample_weight: Optional[np.ndarray] = None, directory=None,
                  glob: bool = False, model_id=None, batch_size: Optional[int] = None, jobs: int = 1,
                  aggregation_mapping: Optional[Dict] = None, static_plots: bool = True,
                  interactive_plots: bool = False, verbose: bool = False) -> Optional[dict]:
    """
    Explain a single data split.
    :param explainer: Explainer object.
    :param x: Encoded data to apply `explainer` to, optional unless `glob` is False. Only features, no labels.
    :param y: Encoded data to apply `explainer` to, optional. Only labels, no features.
    :param sample_weight: Sample weights, optional. Ignored if `x` is None or `glob` is False.
    :param directory: Directory where to save the explanations. If None, results are returned in a dict.
    :param glob: Whether to create global explanations.
    :param model_id: ID(s) of the model(s) to explain.
    :param batch_size: Batch size.
    :param jobs: Number of jobs.
    :param aggregation_mapping: Mapping from target column name to list of source columns.
    The source columns' explanations will be aggregated by the explainer's aggregation function.
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

    title = explainer.name + ' Feature Importance'
    if glob:
        explanations: dict = explainer.explain_global(x=x, y=y, sample_weight=sample_weight, jobs=jobs,
                                                      batch_size=batch_size, model_id=model_id,
                                                      mapping=aggregation_mapping, show_progress=verbose)
        if static_plots:
            _save_plots(plot_bars(explanations, interactive=False, title=title), 'static_plots')
        if interactive_plots:
            _save_plots(plot_bars(explanations, interactive=True, title=title), 'interactive_plots')
    else:
        explanations: dict = explainer.explain(x, y=y, jobs=jobs, batch_size=batch_size,
                                               model_id=model_id, mapping=aggregation_mapping, show_progress=verbose)
        if aggregation_mapping is not None:
            x = explainer.aggregate_features(x, aggregation_mapping)

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
        # ignore all columns ending with " std" for which another column without that suffix exists
        all_columns = [c for c in df.columns
                       if not (isinstance(c, str) and c.endswith(' std') and c[:-4] in df.columns)]
        if len(all_columns) == 2 and '<0' in all_columns and '>0' in all_columns:
            groups = {'': all_columns}
        else:
            cols = {c[:-3] if isinstance(c, str) and (c.endswith('_>0') or c.endswith('_<0')) else c
                    for c in all_columns}
            groups = {str(c): [c0 for c0 in (c, f'{c}_>0', f'{c}_<0') if c0 in all_columns] for c in cols}
        idx = pd.Series(0., index=df.index)
        for columns in groups.values():
            idx = np.maximum(idx, np.maximum(0, df[columns].max(axis=1)) - np.minimum(0, df[columns].min(axis=1)))
        idx = idx.sort_values(ascending=False).index

    if max_features < len(idx) and add_sum_of_remaining:
        df0 = df.reindex(idx[:max_features - 1])
        df0.loc[f'Sum of {len(idx) + 1 - max_features} remaining features'] = \
            df.loc[idx[max_features - 1:]].sum(axis=0)
    else:
        df0 = df.reindex(idx[:max_features])
    return df0, groups
