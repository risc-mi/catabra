from typing import Union, Optional, Iterable, Type
from pathlib import Path
import numpy as np
import pandas as pd

from .config import AnalysisInvocation, AnalysisConfig
from ..core.config import Invocation
from ..ood.base import OODDetector
from ..util import table as tu
from ..util import common as cu
from ..core import logging, io, CaTabRaBase
from ..util import plotting
from ..util.encoding import Encoder
from ..automl.base import AutoMLBackend
from ..util import statistics
from ..core.paths import CaTabRaPaths


def analyze(*table: Union[str, Path, pd.DataFrame], classify: Optional[Iterable[Union[str, Path, pd.DataFrame]]] = None,
            regress: Optional[Iterable[Union[str, Path, pd.DataFrame]]] = None, group: Optional[str] = None,
            split: Optional[str] = None, sample_weight: Optional[str] = None, ignore: Optional[Iterable[str]] = None,
            time: Optional[int] = None, out: Union[str, Path, None] = None, config: Union[str, Path, dict, None] = None,
            default_config: Optional[str] = None, jobs: Optional[int] = None,
            from_invocation: Union[str, Path, dict, None] = None):
    """
    Analyze a table by creating descriptive statistics and training models for predicting one or more columns from
    the remaining ones.
    :param table: The table(s) to analyze. If multiple are given, their columns are merged into a single table.
    :param classify: Optional, column(s) to classify. If more than one, a multilabel classification problem is
    solved, which means that each of these columns can take on only two distinct values.
    Must be None if `regress` is given.
    :param regress: Optional, column(s) to regress. Must have numerical or time-like data type.
    Must be None if `classify` is given.
    :param group: Optional, column used for grouping samples for internal (cross) validation. If not specified or set
    to "", and the row index of the given table has a name, group by row index.
    :param split: Optional, column used for splitting the data into train- and test set. If specified and not "",
    descriptive statistics, OOD-detectors and prediction models are generated based exclusively on the training split
    and then automatically evaluated on the test split. The name and/or values of the column must contain the string
    "train", "test" or "val", to clearly indicate what is the training- and what is the test data.
    :param sample_weight: Optional, column with sample weights. If specified and not "", must have numeric data type.
    Sample weights are used both for training and evaluating prediction models.
    :param ignore: Optional, list of columns to ignore when training prediction models. Automatically includes `group`
    and `split`, but may contain further columns.
    :param time: Optional, time budget for model training, in minutes. Some AutoML backends require a fixed budget,
    others might not. Overwrites the "time_limit" config param.
    :param out: Optional, directory where to save all generated artifacts. Defaults to a directory located in the
    parent directory of `table`, with a name following a fixed naming pattern. If `out` already exists, the user is
    prompted to specify whether it should be replaced; otherwise, it is automatically created.
    :param config: Optional, configuration dict or path to JSON file containing such a dict. Merged with the default
    configuration specified via `default_config`. Empty string means that the default configuration is used.
    :param default_config: Default configuration to use, one of "full", "", "basic", "interpretable" or None.
    :param jobs: Optional, number of jobs to use. Overwrites the "jobs" config param.
    :param from_invocation: Optional, dict or path to an invocation.json file. All arguments of this function not
    explicitly specified are taken from this dict; this also includes the table to analyze.
    """

    analyzer = Analyzer(invocation=from_invocation)
    analyzer(
        *table,
        classify=classify,
        regress=regress,
        group=group,
        split=split,
        sample_weight=sample_weight,
        ignore=ignore,
        time=time,
        out=out,
        config=config,
        default_config=default_config,
        jobs=jobs
    )


class Analyzer(CaTabRaBase):

    @property
    def invocation_class(self) -> Type[Invocation]:
        return AnalysisInvocation

    def _call(
            self,
            *table: Union[str, Path, pd.DataFrame],
            classify: Optional[Iterable[Union[str, Path, pd.DataFrame]]] = None,
            regress: Optional[Iterable[Union[str, Path, pd.DataFrame]]] = None,
            group: Optional[str] = None,
            split: Optional[str] = None,
            sample_weight: Optional[str] = None,
            ignore: Optional[Iterable[str]] = None,
            time: Optional[int] = None,
            out: Union[str, Path, None] = None,
            config: Union[str, Path, dict, AnalysisConfig, None] = None,
            default_config: Optional[str] = None,
            jobs: Optional[int] = None,
    ):

        if len(self._invocation.table) == 0:
            raise ValueError('No table specified.')

        if isinstance(self._invocation.table[0], Path):
            dataset_name = self._invocation.table[0].stem
        else:
            dataset_name = None
        if isinstance(self._invocation.config_src, (str, Path)):
            self._invocation._config_src = io.make_path(self._invocation.config_src, absolute=True)
            config = io.load(self._invocation.config_src)
            self._config = AnalysisConfig(config, self._invocation.default_config)
        elif isinstance(config, AnalysisConfig):
            self._config = config
        else:
            self._config = AnalysisConfig()

        out_ok = self._resolve_output_dir(
            prompt=f'Output folder "{self._invocation.out.as_posix()}" already exists. Delete?'
        )
        if out_ok:
            io.dump(self._config.src, self._invocation.out / CaTabRaPaths.Config)
        else:
            logging.log('### Aborting')
            return

        # version info
        versions = cu.get_versions()

        # why save before and overwrite?
        # cu.save_versions(versions, (self._out / 'versions.txt').as_posix())
        with logging.LogMirror((self._invocation.out / CaTabRaPaths.ConsoleLogs).as_posix()):
            logging.log(f'### Analysis started at {self._invocation.start}')
            invocation_dict = self._invocation.to_dict()
            io.dump(io.to_json(invocation_dict), self._invocation.out / CaTabRaPaths.Invocation)

            # merge tables
            df, id_cols = tu.merge_tables(self._invocation.table)
            if df.columns.nlevels != 1:
                raise ValueError(f'Table must have 1 column level, but found {df.columns.nlevels}.')

            # set target column(s)
            df, target = self.set_target_columns(self._invocation.target, df, id_cols)

            ignore_target = [c for c in target if c in self._invocation.ignore]
            if ignore_target:
                logging.log(f'Ignoring {len(ignore_target)} target columns:',
                            cu.repr_list(ignore_target, brackets=False))
                target = [c for c in target if c not in ignore_target]

            # train-test split
            df_train, split_masks = self._make_train_split(df, self._invocation.split)
            self._invocation.ignore.update(self._invocation.split)

            # copy training data
            copy_data = self._config.copy_analysis_data
            if isinstance(copy_data, (int, float)):
                copy_data = df_train.memory_usage(index=True, deep=True).sum() <= copy_data * 1000000
            if copy_data:
                io.write_df(df_train, self._invocation.out / CaTabRaPaths.TrainData)

            # grouping
            # if self._group in df_train.columns: # check not necessary as errors='ignore')
            self._invocation.ignore.update({self._invocation.group})
            group_indices = self.get_group_indices(df_train, self._invocation.group, split_masks)

            sample_weights = self._get_sample_weights(df_train)
            self.check_for_id_cols(df_train, id_cols, target)

            # drop columns
            if self._invocation.ignore:
                df_train.drop(self._invocation.ignore, axis=1, inplace=True, errors='ignore')

            if len(target) > 0:
                y_train = df_train[target].copy()
                df_train.drop(target, axis=1, inplace=True)
            else:
                y_train = None

            # descriptive statistics for overall dataset
            statistics.save_descriptive_statistics(df=df.drop(self._invocation.ignore, axis=1, errors='ignore'),
                                                   target=target, classify=self._invocation.classify,
                                                   fn =self._invocation.out / CaTabRaPaths.Statistics)

            # encoder
            encoder = Encoder(classify=self._invocation.classify)
            x_train, y_train = encoder.fit_transform(df_train, y=y_train)
            encoder.dump(self._invocation.out / CaTabRaPaths.Encoder)

            # backend = None
            # why time_limit and time?
            if y_train is not None and (self._config.time_limit if self._invocation.time is None else self._invocation.time) != 0:
                if self._config.automl is not None:
                    backend = AutoMLBackend.get(self._config.automl, task=encoder.task_, config=self._config.src,
                                                tmp_folder=self._invocation.out / self._config.automl)
                    if backend is None:
                        raise ValueError(f'Unknown AutoML backend: {self._config.automl}')
                    logging.log(f'Using AutoML-backend {self._config.automl} for {encoder.task_}')
                    versions.update(backend.get_versions())
                    cu.save_versions(versions, (self._invocation.out / 'versions.txt').as_posix())   # overwrite existing file

                    backend.fit(x_train, y_train, groups=group_indices, sample_weights=sample_weights, time=self._invocation.time,
                                jobs=self._invocation.jobs, dataset_name=dataset_name)
                    io.dump(backend, self._invocation.out / CaTabRaPaths.Model)
                    io.dump(io.to_json(backend.summary()), self._invocation.out / CaTabRaPaths.ModelSummary)

                    hist = backend.training_history()
                    io.write_df(hist, self._invocation.out / CaTabRaPaths.TrainingHistory)
                    sub_histories, n_models = self.get_training_stats(hist)
                    msg = ['Final training statistics:', '    n_models_trained: ' + str(n_models)]
                    msg += ['    {}: {}'.format(sub_histories.index[i], sub_histories.iloc[i])
                            for i in range(len(sub_histories))]
                    logging.log('\n'.join(msg))
                    self._make_training_plots(hist)

                    self._make_explainer(self._config.explainer, backend, encoder, x_train, y_train, versions)

            cu.save_versions(versions, (self._invocation.out / 'versions.txt').as_posix())  # overwrite existing file
            self._make_ood_detector(x_train, y_train)

            end = pd.Timestamp.now()
            logging.log(f'### Analysis finished at {end}')
            logging.log(f'### Elapsed time: {end - self._invocation.start}')
            logging.log(f'### Output saved in {self._invocation.out.as_posix()}')

            if len(split_masks) > 0:
                from ..evaluation import Evaluator
                evaluate = Evaluator(invocation=self._invocation_src)
                evaluate(df,
                         folder=self._invocation.out,
                         split=self._invocation.split,
                         sample_weight=self._invocation.sample_weight,
                         out=self._invocation.out / 'eval',
                         jobs=self._invocation.jobs)

    @staticmethod
    def _make_train_split(df, split):
        if split is None:
            split_masks = {}
            df_train = df.copy()
        else:
            split_masks, train_key = tu.train_test_split(df, split)
            train_mask = split_masks.get(train_key)
            if train_mask is None:
                raise ValueError(f'Name and values of train-test-split column "{split}" are ambiguous.')
            elif train_mask.all():
                df_train = df.copy()
            else:
                df_train = df[train_mask].copy()
            del split_masks[train_key]
        return df_train, split_masks

    def _make_training_plots(self, hist):
        if self._config.static_plots:
            plotting.save(plot_training_history(hist, interactive=False), self._invocation.out)
        if self._config.interactive_plots:
            plotting.save(plot_training_history(hist, interactive=True), self._invocation.out)

    def _make_ood_detector(self, x_train, y_train):
        if self._config.ood_class is not None:
            ood = OODDetector.create(
                self._config.ood_class,
                source=self._config.ood_src,
                kwargs=self._config.ood_kwargs
            )
            ood.fit(x_train, y_train)
            io.dump(ood, self._invocation.out / CaTabRaPaths.OODModel)

    def _make_explainer(self, explainer_name: str, backend: AutoMLBackend, encoder: Encoder, x_train, y_train,
                        versions) -> Optional['EnsembleExplainer']:
        from ..explanation import EnsembleExplainer
        logging.log(f'Creating {explainer_name} explainer')

        try:
            explainer = EnsembleExplainer.get(
                explainer_name,
                ensemble=backend.fitted_ensemble(),
                feature_names=encoder.feature_names_,
                target_names=encoder.get_target_or_class_names(),
                x=x_train,
                y=y_train
            )

            if explainer is None:
                logging.warn(f'Unknown explanation backend: {explainer_name}')
            else:
                if explainer:
                    (self._invocation.out / explainer.name()).mkdir(exist_ok=True, parents=True)
                    io.dump(explainer.params_, self._invocation.out / explainer.name() / 'params.joblib')
                    versions.update(explainer.get_versions())
        except Exception as e:  # noqa
            logging.warn(f'Error when creating explainer; skipping\n' + str(e))

    @staticmethod
    def get_training_stats(hist):
        if hist.empty:
            sub_histories = pd.Series([], dtype=np.float32)
        else:
            cols = [c for c in hist.columns if c.startswith('ensemble_val_')]
            if cols:
                if 'timestamp' in hist.columns:
                    sub_histories = hist[cols].iloc[np.argmax(hist['timestamp'].values)]
                else:
                    sub_histories = hist[cols].iloc[-1]
            else:
                sub_histories = pd.Series([], dtype=np.float32)
        n_models = len(hist)
        return sub_histories, n_models

    def check_for_id_cols(self, df, id_cols, target):
        id_cols = [c for c in id_cols if c not in self._invocation.ignore and c not in target]
        if id_cols:
            logging.warn(f'{len(id_cols)} columns appear to contain IDs, but are used as features:',
                         cu.repr_list(id_cols, brackets=False))
        obj_cols = [c for c in df.columns if df[c].dtype.name == 'object' and c not in self._invocation.ignore]
        if obj_cols:
            logging.warn(f'{len(obj_cols)} columns have object data type, and hence cannot be used as features:',
                         cu.repr_list(obj_cols, brackets=False))
            self._invocation.ignore.update(obj_cols)

    @staticmethod
    def get_group_indices(df, group, split_masks):
        if group is None and df.index.name is not None:
            group = df.index.name
            logging.log(f'Grouping by row index "{group}"')

        if group is not None:
            if group == df.index.name:
                for k, m in split_masks.items():
                    n = len(np.intersect1d(df.index, df[m].index))
                    if n > 0:
                        logging.warn(f'{n} groups in "{k}" overlap with training set')
                if df.index.is_unique:
                    group = None
                else:
                    group = df.index.values
            elif group in df.columns:
                for k, m in split_masks.items():
                    n = len(np.intersect1d(df[group], df.loc[m, group]))
                    if n > 0:
                        logging.warn(f'{n} groups in "{k}" overlap with training set')
                if df[group].is_unique:
                    group = None
                else:
                    group = df[group].values
            else:
                raise ValueError(f'"{group}" is no column of the specified table.')
        return group

    @staticmethod
    def set_target_columns(target, df, id_cols):
        if len(target) > 0:
            if len(target) == 1:
                if isinstance(target[0], str) and target[0] not in df.columns:
                    target_path = Path(target[0])
                    if target_path.exists():
                        target = target_path
                elif isinstance(target[0], (Path, pd.DataFrame)):
                    target = target[0]
            if isinstance(target, Path):
                df_target = io.read_df(target)
            elif isinstance(target, pd.DataFrame):
                df_target = target.copy()
            else:
                df_target = None
            if df_target is None:
                missing = [c for c in target if c not in df.columns]
                if missing:
                    raise ValueError(
                        f'{len(missing)} target columns do not appear in the specified table: ' +
                        cu.repr_list(missing, brackets=False)
                    )
            else:
                df_target = tu.convert_object_dtypes(df_target, inplace=True)
                df_target.columns = cu.fresh_name(list(df_target.columns), df.columns)
                if df_target.index.name is None:
                    if df_target.index.nlevels == 1 and len(df) == len(df_target) \
                            and (df_target.index == pd.RangeIndex(len(df_target))).all():
                        df_target.index = df.index
                        df = df.join(df_target, how='left')
                    else:
                        raise RuntimeError('Cannot join target table with features table.')
                elif df_target.index.name == df.index.name:
                    df = df.join(df_target, how='left')
                elif df_target.index.name in id_cols:
                    df = df.join(df_target, how='left', on=df_target.name)
                else:
                    raise RuntimeError('Cannot join target table with features table.')
                target = list(df_target.columns)

                obj_cols = [c for c in df.columns if df[c].dtype.name == 'object' and c in target]
                if obj_cols:
                    raise ValueError(
                        f'{len(obj_cols)} target columns have object data type: ' + cu.repr_list(obj_cols,
                                                                                                 brackets=False)
                    )

        return df, target


def plot_training_history(hist: Union[pd.DataFrame, str, Path], interactive: bool = False) -> dict:
    """
    Plot the evolution of performance scores during model training.
    :param hist: The history to plot, as saved in "training_history.xlsx".
    :param interactive: Whether to create static Matplotlib plots or interactive plotly plots.
    :return: Dict with single key "training_history", which is mapped to a Matplotlib or plotly figure object.
    The sole reason for returning a dict is consistency with other plotting functions.
    """

    if isinstance(hist, (str, Path)):
        hist = io.read_df(hist)

    if len(hist) <= 1:
        return {}
    elif interactive:
        if plotting.plotly_backend is None:
            logging.warn(plotting.PLOTLY_WARNING)
            return {}
        else:
            backend = plotting.plotly_backend
    else:
        backend = plotting.mpl_backend

    if 'timestamp' in hist.columns:
        x = hist['timestamp'] - hist['timestamp'].iloc[0]
    else:
        x = np.arange(len(hist))
    ms = [c for c in hist.columns if c.startswith('val_') or c.startswith('train_') or c.startswith('ensemble_val_')]
    opt = ('model_id', 'type', 'ensemble_weight')
    if any(c in hist.columns for c in opt):
        text = [''] * len(hist)
        for c in opt:
            if c in hist.columns:
                text = [((t + ', ') if t else t) + c + '=' + ('{:.2f}'.format(v) if isinstance(v, float) else str(v))
                        for t, v in zip(text, hist[c])]
    else:
        text = None
    return dict(training_history=backend.training_history(x, [hist[m] for m in ms], legend=ms, text=text))
