#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

from pathlib import Path
from typing import Iterable, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

from catabra.automl.base import AutoMLBackend
from catabra.core import config as cfg
from catabra.core.base import CaTabRaBase, Invocation
from catabra.core.paths import CaTabRaPaths
from catabra.monitoring.base import TrainingMonitorBackend
from catabra.ood.base import OODDetector
from catabra.util import common as cu
from catabra.util import io, logging, plotting, statistics
from catabra.util import table as tu
from catabra.util.encoding import Encoder


def analyze(*table: Union[str, Path, pd.DataFrame], classify: Optional[Iterable[Union[str, Path, pd.DataFrame]]] = None,
            regress: Optional[Iterable[Union[str, Path, pd.DataFrame]]] = None, group: Optional[str] = None,
            split: Optional[str] = None, sample_weight: Optional[str] = None, ignore: Optional[Iterable[str]] = None,
            create_stats: Optional[bool] = None, calibrate: Optional[str] = None, time: Optional[int] = None,
            memory: Union[int, str, None] = None, out: Union[str, Path, None] = None,
            config: Union[str, Path, dict, None] = None, default_config: Optional[str] = None,
            monitor: Optional[str] = None, jobs: Optional[int] = None,
            from_invocation: Union[str, Path, dict, None] = None):
    """
    Analyze a table by creating descriptive statistics and training models for predicting one or more columns from
    the remaining ones.  Wrapper for Analyzer.__call__.

    Parameters
    ----------
    *table : str | Path | DataFrame
        The table(s) to analyze. If multiple are given, their columns are merged into a single table.
    classify: Iterable[str | Path | pd.DataFrame], optional
        Column(s) to classify. If more than one, a multilabel classification problem is solved, which means
        that each of these columns can take on only two distinct values. Must be `None` if regress` is given.
    regress: Iterable[str | Path | pd.DataFrame], optional
        Column(s) to regress. Must have numerical or time-like data type. Must be `None` if `classify` is given.
    group: str, optional
        Column used for grouping samples for internal (cross) validation. If not specified or set to `""`, and the row
        index of the given table has a name, group by row index.
    split: str, optional
        Column used for splitting the data into train- and test set. If specified and not `""`,
        descriptive statistics, OOD-detectors and prediction models are generated based exclusively on the training
        split and then automatically evaluated on the test split. The name and/or values of the column must contain the
        string `"train"`, `"test"` or `"val"`, to clearly indicate what is the training- and what is the test data.
    sample_weight: str, optional
        Column with sample weights. If specified and not `""`, must have numeric data type. Sample weights are used both
        for training and evaluating prediction models.
    ignore: Iterable[str], optional
        List of columns to ignore when training prediction models. Automatically includes `group`and `split`, but may
        contain further columns.
    create_stats: bool, optional
        Whether to generate and save descriptive statistics of the given data table.
    calibrate: str, optional
        Value in column `split` defining the subset to calibrate the trained classifier on. If `None`, no calibration
        happens. Ignored in regression tasks or if `split` is not specified.
    time: int, optional
        Time budget for model training, in minutes. Some AutoML backends require a fixed budget, others might not.
        Overwrites the `time_limit` config param.
    memory: int | str, optional
        Memory budget for model training. Integers are assumed in MB, strings can specify the unit explicitly via
        suffix "mb" or "gb" (case-insensitive). Some AutoML backends require a fixed budget, others might not.
        Overwrites the `memory_limit` config param.
    out: str | Path, optional
        Directory where to save all generated artifacts. Defaults to a directory located in the parent directory of
        `table`, with a name following a fixed naming pattern. If `out` already exists, the user is prompted to specify
        whether it should be replaced; otherwise, it is automatically created.
    config: dict | str | Path, optional
        Configuration dict or path to JSON file containing such a dict. Merged with the default configuration
        specified via `default_config`. Empty string means that the default configuration is used.
    default_config: str, optional
        Default configuration to use, one of `full`, `""`, `basic`, `interpretable` or `None`.
    monitor: str, optional
        Training monitor to use.
    jobs: int
        Number of jobs to use. Overwrites the `"jobs"` config param.
    from_invocation: dict | str | Path, optional
        dict or path to an invocation.json file. All arguments of this function not explicitly specified are taken from
        this dict; this also includes the table to analyze.
    """

    analyzer = CaTabRaAnalysis(invocation=from_invocation)
    analyzer(
        *table,
        classify=classify,
        regress=regress,
        group=group,
        split=split,
        sample_weight=sample_weight,
        ignore=ignore,
        create_stats=create_stats,
        calibrate=calibrate,
        time=time,
        memory=memory,
        out=out,
        config=config,
        default_config=default_config,
        monitor=monitor,
        jobs=jobs
    )


class CaTabRaAnalysis(CaTabRaBase):

    @property
    def invocation_class(self) -> Type['AnalysisInvocation']:
        return AnalysisInvocation

    def _call(self):
        if isinstance(self._invocation.table[0], Path):
            dataset_name = self._invocation.table[0].stem
        else:
            dataset_name = None
        self._config = self._get_config_dict()

        out_ok = self._invocation.resolve_output_dir(
            prompt=f'Output folder "{self._invocation.out.as_posix()}" already exists. Delete?'
        )
        if out_ok:
            io.dump(self._config, self._invocation.out / CaTabRaPaths.Config)
        else:
            logging.log('### Aborting')
            return

        # version info
        versions = cu.get_versions()

        cu.save_versions(versions, (self._invocation.out / 'versions.txt').as_posix())
        with logging.LogMirror((self._invocation.out / CaTabRaPaths.ConsoleLogs).as_posix()):
            logging.log(f'### Analysis started at {self._invocation.start}')
            io.dump(io.to_json(self._invocation), self._invocation.out / CaTabRaPaths.Invocation)

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
            if self._invocation.split is not None:
                self._invocation.ignore.add(self._invocation.split)

            # copy training data
            copy_data = self._config.get('copy_analysis_data', False)
            if isinstance(copy_data, (int, float)):
                copy_data = df_train.memory_usage(index=True, deep=True).sum() <= copy_data * 1000000
            if copy_data:
                io.write_df(df_train, self._invocation.out / CaTabRaPaths.TrainData)

            # grouping
            if self._invocation.group is not None:
                self._invocation.ignore.add(self._invocation.group)
            group_indices = self.get_group_indices(df, df_train, self._invocation.group, split_masks)

            # sample weights
            if self._invocation.sample_weight is not None:
                self._invocation.ignore.add(self._invocation.sample_weight)
            sample_weights = self._invocation.get_sample_weights(df_train)

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
            if self._invocation.create_stats:
                statistics.save_descriptive_statistics(df=df.drop(self._invocation.ignore, axis=1, errors='ignore'),
                                                       target=target, classify=self._invocation.classify,
                                                       fn=self._invocation.out / CaTabRaPaths.Statistics)

            # encoder
            encoder = Encoder(classify=self._invocation.classify)
            x_train, y_train = encoder.fit_transform(df_train, y=y_train)
            encoder.dump(self._invocation.out / CaTabRaPaths.Encoder)

            if y_train is not None and (
            self._config.get('time_limit') if self._invocation.time is None else self._invocation.time) != 0:
                automl = self._config.get('automl')
                if automl is not None:
                    backend = AutoMLBackend.get(automl, task=encoder.task_, config=self._config,
                                                tmp_folder=self._invocation.out / automl)
                    if backend is None:
                        raise ValueError(f'Unknown AutoML backend: {automl}')
                    logging.log(f'Using AutoML-backend {automl} for {encoder.task_}')
                    versions.update(backend.get_versions())
                    # overwrite existing file
                    cu.save_versions(versions, (self._invocation.out / 'versions.txt').as_posix())

                    with self._make_training_monitor() as monitor:
                        backend.fit(x_train, y_train, groups=group_indices, sample_weights=sample_weights,
                                    time=self._invocation.time, memory=self._invocation.memory,
                                    jobs=self._invocation.jobs, dataset_name=dataset_name, monitor=monitor)
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

                    self._make_explainer(backend, encoder, x_train, y_train, versions)

            # overwrite existing file
            cu.save_versions(versions, (self._invocation.out / CaTabRaPaths.Versions).as_posix())
            self._make_ood_detector(x_train, y_train)

            end = pd.Timestamp.now()
            logging.log(f'### Analysis finished at {end}')
            logging.log(f'### Elapsed time: {end - self._invocation.start}')
            logging.log(f'### Output saved in {self._invocation.out.as_posix()}')

            if self._invocation.calibrate is not None and self._invocation.split is not None \
                    and encoder.task_ != 'regression':
                from ..calibration import CaTabRaCalibration
                calib = CaTabRaCalibration(invocation=self._invocation_src)
                calib(df,
                      folder=self._invocation.out,
                      split=self._invocation.split,
                      subset=self._invocation.calibrate,
                      sample_weight=self._invocation.sample_weight,
                      out=self._invocation.out / 'calib',
                      jobs=self._invocation.jobs)

            if len(split_masks) > 0:
                from ..evaluation import CaTabRaEvaluation
                evaluate = CaTabRaEvaluation(invocation=self._invocation_src)
                evaluate(df,
                         folder=self._invocation.out,
                         split=self._invocation.split,
                         sample_weight=self._invocation.sample_weight,
                         out=self._invocation.out / 'eval',
                         jobs=self._invocation.jobs,
                         create_stats=self._invocation.create_stats)

    def _get_config_dict(self):
        if isinstance(self._invocation.config_src, (str, Path)):
            self._invocation._config_src = io.make_path(self._invocation.config_src, absolute=True)
            config = io.load(self._invocation.config_src)
        elif isinstance(self._invocation.config_src, dict):
            config = self._invocation.config_src.copy()
        else:
            config = {}
        return cfg.add_defaults(config, default=self._invocation.default_config)

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

    def _make_training_monitor(self) -> TrainingMonitorBackend:
        name, kwargs = self._invocation.get_monitor_with_kwargs()
        monitor = TrainingMonitorBackend.get(name, folder=self._invocation.out.as_posix(), **kwargs)
        if monitor is None:
            if name not in (None, ''):
                logging.warn(f'Training monitor backend "{name}" not found. Disabling live monitoring.')

            class _Aux:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass

            monitor = _Aux()
        else:
            msg = monitor.get_info()
            if msg is not None:
                logging.log(msg)
        return monitor

    def _make_training_plots(self, hist):
        if self._config.get('static_plots', True):
            plotting.save(self.plot_training_history(hist, interactive=False), self._invocation.out)
        if self._config.get('interactive_plots', False):
            plotting.save(self.plot_training_history(hist, interactive=True), self._invocation.out)

    def _make_ood_detector(self, x_train, y_train):
        ood_class = self._config.get('ood_class')
        if ood_class is not None:
            ood = OODDetector.create(ood_class, source=self._config.get('ood_source'),
                                     kwargs=self._config.get('ood_kwargs'))
            ood.fit(x_train, y_train)
            io.dump(ood, self._invocation.out / CaTabRaPaths.OODModel)

    def _make_explainer(self, backend: AutoMLBackend, encoder: Encoder, x_train, y_train,
                        versions) -> Optional['EnsembleExplainer']: # noqa F821
        from ..explanation import EnsembleExplainer
        explainers = self._config.get('explainer') or []
        if not isinstance(explainers, (list, set, tuple)):
            explainers = [explainers]
        unknown = False

        for e in explainers:
            logging.log(f'Creating {e} explainer')

            try:
                explainer = EnsembleExplainer.get(
                    e,
                    config=self._config,
                    ensemble=backend.fitted_ensemble(),
                    feature_names=encoder.feature_names_,
                    target_names=encoder.get_target_or_class_names(),
                    x=x_train,
                    y=y_train
                )

                if explainer is None:
                    logging.warn(f'Unknown explanation backend: {e}')
                    unknown = True
                else:
                    (self._invocation.out / explainer.name).mkdir(exist_ok=True, parents=True)
                    io.dump(explainer.params_, self._invocation.out / explainer.name / 'params.joblib')
                    versions.update(explainer.get_versions())
            except Exception as ex:  # noqa
                logging.warn('Error when creating explainer; skipping\n' + str(ex))

        if unknown:
            logging.warn('Unknown explanation backend(s) specified. Choose among ' +
                         cu.repr_list(EnsembleExplainer.list_explainers()))

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
    def get_group_indices(df, df_train, group, split_masks):
        if group is None and df_train.index.name is not None:
            group = df_train.index.name
            logging.log(f'Grouping by row index "{group}"')

        if group is not None:
            if group == df_train.index.name:
                for k, m in split_masks.items():
                    n = len(np.intersect1d(df_train.index, df[m].index))
                    if n > 0:
                        logging.warn(f'{n} groups in "{k}" overlap with training set')
                if df_train.index.is_unique:
                    group = None
                else:
                    group = df_train.index.values
            elif group in df.columns:
                for k, m in split_masks.items():
                    n = len(np.intersect1d(df_train[group], df.loc[m, group]))
                    if n > 0:
                        logging.warn(f'{n} groups in "{k}" overlap with training set')
                if df_train[group].is_unique:
                    group = None
                else:
                    group = df_train[group].values
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
                        f'{len(obj_cols)} target columns have object data type: ' +
                        cu.repr_list(obj_cols, brackets=False)
                    )

        return df, target

    @staticmethod
    def plot_training_history(hist: Union[pd.DataFrame, str, Path], interactive: bool = False) -> dict:
        """
        Plot the evolution of performance scores during model training.

        Parameters
        ----------
        hist: DataFrame | str | Path
            The history to plot, as saved in "training_history.xlsx".
        interactive: bool, default=False
            Whether to create static Matplotlib plots or interactive plotly plots.

        Returns
        -------
        dict
            Dict with single key "training_history", which is mapped to a Matplotlib or plotly figure object. The sole
            reason for returning a dict is consistency with other plotting functions.
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

        if 'total_elapsed_time' in hist.columns:
            x = hist['total_elapsed_time']
        elif 'timestamp' in hist.columns:
            x = hist['timestamp'] - hist['timestamp'].iloc[0]
        else:
            x = np.arange(len(hist))
        ms = [c for c in hist.columns if
              c.startswith('val_') or c.startswith('train_') or c.startswith('ensemble_val_')]
        opt = ('model_id', 'type', 'ensemble_weight')
        if any(c in hist.columns for c in opt):
            text = [''] * len(hist)
            for c in opt:
                if c in hist.columns:
                    text = [
                        ((t + ', ') if t else t) + c + '=' + ('{:.2f}'.format(v) if isinstance(v, float) else str(v))
                        for t, v in zip(text, hist[c])]
        else:
            text = None
        return dict(training_history=backend.training_history(x, [hist[m] for m in ms], legend=ms, text=text))


class AnalysisInvocation(Invocation):

    @property
    def classify(self) -> Optional[Iterable[Union[str, Path, pd.DataFrame]]]:
        return self._classify

    @property
    def regress(self) -> Optional[Iterable[Union[str, Path, pd.DataFrame]]]:
        return self._regress

    @property
    def target(self) -> Optional[Iterable[str]]:
        return self._target

    @property
    def group(self) -> Optional[str]:
        return self._group

    @property
    def time(self) -> Optional[int]:
        return self._time

    @property
    def memory(self) -> Optional[int]:
        return self._memory

    @property
    def ignore(self) -> Optional[Iterable[str]]:
        return self._ignore

    @property
    def calibrate(self) -> Optional[str]:
        return self._calibrate

    @property
    def config_src(self) -> Union[str, Path, dict, None]:
        return self._config_src

    @config_src.setter
    def config_src(self, value: Union[str, Path, dict, None]):
        self._config_src = value

    @property
    def default_config(self) -> Optional[str]:
        return self._default_config

    @property
    def create_stats(self) -> Optional[bool]:
        return self._create_stats

    def __init__(
            self,
            *table: Union[str, Path, pd.DataFrame],
            sample_weight: Optional[str] = None,
            out: Union[str, Path, None] = None,
            jobs: Optional[int] = None,
            classify: Optional[Iterable[Union[str, Path, pd.DataFrame]]] = None,
            regress: Optional[Iterable[Union[str, Path, pd.DataFrame]]] = None,
            group: Optional[str] = None,
            split: Optional[str] = None,
            time: Optional[int] = None,
            memory: Union[int, str, None] = None,
            ignore: Optional[Iterable[str]] = None,
            calibrate: Optional[str] = None,
            config: Union[str, Path, dict, None] = None,
            default_config: Optional[str] = None,
            monitor: Optional[str] = None,
            create_stats: Optional[bool] = None,
            **_
    ):

        super().__init__(*table, split=split, sample_weight=sample_weight, out=out, jobs=jobs)

        self._classify = classify
        self._regress = regress
        self._group = group
        self._time = time
        self._memory = memory
        self._ignore = ignore
        self._calibrate = calibrate
        self._config_src = config
        self._default_config = default_config
        self._monitor = monitor
        self._target = None
        self._create_stats = create_stats

    def update(self, src: dict = None):
        super().update(src)
        if src:
            if self._classify is None:
                self._classify = src.get('classify')
            if self._regress is None:
                self._regress = src.get('regress')
            if self._group is None:
                self._group = src.get('group')
            if self._time is None:
                self._time = src.get('time')
            if self._memory is None:
                self._memory = src.get('memory')
            if self._ignore is None:
                self._ignore = src.get('ignore')
            if self._calibrate is None:
                self._calibrate = src.get('calibrate')
            if self._config_src is None:
                self._config_src = src.get('config')
            if self._default_config is None:
                self._default_config = src.get('default_config')
            if self._monitor is None:
                self._monitor = src.get('monitor')
            if self._create_stats is None:
                self._create_stats = src.get('create_stats')
            self._target = src.get('target')

    def resolve(self):
        super().resolve()
        if self._group == '':
            self._group = None
        if self._calibrate == '':
            self._calibrate = None
        if self._config_src == '':
            self._config_src = None
        if self._default_config in (None, ''):
            self._default_config = 'full'
        if self._monitor == '':
            self._monitor = None
        if self._create_stats is None:
            self._create_stats = True
        self._ignore = set() if self._ignore is None else set(self._ignore)

        if isinstance(self._memory, str):
            mem = self._memory.strip().lower()
            factor = 1
            if mem.endswith('mb'):
                mem = mem[:-2]
            elif mem.endswith('gb'):
                factor = 1000
                mem = mem[:-2]
            try:
                self._memory = factor * int(mem)
            except ValueError:
                raise ValueError('Invalid memory specification: ' + self._memory)

        if not self._target:
            if self._classify is None:
                if self._regress is None:
                    self._target = []
                else:
                    self._target = [self._regress] if isinstance(self._regress, str) else list(self._regress)
                self._classify = False
            elif self._regress is None:
                self._target = [self._classify] if isinstance(self._classify, str) else list(self._classify)
                self._classify = True
            else:
                raise ValueError('At least one of `classify` and `regress` must be None.')
        elif isinstance(self._regress, str):
            self._target = [self._target]

        if self._out is None:
            self._out = self._table[0]
            if isinstance(self._out, pd.DataFrame):
                raise ValueError('Output directory must be specified when passing a DataFrame.')
            self._out = self._out.parent / (self._out.stem + '_catabra_' + self._start.strftime('%Y-%m-%d_%H-%M-%S'))
        else:
            self.out = io.make_path(self._out, absolute=True)

    def to_dict(self) -> dict:
        dic = super().to_dict()
        dic.update(dict(
            table=['<DataFrame>' if isinstance(tbl, pd.DataFrame) else tbl for tbl in self._table],
            target=['<DataFrame>' if isinstance(tgt, pd.DataFrame) else tgt for tgt in self._target],
            classify=self._classify,
            group=self._group,
            ignore=self._ignore,
            create_stats=self._create_stats,
            calibrate=self._calibrate,
            config=self._config_src,
            default_config=self._default_config,
            monitor=self._monitor,
            time=self._time,
            memory=self._memory,
        ))
        return dic

    @staticmethod
    def requires_table() -> bool:
        return True

    def get_monitor_with_kwargs(self) -> Tuple[Optional[str], dict]:
        if self._monitor in (None, ''):
            return None, {}
        else:
            parts = self._monitor.split(' ')
            return parts[0], dict([p.split('=', 1) for p in parts[1:]])
