from typing import Union, Optional, Iterable
from pathlib import Path
import copy
import shutil
import numpy as np
import pandas as pd

from ..util import io
from ..util import logging
from ..util import table as tu
from ..util import common as cu
from ..util import config as cfg
from ..util import plotting
from ..util.encoding import Encoder
from ..automl.base import AutoMLBackend
from ..util import statistics


def analyze(*table: Union[str, Path, pd.DataFrame], classify: Optional[Iterable[Union[str, Path, pd.DataFrame]]] = None,
            regress: Optional[Iterable[Union[str, Path, pd.DataFrame]]] = None, group: Optional[str] = None,
            split: Optional[str] = None, ignore: Optional[Iterable[str]] = None, time: Optional[int] = None,
            out: Union[str, Path, None] = None, config: Union[str, Path, dict, None] = None,
            jobs: Optional[int] = None, from_invocation: Union[str, Path, dict, None] = None):
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
    :param ignore: Optional, list of columns to ignore when training prediction models. Automatically includes `group`
    and `split`, but may contain further columns.
    :param time: Optional, time budget for model training, in minutes. Some AutoML backends require a fixed budget,
    others might not. Overwrites the "time_limit" config param.
    :param out: Optional, directory where to save all generated artifacts. Defaults to a directory located in the
    parent directory of `table`, with a name following a fixed naming pattern. If `out` already exists, the user is
    prompted to specify whether it should be replaced; otherwise, it is automatically created.
    :param config: Optional, configuration dict or path to JSON file containing such a dict. Merged with the default
    configuration in `util/config.py`. Empty string means that the default configuration is used.
    :param jobs: Optional, number of jobs to use. Overwrites the "jobs" config param.
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
        if classify is None and regress is None:
            target = from_invocation.get('target') or []
            if '<DataFrame>' in target:
                raise ValueError('Invocations must not contain "<DataFrame>" targets.')
            if from_invocation.get('classify', True):
                classify = target
            else:
                regress = target
        if group is None:
            group = from_invocation.get('group')
        if split is None:
            split = from_invocation.get('split')
        if ignore is None:
            ignore = from_invocation.get('ignore')
        if out is None:
            out = from_invocation.get('out')
        if config is None:
            config = from_invocation.get('config')
        if time is None:
            time = from_invocation.get('time')
        if jobs is None:
            jobs = from_invocation.get('jobs')

    if len(table) == 0:
        raise ValueError('No table specified.')
    if classify is None:
        if regress is None:
            target = []
        else:
            target = [regress] if isinstance(regress, str) else list(regress)
        classify = False
    elif regress is None:
        target = [classify] if isinstance(classify, str) else list(classify)
        classify = True
    else:
        raise ValueError('At least one of `classify` and `regress` must be None.')

    start = pd.Timestamp.now()
    table = [io.make_path(tbl, absolute=True) if isinstance(tbl, (str, Path)) else tbl for tbl in table]
    if isinstance(table[0], Path):
        dataset_name = table[0].stem
    else:
        dataset_name = None

    if config is None:
        config = {}
    elif isinstance(config, dict):
        config = copy.deepcopy(config)
    else:
        # if `config` is in `out`, it's better to load it before deleting `out`
        config = io.load(config)
    config = cfg.add_defaults(config)

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

    if group == '':
        group = None
    if split == '':
        split = None
    if config == '':
        config = None
    if isinstance(config, (str, Path)):
        config = io.make_path(config, absolute=True)

    with logging.LogMirror((out / 'console.txt').as_posix()):
        logging.log(f'### Analysis started at {start}')
        invocation = dict(
            table=['<DataFrame>' if isinstance(tbl, pd.DataFrame) else tbl for tbl in table],
            target=['<DataFrame>' if isinstance(tgt, pd.DataFrame) else tgt for tgt in target],
            classify=classify,
            group=group,
            split=split,
            ignore=ignore,
            out=out,
            config=config,
            time=time,
            jobs=jobs,
            timestamp=start
        )
        io.dump(io.to_json(invocation), out / 'invocation.json')
        io.dump(config, out / 'config.json')

        # merge tables
        df, id_cols = tu.merge_tables(table)
        if df.columns.nlevels != 1:
            raise ValueError(f'Table must have 1 column level, but found {df.columns.nlevels}.')

        # set target column(s)
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
        ignore = set() if ignore is None else set(ignore)
        ignore_target = [c for c in target if c in ignore]
        if ignore_target:
            logging.log(f'Ignoring {len(ignore_target)} target columns:', cu.repr_list(ignore_target, brackets=False))
            target = [c for c in target if c not in ignore_target]
        obj_cols = [c for c in df.columns if df[c].dtype.name == 'object' and c in target]
        if obj_cols:
            raise ValueError(
                f'{len(obj_cols)} target columns have object data type: ' + cu.repr_list(obj_cols, brackets=False)
            )
        # train-test split
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
            ignore.update({split})

        # copy training data
        copy_data = config.get('copy_analysis_data', False)
        if isinstance(copy_data, (int, float)):
            copy_data = df_train.memory_usage(index=True, deep=True).sum() <= copy_data * 1000000
        if copy_data:
            io.write_df(df_train, out / 'train_data.h5')

        # grouping
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
            elif group in df_train.columns:
                ignore.update({group})
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

        id_cols = [c for c in id_cols if c not in ignore and c not in target]
        if id_cols:
            logging.warn(f'{len(id_cols)} columns appear to contain IDs, but are used as features:',
                         cu.repr_list(id_cols, brackets=False))
        obj_cols = [c for c in df_train.columns if df_train[c].dtype.name == 'object' and c not in ignore]
        if obj_cols:
            logging.warn(f'{len(obj_cols)} columns have object data type, and hence cannot be used as features:',
                         cu.repr_list(obj_cols, brackets=False))
            ignore.update(obj_cols)

        # drop columns
        if ignore:
            df_train.drop(ignore, axis=1, inplace=True, errors='ignore')

        if len(target) > 0:
            y_train = df_train[target].copy()
            df_train.drop(target, axis=1, inplace=True)
        else:
            y_train = None

        static_plots = config.get('static_plots', True)
        interactive_plots = config.get('interactive_plots', False)
        if interactive_plots and plotting.plotly_backend is None:
            logging.warn(plotting.PLOTLY_WARNING)
            interactive_plots = False

        # descriptive statistics for overall dataset
        statistics.save_descriptive_statistics(df=df.drop(ignore, axis=1, errors='ignore'),
                                               target=target, classify=classify, fn=out / 'statistics')

        # encoder
        encoder = Encoder(classify=classify)
        x_train, y_train = encoder.fit_transform(df_train, y=y_train)
        encoder.dump(out / 'encoder.json')

        if y_train is not None and (config.get('time_limit') if time is None else time) != 0:
            automl = config.get('automl')
            if automl is not None:
                backend = AutoMLBackend.get(automl, task=encoder.task_, config=config, tmp_folder=out / automl)
                if backend is None:
                    raise ValueError(f'Unknown AutoML backend: {automl}')
                logging.log(f'Using AutoML-backend {automl} for {encoder.task_}')
                backend.fit(x_train, y_train, groups=group, time=time, jobs=jobs, dataset_name=dataset_name)
                io.dump(backend, out / 'model.joblib')
                io.dump(io.to_json(backend.summary()), out / 'model_summary.json')
                hist = backend.training_history()
                io.write_df(hist, out / 'training_history.xlsx')
                if hist.empty:
                    sub = pd.Series([])
                else:
                    cols = [c for c in hist.columns if c.startswith('ensemble_val_')]
                    if cols:
                        if 'timestamp' in hist.columns:
                            sub = hist[cols].iloc[np.argmax(hist['timestamp'].values)]
                        else:
                            sub = hist[cols].iloc[-1]
                    else:
                        sub = pd.Series([])
                msg = ['Final training statistics:', '    n_models_trained: ' + str(len(hist))]
                msg += ['    {}: {}'.format(sub.index[i], sub.iloc[i]) for i in range(len(sub))]
                logging.log('\n'.join(msg))
                if static_plots:
                    plotting.save(plot_training_history(hist, interactive=False), out)
                if interactive_plots:
                    plotting.save(plot_training_history(hist, interactive=True), out)
                logging.log('Finished model building')

                explainer = config.get('explainer')
                if explainer is not None:
                    from ..explanation import EnsembleExplainer
                    logging.log(f'Creating {explainer} explainer')
                    try:
                        explainer = EnsembleExplainer.get(
                            explainer,
                            ensemble=backend.fitted_ensemble(),
                            feature_names=encoder.feature_names_,
                            target_names=encoder.get_target_or_class_names(),
                            x=x_train,
                            y=y_train
                        )
                    except Exception as e:      # noqa
                        logging.warn(f'Error when creating explainer; skipping\n' + str(e))
                    else:
                        if explainer is None:
                            logging.warn(f'Unknown explanation backend: {config["explainer"]}')
                        else:
                            (out / explainer.name()).mkdir(exist_ok=True, parents=True)
                            io.dump(explainer.params_, out / explainer.name() / 'params.joblib')

        end = pd.Timestamp.now()
        logging.log(f'### Analysis finished at {end}')
        logging.log(f'### Elapsed time: {end - start}')
        logging.log(f'### Output saved in {out.as_posix()}')

    if len(split_masks) > 0:
        from .. import evaluation
        evaluation.evaluate(df, folder=out, split=split, out=out / 'eval', jobs=jobs)


def plot_training_history(hist: pd.DataFrame, interactive: bool = False) -> dict:
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