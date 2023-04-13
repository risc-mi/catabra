#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import re
from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from catabra.explanation import average_local_explanations
from catabra.util.io import Path, make_path, read_dfs
from catabra.util.logging import warn

_METRICS_REGEX = re.compile(r'(?:([^:]+):)?([^:@() ]+)(?:@([0-1]?\.?[0-9]*))?(?:\((.+)\))?')


def summarize_performance(directories: Iterable[Union[str, Path]], metrics: Iterable[Union[str, tuple]],
                          split: Union[str, Iterable[str], None] = None, path_callback=None) -> pd.DataFrame:
    """
    Summarize the performance of prediction models trained and evaluated with CaTabRa.

    IMPORTANT: Only pre-evaluated metrics in "metrics.xlsx" and "bootstrapping.xlsx" are considered!

    Parameters
    ----------
    directories: Iterable[str | Path]
        The directories to consider, an iterable of path-like objects. Each directory must be the output directory of
        an invocation of `catabra.evaluate`, or a subdirectory corresponding to a specific split (containing
        "metrics.xlsx" and maybe also "bootstrapping.xlsx"). A convenient way to specify a couple of directories
        matching a certain pattern is by using `Path(root_path).rglob(pattern)`.
    metrics: Iterable[str]
        List of metrics to include in the summary, an iterable of strings. Values must match the following pattern:

        ::

          [target:]metric_name[@threshold][(bootstrapping_aggregation)]

        * `target` is optional and specifies the target (or class in case of multiclass classification); can be "*" to
          include all available targets, and can be a sequence separated by ",". Ignored if
          `bootstrapping_aggregation` is specified.
        * `metric_name` is the name of the actual metric, exactly as written in "metrics.xlsx" or "bootstrapping.xlsx";
          can be "*" to include all available pre-evaluated metrics, and can be a sequence separated by ",".
        * `threshold` is optional and must be a numeral between 0 and 1 (cannot be a string like "balance"), and cannot
          be "*". Only relevant for threshold-dependent classification metrics, and mutually exclusive with
          `bootstrapping_aggregation`. Note that the given threshold must exactly match one of the thresholds
          evaluated in "metrics.xlsx".
        * `bootstrapping_aggregation` is optional and specifies the bootstrapping aggregation to include, like "mean",
          "std", etc.; can be "*" to include all available pre-evaluated aggregations in "bootstrapping.xlsx", and
          can be a sequence separated by ",".

    split: Iterable[str], default=None
        If a directory in `directories` has subdirectories corresponding to data splits that were evaluated separately,
        only include the splits in `split`. If None, all splits are included.
    path_callback: Callable, default=None
        Callback function applied to every path visited. Must return None, True, False or a dict; False indicates that
        the current path should be dropped from the output, True and None are aliases for `{}`, and a dict adds a
        column for every key to the output DataFrame, with the corresponding values in them.

    Returns
    -------
    DataFrame
        One row per evaluation and one column per performance metric. If multiple splits are included in the
        performance summary, each is put into a separate row.

    Examples
    --------
    Example metric specifications:

        * `"roc_auc"`
        * `"roc_auc(mean,std)"`
        * `"accuracy,sensitivity@0.5"`
        * `"*@0.5"`
        * `"r2(*)"`
        * `"*(*)"`
        * `"target_1:mean_squared_error"`
        * `"*:mean_squared_error"`
        * `"*:*(*)"`
        * `"__threshold(mean,std)"`

    See Also
    --------
    summarize_importance : Summarize feature importance scores.
    """

    if isinstance(directories, (str, Path)):
        directories = [directories]
    if isinstance(metrics, str):
        metrics = [metrics]
    if isinstance(split, str):
        split = [split]

    aux = []
    for m in metrics:
        if isinstance(m, str):
            match = _METRICS_REGEX.fullmatch(m)
            if match is None:
                raise ValueError(
                    f'"{m}" is no valid metric specifier. It must match'
                    ' "[target:]metric_name[@threshold][(bootstrapping_aggregation)]".'
                )
            target, name, threshold, agg = match.groups()
            if target is not None:
                target = target.split(',')
                if '*' in target:
                    target = ['*']
            if name is None:
                raise ValueError(f'"{m}" lacks the name of a metric.')
            else:
                name = name.split(',')
                if '*' in name:
                    name = ['*']
            if threshold is not None:
                if agg is not None:
                    raise ValueError(f'"{m}" specifies both threshold and bootstrapping aggregations.')
                threshold = float(threshold)
            elif agg is not None:
                agg = agg.split(',')
                if '*' in agg:
                    agg = ['*']
                if target is not None:
                    warn('Bootstrapped results can only be summarized for the overall target,'
                         ' not for specific targets individually.')
                    target = None
            m = (target, name, threshold, agg)
        else:
            assert isinstance(m, tuple) and len(m) == 4

        aux.append(m)

    metrics = aux

    if path_callback is None:
        def path_callback(_):
            return True

    dfs = []
    for d in directories:
        if not isinstance(d, (str, Path)):
            dfs.append(summarize_performance(d, metrics, split=split, path_callback=path_callback))
            continue

        d = make_path(d, absolute=True)
        if d.is_dir() and d.exists():
            if (d / 'metrics.xlsx').exists() or (d / 'bootstrapping.xlsx').exists():
                candidates = [d]
            else:
                candidates = [d0 for d0 in d.iterdir() if d0.is_dir()]
                if split is not None:
                    candidates = [c for c in candidates if c.stem in split]
            dfs.append(_collect_evaluations_from_directories(candidates, metrics, path_callback))

    return pd.concat(dfs, axis=0, sort=False, ignore_index=True) if dfs else pd.DataFrame()


def summarize_importance(directories: Iterable[Union[str, Path]], columns: Union[str, Iterable[str], None] = None,
                         new_column_name: str = '{feature} {column}', glob: bool = False,
                         split: Union[str, Iterable[str], None] = None,
                         model_id: Union[str, Iterable[str], None] = None, path_callback=None) -> pd.DataFrame:
    """
    Summarize the feature importance of prediction models trained and explained with CaTabRa.

    IMPORTANT: Only pre-evaluated feature importance scores are considered!

    Parameters
    ----------
    directories: Iterable[str | Path]
        The directories to consider, an iterable of path-like objects. Each directory must be the output directory of
        an invocation of `catabra.explain`, or a subdirectory corresponding to a specific split (containing HDF5 files
        with feature importance scores). A convenient way to specify a couple of directories matching a certain pattern
        is by using `Path(root_path).rglob(pattern)`.
    columns: Iterable[str], default=None
        The columns in global feature importance scores to consider. For instance, if
        `catabra.explanation.average_local_explanations()` is used to produce global scores, 4 columns ">0", "<0",
        ">0 std" and "<0 std" are normally generated. This parameter allows to include only a subset in the summary.
        None defaults to all columns.
    new_column_name: str
        String pattern specifying the names of the columns in the output DataFrame. May have two named fields `feature`
        and `column`, which are filled with original feature- and column names, respectively.
    glob: bool
        Whether feature importance scores in `directories` are global. If not,
        `catabra.explanation.average_local_explanations()` is applied.
    split: Iterable[str], default=None
        If a directory in `directories` has subdirectories corresponding to data splits that were explained separately,
        only include the splits in `split`. If None, all splits are included.
    model_id: Iterable[str], default=None
        Model-IDs to consider, optional. Determines the names of the HDF5 files to be included. None defaults to all
        found model-IDs.
    path_callback: Callable, default=None
        Callback function applied to every path visited. Must return None, True, False or a dict; False indicates that
        the current path should be dropped from the output, True and None are aliases for `{}`, and a dict adds a
        column for every key to the output DataFrame, with the corresponding values in them.

    Returns
    -------
    DataFrame
        One row per explanation and one column per feature-column pair. If multiple splits are included in the
        importance summary, each is put into a separate row. If there are multiple targets (multiclass/multilabel
        classification, multioutput regression) and the feature importance scores for each target are stored in a
        separate table, each is put into a separate row and an additional column "__target__" is added.

    See Also
    --------
    summarize_performance : Summarize model performance.
    """

    if isinstance(directories, (str, Path)):
        directories = [directories]
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(split, str):
        split = [split]
    if isinstance(model_id, str):
        model_id = [model_id]
    if path_callback is None:
        def path_callback(_):
            return True

    dfs = []
    for d in directories:
        if not isinstance(d, (str, Path)):
            dfs.append(
                summarize_importance(d, columns=columns, new_column_name=new_column_name, glob=glob, split=split,
                                     model_id=model_id, path_callback=path_callback)
            )
            continue

        d = make_path(d, absolute=True)
        if d.is_dir() and d.exists():
            if any(f.suffix.lower() == '.h5' for f in d.iterdir()):
                candidates = [d]
            else:
                candidates = [d0 for d0 in d.iterdir() if d0.is_dir()]
                if split is not None:
                    candidates = [c for c in candidates if c.stem in split]
            candidates = [f for c in candidates for f in c.iterdir()
                          if f.suffix.lower() == '.h5' and (model_id is None or f.stem in model_id)]
            dfs.append(
                _collect_explanations_from_directories(candidates, columns, new_column_name, glob, path_callback)
            )

    return pd.concat(dfs, axis=0, sort=False, ignore_index=True) if dfs else pd.DataFrame()


def _collect_evaluations_from_directories(dirs: List[Path], metrics: List[tuple], path_callback) -> pd.DataFrame:
    bs_metrics = [(m[1], m[3]) for m in metrics if m[-1] is not None]
    other_metrics = [m[:-1] for m in metrics if m[-1] is None]
    df = pd.DataFrame(index=pd.RangeIndex(len(dirs)), data={'__path__': None})
    mask = pd.Series(index=df.index, data=True)

    for i, d in enumerate(dirs):
        res = path_callback(d)
        if res is False:
            mask.loc[i] = False
            continue
        elif res in (None, True):
            res = {}
        elif not isinstance(res, dict):
            raise ValueError(f'path_callback must return None, True, False or a dict, but got {res} ({type(res)})')

        df.loc[i, '__path__'] = d.as_posix()

        for k, v in res.items():
            if k not in df.columns:
                df[k] = None
            df.loc[i, k] = v

        if bs_metrics:
            if (d / 'bootstrapping.xlsx').exists():
                bs = pd.read_excel(d / 'bootstrapping.xlsx', index_col=0)
                for names, aggs in bs_metrics:
                    if '*' in names:
                        names = bs.columns
                    else:
                        names = [n for n in names if n in bs.columns]
                    if '*' in aggs:
                        aggs = bs.index
                    else:
                        aggs = [a for a in aggs if a in bs.index]
                    for name in names:
                        for agg in aggs:
                            col = f'{name}({agg})'
                            if col not in df.columns:
                                df[col] = np.nan
                            df.loc[i, col] = bs.loc[agg, name]

        if other_metrics:
            if (d / 'metrics.xlsx').exists():
                met = pd.read_excel(d / 'metrics.xlsx', index_col=0, sheet_name=None)

                # there are several possibilities:
                #   * binary: `met` contains two relevant tables "overall" and "thresholded"
                #       * "overall" has one single row with name of target on index and one column per metric
                #       * "thresholded" has a "threshold" column and one column per metric
                #   * multiclass: `met` contains two relevant tables "overall" and "per_class"
                #       * "overall" is like in binary classification (1 row)
                #       * "per_class" has one row per class and one column per metric, with class names on index
                #   * multilabel: `met` contains several relevant tables, "overall", "thresholded" and one per target
                #       * "overall" has one row per target, plus "__micro__", "__macro__" and "__weighted__", and one
                #           column per metric
                #       * "thresholded" is like in binary classification for micro/macro/weighted averaged metrics
                #       * per-target tables are like "thresholded", but for each target individually
                #   * regression: `met` contains one relevant table with unspecified name, with one row per target plus
                #       "__overall__", and one column per metric
                if len(met) == 1:
                    k = list(met)[0]
                    all_targets = {idx: k for idx in list(met.values())[0].index}
                elif 'overall' in met:
                    all_targets = {idx: 'overall' for idx in met['overall'].index}
                    if 'per_class' in met:
                        all_targets.update({idx: 'per_class' for idx in met['per_class'].index})
                else:
                    all_targets = {}
                # actual targets could be non-strings, so we must consider their string representations to handle
                # string specifications
                str_targets = {str(k): k for k in all_targets if not isinstance(k, str)}

                for targets, names, threshold in other_metrics:
                    if targets is None:
                        if len(all_targets) == 1:
                            # binary classification
                            targets = [(None, t) for t in all_targets]
                        elif '__overall__' in all_targets:
                            # regression
                            targets = [(None, '__overall__')]
                        else:
                            # multiclass or multilabel classification
                            sub = [t for t, s in all_targets.items() if s == 'overall']
                            if len(sub) == 1:
                                # multiclass
                                targets = [(None, sub[0])]
                            else:
                                # multilabel
                                if threshold is None:
                                    # cannot fetch non-thresholded metric without specifying target
                                    targets = []
                                else:
                                    targets = [(None, 'thresholded')]
                    elif '*' in targets:
                        targets = [(t, t) for t in all_targets]
                    else:
                        aux = []
                        for t in targets:
                            if t in all_targets:
                                aux.append((t, t))
                            elif isinstance(t, str):
                                t0 = str_targets.get(t)
                                if t0 is not None:
                                    aux.append((t, t0))
                        targets = aux

                    for t_name, t in targets:
                        if threshold is None:
                            non_thresholded = met[all_targets[t]]
                            for name in (non_thresholded.columns if '*' in names else names):
                                if name in non_thresholded.columns:
                                    col = str(name)
                                    if t_name is not None:
                                        col = str(t_name) + ':' + col
                                    if col not in df.columns:
                                        df[col] = np.nan
                                    df.loc[i, col] = non_thresholded.loc[t, name]
                        else:
                            thresholded = met.get(t, None)
                            if thresholded is None:
                                thresholded = met.get('thresholded', None)
                            if thresholded is not None and 'threshold' in thresholded.columns:
                                j = np.abs(thresholded['threshold'] - threshold).idxmin()
                                if abs(thresholded.loc[j, 'threshold'] - threshold) < 1e-4:
                                    for name in (thresholded.columns if '*' in names else names):
                                        if name in thresholded.columns:
                                            col = f'{name}@{threshold}'
                                            if t_name is not None:
                                                col = str(t_name) + ':' + col
                                            if col not in df.columns:
                                                df[col] = np.nan
                                            df.loc[i, col] = thresholded.loc[j, name]

    return df[mask]


def _collect_explanations_from_directories(files: List[Path], columns: Optional[Iterable[str]], new_column_name: str,
                                           glob: bool, path_callback) -> pd.DataFrame:
    dfs = []
    for f in files:
        res = path_callback(f)
        if res is False:
            continue
        elif res in (None, True):
            res = {}
        elif not isinstance(res, dict):
            raise ValueError(f'path_callback must return None, True, False or a dict, but got {res} ({type(res)})')

        exs = read_dfs(f)

        data = {'__path__': f.as_posix()}
        if len(exs) > 1:
            data['__target__'] = list(exs)
        data.update(res)
        df = pd.DataFrame(index=pd.RangeIndex(len(exs)), data=data)

        for i, (target, ex) in enumerate(exs.items()):
            if not glob:
                ex = average_local_explanations(ex)
            if columns is not None:
                ex = ex[[c for c in columns if c in ex.columns]]

            for feat in ex.index:
                for col in ex.columns:
                    new_name = new_column_name.format(feature=feat, column=col)
                    if new_name not in df.columns:
                        df[new_name] = np.nan
                    df.loc[i, new_name] = ex.loc[feat, col]

        dfs.append(df)

    df = pd.concat(dfs, axis=0, sort=False, ignore_index=True) if dfs else pd.DataFrame(index=[], columns=['__path__'])
    return df
