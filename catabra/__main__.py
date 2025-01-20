#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import argparse
import shlex


def make_parser():
    _parser = argparse.ArgumentParser(
        prog='catabra',
        description='Command line interface to the CaTabRa table analyzer tool.'
    )
    _parser.add_argument('--version', action='version', version='%(prog)s ' + '1.0.1')
    subparsers = _parser.add_subparsers(help='The function to invoke.')

    def _add_weights(_p: argparse.ArgumentParser):
        _p.add_argument(
            '-w', '--sample-weight',
            type=str,
            nargs='?',
            const='',
            metavar='SAMPLE_WEIGHT',
            help='Name of the column containing sample weights.'
        )

    def _add_jobs(_p: argparse.ArgumentParser):
        _p.add_argument(
            '-j', '--jobs',
            type=int,
            metavar='JOBS',
            help='The number of jobs to use. -1 means all available processors.'
        )

    def _add_from(_p: argparse.ArgumentParser):
        _p.add_argument(
            '--from',
            type=str,
            metavar='FROM',
            help='The path to an invocation.json file. All command-line arguments not explicitly specified are taken'
                 ' from this file.'
        )

    def _add_stats(_p: argparse.ArgumentParser):
        _p.add_argument(
            '--no-stats',
            action='store_true',
            help='Disable generating descriptive statistics.'
        )

    def _analyze(args: argparse.Namespace):
        from .analysis import analyze
        analyze(*args.table, classify=args.classify, regress=args.regress, group=args.group, split=args.split,
                sample_weight=args.sample_weight, ignore=args.ignore, calibrate=args.calibrate, time=args.time,
                memory=args.memory, out=args.out, config=args.config, default_config=args.default_config,
                monitor=args.monitor, jobs=args.jobs, from_invocation=getattr(args, 'from', None),
                create_stats=not args.no_stats)

    def _evaluate(args: argparse.Namespace):
        from .evaluation import evaluate
        expl = args.explain
        if expl is not None and len(expl) == 0:
            expl = '__all__'
        bs_metrics = args.bootstrapping_metrics
        if bs_metrics == ['__all__']:
            bs_metrics = '__all__'
        evaluate(*(args.on or []), folder=args.src, split=args.split, sample_weight=args.sample_weight,
                 model_id=parse_model_id(args.model_id), explain=expl, glob=getattr(args, 'global'), out=args.out,
                 jobs=args.jobs, batch_size=args.batch_size, threshold=args.threshold, bootstrapping_metrics=bs_metrics,
                 bootstrapping_repetitions=args.bootstrapping_repetitions, from_invocation=getattr(args, 'from', None),
                 create_stats=not args.no_stats, check_ood=not args.no_ood)

    def _explain(args: argparse.Namespace):
        if args.explainer == '':
            # list available explanation backends
            from .explanation.base import EnsembleExplainer
            print('Available explanation backends:\n  ' + '\n  '.join(EnsembleExplainer.list_explainers()))
            return

        from .explanation import explain
        glob = getattr(args, 'global')
        loc = getattr(args, 'local')
        if glob:
            if loc:
                raise ValueError('GLOBAL and LOCAL are mutually exclusive.')
        elif not loc:
            glob = None
        explain(*(args.on or []), folder=args.src, split=args.split, sample_weight=args.sample_weight, glob=glob,
                model_id=parse_model_id(args.model_id), explainer=args.explainer, out=args.out, jobs=args.jobs,
                batch_size=args.batch_size, aggregation_mapping=args.aggregation_mapping,
                from_invocation=getattr(args, 'from', None))

    def _calibrate(args: argparse.Namespace):
        from .calibration import calibrate
        calibrate(*(args.on or []), folder=args.src, split=args.split, subset=args.subset, method=args.method,
                  sample_weight=args.sample_weight, out=args.out, from_invocation=getattr(args, 'from', None))

    def _apply(args: argparse.Namespace):
        from .application import apply
        expl = args.explain
        if expl is not None and len(expl) == 0:
            expl = '__all__'
        apply(*(args.on or []), folder=args.src, model_id=parse_model_id(args.model_id), explain=expl,
              check_ood=not args.no_ood, out=args.out, jobs=args.jobs, batch_size=args.batch_size,
              from_invocation=getattr(args, 'from', None), print_results='auto')

    analyzer = subparsers.add_parser(
        'analyze',
        help='Analyze a table, for instance by training classification- or regression models.'
    )
    analyzer.add_argument(
        'table',
        type=str,
        nargs='*',
        metavar='TABLE',
        help='The table(s) to analyze. Must be CSV- or Excel files, or tables stored in HDF5 files.'
    )
    analyzer.add_argument(
        '-c', '--classify',
        type=str,
        nargs='*',
        metavar='CLASSIFY',
        help='The name of the column(s) containing the variable(s) to classify, or the path to a table.'
             ' Must be omitted if flag "-r" is provided.'
    )
    analyzer.add_argument(
        '-r', '--regress',
        type=str,
        nargs='*',
        metavar='REGRESS',
        help='The name of the column(s) containing the variable(s) to regress, or the path to a table.'
             ' Must be omitted if flag "-c" is provided.'
    )
    analyzer.add_argument(
        '-g', '--group',
        type=str,
        nargs='?',
        const='',
        metavar='GROUP',
        help='Name of the column used for grouping samples for internal (cross) validation. If no GROUP is given,'
             ' samples are grouped by the row index of TABLE, if it has a name; otherwise, grouping is disabled.'
    )
    analyzer.add_argument(
        '-s', '--split',
        type=str,
        nargs='?',
        const='',
        metavar='SPLIT',
        help='Name of the column used for splitting the data into train- and tests set.'
             ' If given, models are trained on the training data and automatically evaluated on all tests splits'
             ' afterward.'
    )
    analyzer.add_argument(
        '--calibrate',
        type=str,
        nargs='?',
        const='',
        metavar='CALIBRATE',
        help='Value in column SPLIT defining the subset to calibrate the trained classifier on. For instance, if the'
             ' column specified by SPLIT contains values "train", "val" and "tests", and CALIBRATION is set to "val",'
             ' the classifier is calibrated on the "val"-entries. If omitted, no calibration happens.'
    )
    _add_weights(analyzer)
    analyzer.add_argument(
        '-i', '--ignore',
        type=str,
        nargs='*',
        metavar='IGNORE',
        help='Names of columns to ignore, typically ID-columns. GROUP, SPLIT and SAMPLE_WEIGHT are always ignored.'
    )
    _add_stats(analyzer)
    analyzer.add_argument(
        '-t', '--time',
        type=int,
        metavar='TIME',
        help='The time budget for training prediction models, in minutes.'
    )
    analyzer.add_argument(
        '-m', '--memory',
        type=str,
        metavar='MEM',
        help='The memory budget for training prediction models, in MB unless explicitly specified otherwise by'
             ' adding suffix "mb" or "gb" (case-insensitive).'
    )
    analyzer.add_argument(
        '-o', '--out',
        type=str,
        metavar='OUT',
        help='The name of the directory where to store the analysis results.'
             ' Defaults to "TABLE_catabra_DATE_TIME" in the parent directory of TABLE,'
             ' where DATE and TIME are the current date and time. "." is a shortcut for the current working directory.'
    )
    analyzer.add_argument(
        '--config',
        type=str,
        nargs='?',
        const='',
        metavar='CONFIG',
        help='The path to an alternative config file. Merged with the default config specified via --default-config.'
    )
    analyzer.add_argument(
        '--default-config',
        type=str,
        metavar='DEFAULT_CONFIG',
        help='Default config to use. Possible values are "full" (default; full range of preprocessing steps and ML'
             ' algorithms for model training), "basic" (only very basic preprocessing and ML algorithms) and'
             ' "interpretable" (only inherently interpretable preprocessing and ML algorithms).'
    )
    analyzer.add_argument(
        '--monitor',
        type=str,
        metavar='MONITOR',
        default='',
        const='plotly',
        nargs='?',
        help='Enable live monitoring of the training progress. The optional MONITOR argument specifies which backend'
             ' to use (default is "plotly").'
             ' Keyword arguments can be passed as sequences of assignments, as in "plotly update_interval=0".'
    )
    _add_jobs(analyzer)
    _add_from(analyzer)
    analyzer.set_defaults(func=_analyze)

    evaluator = subparsers.add_parser(
        'evaluate',
        help='Evaluate an existing CaTabRa object on held-out tests data.'
    )
    evaluator.add_argument(
        'src',
        type=str,
        nargs='?',
        metavar='SOURCE',
        help='The CaTabRa object to evaluate. Must be the path to a folder which was the output directory of a'
             ' previous invocation of `analyze`. "." is a shortcut for the current working directory.'
    )
    evaluator.add_argument(
        '--on',
        type=str,
        nargs='+',
        metavar='ON',
        help='The table(s) on which to evaluate SOURCE. Must be CSV- or Excel files, or tables stored in HDF5 files.'
    )
    evaluator.add_argument(
        '-s', '--split',
        type=str,
        nargs='?',
        const='',
        metavar='SPLIT',
        help='The name of the column containing information about data splits.'
             ' If given, SOURCE is evaluated on each split separately.'
    )
    _add_weights(evaluator)
    _add_stats(evaluator)
    evaluator.add_argument(
        '--no-ood',
        action='store_true',
        help='Disable OOD detection.'
    )
    evaluator.add_argument(
        '-m', '--model-id',
        type=str,
        nargs='?',
        const='__ensemble__',
        metavar='MODEL_ID',
        help='The ID of the prediction model to evaluate. If no MODEL_ID is given, the sole trained model or the whole'
             ' ensemble is evaluated.'
    )
    evaluator.add_argument(
        '-t', '--threshold',
        type=str,
        default=None,
        metavar='THRESHOLD',
        help='Decision threshold for binary- and multilabel classification. Defaults to 0.5, unless specified in FROM.'
             ' In binary classification this can also be the name of a built-in thresholding strategy, possibly'
             ' followed by "on" and the split on which to calculate the threshold. Splits must be specified by the'
             ' name of the subdirectory containing the corresponding evaluation results. See /doc/metrics.md for a'
             ' list of built-in thresholding strategies.'
    )
    evaluator.add_argument(
        '-br', '--bootstrapping-repetitions',
        type=int,
        default=None,
        metavar='BS_REPETITIONS',
        help='Number of bootstrapping repetitions. Set to 0 to disable bootstrapping.'
             ' Overwrites config param "bootstrapping_repetitions".'
    )
    evaluator.add_argument(
        '-bm', '--bootstrapping-metrics',
        type=str,
        nargs='+',
        default=None,
        metavar='BS_METRICS',
        help='Metrics for which to report bootstrapped results. Can also be __all__, in which case all suitable'
             ' metrics are included. Ignored if bootstrapping is disabled.'
    )
    evaluator.add_argument(
        '-e', '--explain',
        type=str,
        nargs='*',
        default=None,
        metavar='EXPLAIN',
        help='Explain prediction model(s). If passed without arguments, all models specified by MODEL_ID are explained;'
             ' otherwise, EXPLAIN contains the model ID(s) to explain.'
    )
    evaluator.add_argument(
        '-g', '--global',
        action='store_true',
        help='Create global explanations. Ignored unless --explain is passed.'
    )
    evaluator.add_argument(
        '-b', '--batch-size',
        type=str,
        metavar='BATCH_SIZE',
        help='The batch size to use for applying the trained prediction model.'
    )
    evaluator.add_argument(
        '-o', '--out',
        type=str,
        metavar='OUT',
        help='The name of the directory where to store the evaluation results.'
             ' Defaults to "SOURCE/eval_ON_DATE_TIME", where DATE and TIME are the current date and time.'
             ' "." is a shortcut for the current working directory.'
    )
    _add_jobs(evaluator)
    _add_from(evaluator)
    evaluator.set_defaults(func=_evaluate)

    explainer = subparsers.add_parser(
        'explain',
        help='Explain an existing CaTabRa object in terms of feature importance.'
    )
    explainer.add_argument(
        'src',
        type=str,
        nargs='?',
        metavar='SOURCE',
        help='The CaTabRa object to explain. Must be the path to a folder which was the output directory of a'
             ' previous invocation of `analyze`. "." is a shortcut for the current working directory.'
    )
    explainer.add_argument(
        '--on',
        type=str,
        nargs='+',
        metavar='ON',
        help='The table(s) on which to explain SOURCE. Must be CSV- or Excel files, or tables stored in HDF5 files.'
             ' Note that in contrast to command `evaluate`, no target columns need to be present.'
    )
    explainer.add_argument(
        '-s', '--split',
        type=str,
        nargs='?',
        const='',
        metavar='SPLIT',
        help='The name of the column containing information about data splits.'
             ' If given, SOURCE is explained on each split separately.'
    )
    _add_weights(explainer)
    explainer.add_argument(
        '-m', '--model-id',
        type=str,
        nargs='+',
        default='__ensemble__',
        metavar='MODEL_ID',
        help='The ID(s) of the prediction model(s) to explain. If no MODEL_ID is given, all models in the ensemble are'
             ' explained, if possible. Note that due to technical restrictions not all models might be explainable.'
    )
    explainer.add_argument(
        '-e', '--explainer',
        type=str,
        nargs='?',
        const='',
        metavar='EXPLAINER',
        help='Name of the explainer to use. Defaults to the first explainer specified in config param "explainer". '
             'Note that only explainers that were fitted to training data during "analyze" can be used, as well as'
             ' explainers that do not need to be fit to training data (e.g., "permutation"). Pass `-e` without'
             ' arguments to get a list of all available explainers.'
    )
    explainer.add_argument(
        '-g', '--global',
        action='store_true',
        help='Create global explanations. If specified, ON might not be required (depends on the explanation backend).'
    )
    explainer.add_argument(
        '-l', '--local',
        action='store_true',
        help='Create local explanations for each sample. Mutually exclusive with GLOBAL.'
    )
    explainer.add_argument(
        '-b', '--batch-size',
        type=str,
        metavar='BATCH_SIZE',
        help='The batch size to use for explaining the CaTabRa object.'
    )
    explainer.add_argument(
        '--aggregation-mapping',
        type=str,
        metavar='AGGREGATION_MAPPING',
        help="Mapping from target column names to lists of source column names in ON, whose explanations will be"
             " aggregated by the explainer's aggregation function. Must be the name of a JSON file containing a"
             " corresponding dict."
    )
    explainer.add_argument(
        '-o', '--out',
        type=str,
        metavar='OUT',
        help='The name of the directory where to store the explanations.'
             ' Defaults to "SOURCE/explain_ON_DATE_TIME", where DATE and TIME are the current date and time.'
             ' "." is a shortcut for the current working directory.'
    )
    _add_jobs(explainer)
    _add_from(explainer)
    explainer.set_defaults(func=_explain)

    calibrator = subparsers.add_parser(
        'calibrate',
        help='Calibrate an existing CaTabRa classifier.'
    )
    calibrator.add_argument(
        'src',
        type=str,
        nargs='?',
        metavar='SOURCE',
        help='The CaTabRa object to calibrate. Must be the path to a folder which was the output directory of a'
             ' previous invocation of `analyze`. "." is a shortcut for the current working directory.'
    )
    calibrator.add_argument(
        '--on',
        type=str,
        nargs='+',
        metavar='ON',
        help='The table(s) on which to calibrate SOURCE. Must be CSV- or Excel files, or tables stored in HDF5 files.'
    )
    calibrator.add_argument(
        '-s', '--split',
        type=str,
        nargs='?',
        const='',
        metavar='SPLIT',
        help='The name of the column containing information about data splits. In conjunction with SUBSET this enables'
             ' restricting the data used for calibration to a subset of TABLE.'
    )
    calibrator.add_argument(
        '--subset',
        type=str,
        nargs='?',
        const='',
        metavar='SUBSET',
        help='Value in column SPLIT to consider for calibration. For instance, if the column specified by SPLIT'
             ' contains values "train", "val" and "tests", and SUBSET is set to "val", the classifier is calibrated'
             ' only on the "val"-entries.'
    )
    calibrator.add_argument(
        '--method',
        type=str,
        nargs='?',
        const='',
        metavar='METHOD',
        help='Calibration method. Must be one of "sigmoid", "isotonic" or "auto".'
    )
    _add_weights(calibrator)
    calibrator.add_argument(
        '-o', '--out',
        type=str,
        metavar='OUT',
        help='The name of the directory where to store generated artifacts.'
             ' Defaults to "SOURCE/calibrate_ON_DATE_TIME", where DATE and TIME are the current date and time.'
             ' "." is a shortcut for the current working directory.'
    )
    _add_from(calibrator)
    calibrator.set_defaults(func=_calibrate)

    applier = subparsers.add_parser(
        'apply',
        help='Apply an existing CaTabRa object to new data.'
    )
    applier.add_argument(
        'src',
        type=str,
        nargs='?',
        metavar='SOURCE',
        help='The CaTabRa object to apply. Must be the path to a folder which was the output directory of a'
             ' previous invocation of `analyze`. "." is a shortcut for the current working directory.'
    )
    applier.add_argument(
        '--on',
        type=str,
        nargs='+',
        metavar='ON',
        help='The table(s) on which to apply SOURCE. Must be CSV- or Excel files, or tables stored in HDF5 files.'
             ' Note that in contrast to command `evaluate`, no target columns need to be present.'
    )
    applier.add_argument(
        '-m', '--model-id',
        type=str,
        nargs='+',
        default='__ensemble__',
        metavar='MODEL_ID',
        help='The ID(s) of the prediction model(s) to apply. If no MODEL_ID is given, all models in the ensemble are'
             ' applied.'
    )
    applier.add_argument(
        '-e', '--explain',
        type=str,
        nargs='*',
        default=None,
        metavar='EXPLAIN',
        help='Explain prediction model(s). If passed without arguments, all models specified by MODEL_ID are explained;'
             ' otherwise, EXPLAIN contains the model ID(s) to explain.'
    )
    applier.add_argument(
        '--no-ood',
        action='store_true',
        help='Disable OOD detection.'
    )
    applier.add_argument(
        '-b', '--batch-size',
        type=str,
        metavar='BATCH_SIZE',
        help='The batch size to use for applying the CaTabRa object.'
    )
    applier.add_argument(
        '-o', '--out',
        type=str,
        metavar='OUT',
        help='The name of the directory where to save all generated artifacts.'
             ' Defaults to "SOURCE/apply_ON_DATE_TIME", where DATE and TIME are the current date and time.'
             ' "." is a shortcut for the current working directory.'
    )
    _add_jobs(applier)
    _add_from(applier)
    applier.set_defaults(func=_apply)

    return _parser


parser = make_parser()


def parse_model_id(model_id):
    """
    This function tries to convert model-IDs given as strings into integers (if possible).
    Strings starting and ending with either " or ' are returned as strings, but without these quotation marks.

    Examples
    --------
        * non-string atoms (int, float, bool, None, etc.) are returned unchanged
        * "12" --> 12
        * "some non-numeric string" --> "some non-numeric string"
        * "'12'" --> "12"
        * "''" --> ""

    Parameters
    ----------
        model_id:
            The model-ID to parse, an atom of any type or a list or set thereof.

    Returns
    -------
    str
        Parsed model-ID(s).
    """
    if isinstance(model_id, (list, set)):
        return model_id.__class__(parse_model_id(m) for m in model_id)
    elif isinstance(model_id, str):
        if len(model_id) >= 2 and model_id[0] in ("'", '"') and model_id[0] == model_id[-1]:
            return model_id[1:-1]
        else:
            try:
                return int(model_id)
            except ValueError:
                return model_id
    else:
        return model_id


def main(*args: str):
    args = [b for a in args for b in shlex.split(a)]
    cmd_args = parser.parse_args(args) if args else parser.parse_args()
    return cmd_args.func(cmd_args)


if __name__ == '__main__':
    main()
