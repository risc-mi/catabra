import argparse
import shlex

from .__init__ import __version__


def make_parser():
    _parser = argparse.ArgumentParser(
        prog='catabra',
        description='Command line interface to the CaTabRa table analyzer tool.'
    )
    _parser.add_argument('--version', action='version', version='%(prog)s ' + __version__)
    subparsers = _parser.add_subparsers(help='The function to invoke.')

    def _add_jobs(_p: argparse.ArgumentParser):
        _p.add_argument(
            '-j', '--jobs',
            type=int,
            metavar='JOBS',
            help='The number of jobs to use. -1 means all available processors.'
        )

    def _analyze(args: argparse.Namespace):
        from .analysis import analyze
        analyze(*args.table, classify=args.classify, regress=args.regress, group=args.group, split=args.split,
                ignore=args.ignore, t=args.time, out=args.out, config=args.config, jobs=args.jobs)

    analyzer = subparsers.add_parser(
        'analyze',
        help='Analyze a table, for instance by training classification- or regression models.'
    )
    analyzer.add_argument(
        'table',
        type=str,
        nargs='+',
        metavar='TABLE',
        help='The table(s) to analyze. Must be CSV- or Excel files, or tables stored in HDF5 files.'
    )
    analyzer.add_argument(
        '-c', '--classify',
        type=str,
        nargs='+',
        metavar='CLASSIFY',
        help='The name of the column(s) containing the variable(s) to classify, or the path to a table.'
             ' Must be omitted if flag "-r" is provided.'
    )
    analyzer.add_argument(
        '-r', '--regress',
        type=str,
        nargs='+',
        metavar='REGRESS',
        help='The name of the column(s) containing the variable(s) to regress, or the path to a table.'
             ' Must be omitted if flag "-c" is provided.'
    )
    analyzer.add_argument(
        '-g', '--group',
        type=str,
        metavar='GROUP_COL',
        help='The name of the column to group samples by when forming cross-validation splits.'
    )
    analyzer.add_argument(
        '-s', '--split',
        type=str,
        metavar='SPLIT_COL',
        help='The name of the column containing information about train-test splits.'
             ' If given, models are trained on the training data and automatically evaluated on all test splits'
             ' afterward.'
    )
    analyzer.add_argument(
        '-i', '--ignore',
        type=str,
        nargs='*',
        metavar='IGNORE',
        help='Names of columns to ignore, typically ID-columns. GROUP_COL and SPLIT_COL are always ignored.'
    )
    analyzer.add_argument(
        '-t', '--time',
        type=int,
        metavar='TIME',
        help='The time budget for training prediction models, in minutes.'
    )
    analyzer.add_argument(
        '-o', '--out',
        type=str,
        metavar='OUT',
        help='The name of the directory where to store the analysis results.'
             ' Defaults to "TABLE_catabra_DATE_TIME" in the parent directory of TABLE,'
             ' where DATE and TIME are the current date and time.'
    )
    analyzer.add_argument(
        '--config',
        type=str,
        metavar='CONFIG',
        help='The path to an alternative config file.'
    )
    _add_jobs(analyzer)
    analyzer.set_defaults(func=_analyze)

    evaluator = subparsers.add_parser(
        'evaluate',
        help='Evaluate a trained prediction model on held-out test data.'
    )
    _add_jobs(evaluator)
    # TODO

    applier = subparsers.add_parser(
        'apply',
        help='Apply a trained prediction model to new data.'
    )
    _add_jobs(applier)
    # TODO

    return _parser


parser = make_parser()


def main(*args: str):
    args = [b for a in args for b in shlex.split(a)]
    cmd_args = parser.parse_args(args) if args else parser.parse_args()
    return cmd_args.func(cmd_args)


if __name__ == '__main__':
    main()
