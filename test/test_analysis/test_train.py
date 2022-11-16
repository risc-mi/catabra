from typing import Optional
import numpy as np
import pandas as pd
import shutil
from pathlib import Path

from ..util import profile
from .util import create_synthetic_data
from catabra.analysis import analyze
from catabra.util.io import CaTabRaLoader, CaTabRaPaths


_ROOT = Path(__file__).parent / '_generated'
_OUT = _ROOT / 'output'
_TIME = 1       # global time budget, increase if no models are trained


def _test(task: str = 'binary_classification', backend: str = 'auto-sklearn', multi_process: bool = False,
          holdout: bool = True, split: bool = True, group: bool = True, sample_weight: bool = False,
          single_model: bool = False, from_file: bool = False, expected_result: Optional[list] = None, **kwargs):
    _ROOT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(_OUT, ignore_errors=True)

    df = create_synthetic_data(task, **kwargs)
    df.to_hdf(_ROOT / 'data.h5', key='table', format='table', complevel=9)

    targets = [c for c in df.columns if c.startswith('_label')]
    if task in ('regression', 'multioutput_regression'):
        targets = dict(regress=targets)
    else:
        targets = dict(classify=targets)

    analyze(
        _ROOT / 'data.h5' if from_file else df,
        **targets,
        group='_group' if group else None,
        split='_test' if split else None,
        sample_weight='_sample_weight' if sample_weight else None,
        ignore=('_sample_weight', '_group', '_test'),
        time=_TIME + 1 if multi_process else _TIME,
        jobs=2 if multi_process else 1,
        out=_OUT,
        config={
            'automl': backend,
            'ensemble_size': 1 if single_model else 10,
            'auto-sklearn': {} if holdout else dict(resampling_strategy='cv',
                                                    resampling_strategy_arguments=dict(folds=3)),
            'ood': None    # OOD detection is tested separately elsewhere
        }
    )

    loader = CaTabRaLoader(_OUT, check_exists=True)
    encoder = loader.get_encoder()
    assert encoder is not None
    model = loader.get_model()
    assert model is not None
    fe = model.fitted_ensemble()

    x = encoder.transform(x=df[df['_test']])        # only test on test set, even if `split` is False
    if task in ('regression', 'multioutput_regression'):
        y_model = model.predict(x)
        y_fe = fe.predict(x)
    else:
        y_model = model.predict_proba(x)
        y_fe = fe.predict_proba(x)

    # Known bug in auto-sklearn (https://github.com/automl/auto-sklearn/issues/1483):
    #   Softmax is applied to predictions returned by some classifiers even if these predictions already correspond to
    #   probabilities or the task is multilabel classification. This (correctly) does not happen in the FittedEnsemble,
    #   and results in different output. Hence, we unfortunately cannot compare `y_model` and `y_fe`.
    #   TODO: Enable the below test once this is fixed in auto-sklearn.
    if task == 'regression' or backend != 'auto-sklearn':
        assert np.abs(y_model - y_fe).max() < 1e-2

    assert (_OUT / CaTabRaPaths.Config).exists()
    assert (_OUT / CaTabRaPaths.Invocation).exists()
    assert (_OUT / CaTabRaPaths.ConsoleLogs).exists()

    if split and expected_result is not None:
        result = pd.read_excel(_OUT / 'eval/_test/metrics.xlsx', index_col=0)
        for row, col, lower, upper in expected_result:
            r = result.loc[row, col]
            assert lower <= r, f'{row}, {col}: {r}'
            assert r <= upper, f'{row}, {col}: {r}'


def test_binary_classification(sample_weight: bool = False, seed=None):
    if sample_weight:
        expected_result = [
            ('_label', 'n_weighted', 3000, 10000),
            ('_label', 'roc_auc', 0.8, 1),
            ('_label', 'average_precision', 0.65, 0.9),
            ('_label', 'balance_score', 0.7, 0.9)
        ]
    else:
        expected_result = [
            ('_label', 'roc_auc', 0.65, 0.8),
            ('_label', 'average_precision', 0.45, 0.55),
            ('_label', 'balance_score', 0.63, 0.73)
        ]

    _test(task='binary_classification', difficulty=2, frac_unlabeled=0, holdout=False, sample_weight=sample_weight,
          seed=seed, expected_result=expected_result)


def test_multiclass_classification(seed=None):
    _test(task='multiclass_classification', difficulty=1, frac_unlabeled=0.1, holdout=False, seed=seed)


def test_multilabel_classification(seed=None):
    _test(task='multilabel_classification', difficulty=0, frac_unlabeled=0, from_file=True, seed=seed)


def test_regression(seed=None):
    _test(task='regression', difficulty=0, frac_unlabeled=0.2, single_model=True, seed=seed,
          expected_result=[('_label', 'r2', 0.5, 1)])


def test_sample_weight(seed=None):
    test_binary_classification(sample_weight=True, seed=seed)


def test_group(seed=None):
    _test(task='multiclass_classification', difficulty=3, frac_unlabeled=0, holdout=True, group=False,
          multi_process=True, seed=seed,
          expected_result=[('_label', 'accuracy', 0.67, 0.77), ('_label', 'balanced_accuracy', 0.4, 0.5)])


def test_split(seed=None):
    _test(task='multioutput_regression', difficulty=3, frac_unlabeled=0, holdout=False, split=False,
          multi_process=True, single_model=True, seed=seed)


def main(seed=987654):
    tic = pd.Timestamp.now()
    rng = np.random.RandomState(seed=seed)

    s = rng.randint(2 ** 31)
    profile(test_binary_classification, seed=s, _include_kwargs=['seed'])

    s = rng.randint(2 ** 31)
    profile(test_multiclass_classification, seed=s, _include_kwargs=['seed'])

    s = rng.randint(2 ** 31)
    profile(test_multilabel_classification, seed=s, _include_kwargs=['seed'])

    s = rng.randint(2 ** 31)
    profile(test_regression, seed=s, _include_kwargs=['seed'])

    s = rng.randint(2 ** 31)
    profile(test_sample_weight, seed=s, _include_kwargs=['seed'])

    s = rng.randint(2 ** 31)
    profile(test_group, seed=s, _include_kwargs=['seed'])

    s = rng.randint(2 ** 31)
    profile(test_split, seed=s, _include_kwargs=['seed'])

    print('Tests finished successfully!')
    print('    elapsed time:', pd.Timestamp.now() - tic)


if __name__ == '__main__':
    main()
