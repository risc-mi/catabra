#  Copyright (c) 2022-2025. RISC Software GmbH.
#  All rights reserved.

import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from catabra.analysis import analyze
from catabra.util.io import CaTabRaLoader, CaTabRaPaths

from .util import create_synthetic_data

_ROOT = Path(__file__).parent / '_generated'
_OUT = _ROOT / 'output'
_TIME = 1  # global time budget, increase if no models are trained


def _test(
        subtests,
        task: str = 'binary_classification',
        backend: str = 'auto-sklearn',
        multi_process: bool = False,
        holdout: bool = True,
        split: bool = True,
        group: bool = True,
        sample_weight: bool = False,
        single_model: bool = False,
        from_file: bool = False,
        expected_result: Optional[list] = None):
    _ROOT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(_OUT, ignore_errors=True)

    df = create_synthetic_data(task, n_samples=1000, difficulty=2)
    df.to_hdf(_ROOT / 'data.h5', key='table', format='table', complevel=9)

    targets = [c for c in df.columns if c.startswith('_label')]
    if task in ('regression', 'multioutput_regression'):
        targets = dict(regress=targets)
    else:
        targets = dict(classify=targets)

    config = {
        'automl': backend,
        'ensemble_size': 1 if single_model else 10,
        'auto-sklearn': {} if holdout else dict(resampling_strategy='cv',
                                                resampling_strategy_arguments=dict(folds=3)),
        'memory_limit': 9000,
        'ood_class': None  # OOD detection is tested separately elsewhere
    }
    if not holdout:
        config.update(
            {
                'auto-sklearn_resampling_strategy': 'cv',
                'auto-sklearn_resampling_strategy_arguments': dict(folds=3)
            }
        )

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
        config=config
    )

    with subtests.test(msg='output_loading', i=0):
        loader = CaTabRaLoader(_OUT, check_exists=True)
        encoder = loader.get_encoder()
        assert encoder is not None
        model = loader.get_model()
        assert model is not None
        fe = model.fitted_ensemble()

    x = encoder.transform(x=df[df['_test']])  # only tests on tests set, even if `split` is False
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
    #   TODO: Enable the below tests once this is fixed in auto-sklearn.
    if task == 'regression' or backend != 'auto-sklearn':
        assert np.abs(y_model - y_fe).max() < 1e-2

    with subtests.test(msg='files_exist', i=1):
        assert (_OUT / CaTabRaPaths.Config).exists()
        assert (_OUT / CaTabRaPaths.Invocation).exists()
        assert (_OUT / CaTabRaPaths.ConsoleLogs).exists()

    if split and expected_result is not None:
        with subtests.test(msg='expected_range', i=3):
            result = pd.read_excel(_OUT / 'eval/_test/metrics.xlsx', index_col=0)
            for row, col, lower, upper in expected_result:
                r = result.loc[row, col]
                assert lower <= r, f'{row}, {col}: {r}'
                assert r <= upper, f'{row}, {col}: {r}'


@pytest.mark.allowed_to_fail
@pytest.mark.parametrize(
    ids=[
        'sample_weight',
        'no_sample_weight'
    ],
    argnames=['sample_weight'],
    argvalues=[(True,), (False,)]
)
def test_binary_classification(subtests, sample_weight: bool):
    if sample_weight:
        expected_result = [
            ('_label', 'n_weighted', 60, 200),
            ('_label', 'roc_auc', 0.8, 1),
            ('_label', 'average_precision', 0.65, 0.9)
        ]
    else:
        expected_result = [
            ('_label', 'roc_auc', 0.65, 0.8),
            ('_label', 'average_precision', 0.45, 0.55)
        ]

    _test(task='binary_classification', holdout=False, sample_weight=sample_weight, subtests=subtests,
          expected_result=expected_result)


def test_multiclass_classification(subtests):
    _test(task='multiclass_classification', holdout=False, subtests=subtests)


@pytest.mark.slow
def test_multilabel_classification(subtests):
    _test(task='multilabel_classification', from_file=True, subtests=subtests)


@pytest.mark.slow
def test_regression(subtests):
    _test(task='regression', single_model=True, subtests=subtests)


@pytest.mark.allowed_to_fail
def test_regression_expected(subtests):
    _test(task='regression', single_model=True, expected_result=[('_label', 'r2', 0, 1)], subtests=subtests)


@pytest.mark.allowed_to_fail
def test_group(subtests):
    _test(task='multiclass_classification', holdout=True, group=False,
          multi_process=True,
          expected_result=[('_label', 'accuracy', 0.67, 0.77),
                           ('_label', 'balanced_accuracy', 0.5, 0.8)],
          subtests=subtests)


@pytest.mark.slow
def test_split(subtests):
    _test(task='multioutput_regression', holdout=False, split=False,
          multi_process=True, single_model=True, subtests=subtests)
