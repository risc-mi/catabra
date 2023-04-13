#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import shutil
from pathlib import Path

import pytest
from catabra.analysis import analyze
from catabra.application import apply
from catabra.automl import fixed_pipeline
from catabra.calibration import calibrate
from catabra.evaluation import evaluate
from catabra.explanation import explain
from catabra.util.summary import summarize_importance, summarize_performance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

from .util import create_synthetic_data

_ROOT = Path(__file__).parent / '_generated'
_OUT = _ROOT / 'output'
_TASKS = ['binary_classification', 'multiclass_classification', 'multilabel_classification',
          'regression', 'multioutput_regression']


@pytest.mark.parametrize(
    ids=_TASKS,
    argnames=['task'],
    argvalues=[(_t,) for _t in _TASKS]
)
def test_workflow(subtests, task: str):
    _ROOT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(_OUT, ignore_errors=True)

    df = create_synthetic_data(task, difficulty=2, frac_unlabeled=0., n_samples=2000)

    targets = [c for c in df.columns if c.startswith('_label')]
    if task in ('regression', 'multioutput_regression'):
        targets = dict(regress=targets)

        fixed_pipeline.register_backend(
            'lr',
            preprocessing=fixed_pipeline.standard_preprocessing(),
            estimator=LinearRegression()
        )
    elif task == 'multilabel_classification':
        targets = dict(classify=targets)
        fixed_pipeline.register_backend(
            'lr',
            preprocessing=fixed_pipeline.standard_preprocessing(),
            estimator=MultiOutputClassifier(LogisticRegression())
        )
    else:
        targets = dict(classify=targets)
        fixed_pipeline.register_backend(
            'lr',
            preprocessing=fixed_pipeline.standard_preprocessing(),
            estimator=LogisticRegression()
        )

    # `analyze` is not tested in a subtest, because we can immediately abort if it fails
    analyze(
        df,
        **targets,
        split='_test',
        sample_weight='_sample_weight',
        ignore=('_group', '_sample_weight'),
        out=_OUT,
        config={
            'automl': 'lr',
            'ood_class': None  # OOD detection is tested separately elsewhere
        }
    )

    with subtests.test(msg='calibrate', i=0):
        if task in ('binary_classification', 'multilabel_classification', 'multiclass_classification'):
            calibrate(df, folder=_OUT, split='_test', subset=True)

    with subtests.test(msg='evaluate', i=1):
        shutil.rmtree(_OUT / 'eval', ignore_errors=True)
        evaluate(
            df,
            folder=_OUT,
            split='_test',
            sample_weight='_sample_weight',
            threshold='balance on _test' if task == 'binary_classification' else None,
            bootstrapping_repetitions=5,
            out=_OUT / 'eval'
        )

    explainer = 'permutation' if task in ('multioutput_regression', 'multilabel_classification') else 'shap'
    with subtests.test(msg='explain', i=2):
        explain(
            df,
            folder=_OUT,
            explainer=explainer,
            split='_test',
            sample_weight='_sample_weight',
            out=_OUT / 'expl'
        )

    with subtests.test(msg='apply', i=3):
        apply(
            df,
            folder=_OUT,
            out=_OUT / 'apply'
        )

    with subtests.test(msg='summarize_performance', i=4):
        if task == 'binary_classification':
            metrics = ['roc_auc(mean)', 'accuracy@0.5', 'average_precision', '__threshold(mean)']
            na_metrics = ['roc_auc(median)', 'r2']
        elif task == 'multiclass_classification':
            metrics = ['balanced_accuracy(50%)', 'roc_auc_ovr', 'mean_average_precision', '-9:f1', '-8:sensitivity']
            na_metrics = ['xxx:f1', 'roc_auc']
        elif task == 'multilabel_classification':
            metrics = ['_label_bool:roc_auc', '__macro__:log_loss', 'f1_macro@0.5', '_label_binary:specificity@0.5',
                       'f1_macro(max)']
            na_metrics = ['accuracy', 'xxx:roc_auc', 'accuracy_micro@1.1', 'accuracy_micro(mean)']
        elif task == 'regression':
            metrics = ['r2(mean)', 'mean_absolute_percentage_error']
            na_metrics = ['accuracy', '__threshold(50%)']
        elif task == 'multioutput_regression':
            metrics = ['mean_absolute_error(std)', 'explained_variance', '_label_float:r2', '__overall__:n']
            na_metrics = ['xxx:max_error', '__threshold(min)']
        else:
            metrics = []
            na_metrics = []

        df = summarize_performance([_OUT / 'eval'], metrics + na_metrics)
        assert len(df) == 2     # = 2 splits
        assert all(m in df.columns for m in metrics)
        assert not df[metrics].isna().any().any()
        assert not any(m in df.columns for m in na_metrics)
        if task == 'binary_classification':
            assert (df['__threshold(mean)'] != 0.5).all()

    with subtests.test(msg='summarize_importance', i=5):
        df = summarize_importance([_OUT / 'expl'], glob=explainer == 'permutation')
        assert len(df) >= 2
        assert len(df) % 2 == 0
        assert df.shape[1] > 1
        assert not df.isna().any().any()
