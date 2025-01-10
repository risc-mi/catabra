#  Copyright (c) 2025. RISC Software GmbH.
#  All rights reserved.

import numpy as np
import pandas as pd
import pytest
from catabra_lib import preprocessing
from sklearn import cluster, compose, decomposition, feature_selection, impute, pipeline
from sklearn import preprocessing as skl_preprocessing

from catabra.explanation import sklearn_explainer


def _generate_X(n: int, nan: bool, rng) -> pd.DataFrame:
    df = pd.DataFrame(
        data=dict(
            num_1=rng.uniform(-10, 10, size=n),
            datetime=pd.Timestamp('2000-01-01 12:00:00') + pd.to_timedelta(rng.randint(-1000, 1000, size=n), unit='m'),
            bool=rng.uniform(0, 1, size=n) > 0.8,
            cat_2=pd.Categorical.from_codes(rng.randint(2, size=n), categories=['A', 'B']),
            num_2=rng.randn(n) * 2.7 + 1.3,
            cat_5=pd.Categorical.from_codes(rng.randint(6, size=n) - 1, categories=['x', 'y', 'z', 's', 't']),
            obj=rng.randint(10, size=n).astype(str),
            timedelta=pd.to_timedelta(rng.randint(-10000, 10000, size=n), unit='s'),
        )
    )
    if nan:
        df.loc[rng.uniform(0, 1, size=n) > 0.9, 'num_1'] = np.nan
        df.loc[rng.uniform(0, 1, size=n) > 0.8, 'num_2'] = np.nan
    return df


@pytest.mark.parametrize(
    argnames=['n', 'seed'],
    argvalues=[(100, 42), (100, 95047), (50, 30567), (500, 0)]
)
def test_dtype(subtests, n: int, seed: int):
    rng = np.random.RandomState(seed)
    X = _generate_X(n, True, rng)

    dt = preprocessing.DTypeTransformer(
        num=impute.SimpleImputer(strategy='constant', fill_value=0),
        cat=preprocessing.OneHotEncoder(drop_na=True),
        bool=skl_preprocessing.OrdinalEncoder(),
        timedelta=skl_preprocessing.StandardScaler(),
        timedelta_resolution='s',
        datetime=skl_preprocessing.KBinsDiscretizer(n_bins=3),
        datetime_resolution='h',
        obj='drop'
    ).fit(X)

    explainer = sklearn_explainer.sklearn_explainer_factory(dt)
    X_trans = explainer.fit_forward(X, None)

    assert X_trans.shape == (n, 14)

    with subtests.test(msg='local_explanation', i=0):
        importance_0 = rng.uniform(-1, 1, size=X_trans.shape)
        importance = explainer.backward(importance_0)

        assert importance.shape == X.shape
        for i, c in enumerate(X.columns):
            idx = [j for j, c0 in enumerate(X_trans.columns) if c0 == c or c0.startswith(c + '_')]
            assert np.abs(importance[:, i] - importance_0[:, idx].sum(axis=1)).max() < 1e-7
    
    with subtests.test(msg='global_explanation', i=1):
        importance_0 = rng.uniform(-1, 1, size=X_trans.shape[1])
        importance = explainer.backward_global(importance_0)
        
        assert importance.shape == (X.shape[1],)
        for i, c in enumerate(X.columns):
            idx = [j for j, c0 in enumerate(X_trans.columns) if c0 == c or c0.startswith(c + '_')]
            assert np.abs(importance[i] - importance_0[idx].sum()) < 1e-7


@pytest.mark.parametrize(
    argnames=['n', 'seed'],
    argvalues=[(100, 37), (100, 50067), (50, 44444), (500, 888)]
)
def test_pipeline(subtests, n: int, seed: int):
    rng = np.random.RandomState(seed)
    X = _generate_X(n, True, rng)
    X.drop(['obj', 'datetime', 'cat_2', 'timedelta'], axis=1, inplace=True)
    
    ff = preprocessing.FeatureFilter(add_missing=True, remove_unknown=True)

    ct = compose.ColumnTransformer(
        [
            ('impute', impute.SimpleImputer(strategy='most_frequent'), ['num_2', 'cat_5', 'num_1']),
            ('scale', preprocessing.MinMaxScaler(fit_bool=False), ['bool'])
        ],
        remainder='passthrough'
    )

    pip = pipeline.make_pipeline(ff, ct, 'passthrough')
    pip.fit(X)

    explainer = sklearn_explainer.sklearn_explainer_factory(pip)

    X = _generate_X(n, True, rng)
    X.drop(['bool', 'datetime', 'cat_2'], axis=1, inplace=True)
    X_trans = explainer.fit_forward(X, None)

    assert X_trans.shape == (n, 4)  # num_2, cat_5, num_1, bool
    X_trans = pd.DataFrame(data=X_trans, columns=['num_2', 'cat_5', 'num_1', 'bool'])

    with subtests.test(msg='local_explanation', i=0):
        importance_0 = rng.uniform(-1, 1, size=X_trans.shape)
        importance = explainer.backward(importance_0)

        assert importance.shape == X.shape
        for i, c in enumerate(X.columns):
            idx = [j for j, c0 in enumerate(X_trans.columns) if c0 == c or c0.startswith(c + '_')]
            assert np.abs(importance[:, i] - importance_0[:, idx].sum(axis=1)).max() < 1e-7
    
    with subtests.test(msg='global_explanation', i=1):
        importance_0 = rng.uniform(-1, 1, size=X_trans.shape[1])
        importance = explainer.backward_global(importance_0)
        
        assert importance.shape == (X.shape[1],)
        for i, c in enumerate(X.columns):
            idx = [j for j, c0 in enumerate(X_trans.columns) if c0 == c or c0.startswith(c + '_')]
            assert np.abs(importance[i] - importance_0[idx].sum()) < 1e-7


@pytest.mark.parametrize(
    argnames=['n', 'seed'],
    argvalues=[(100, 111111), (100, 6655), (50, 7337), (500, 105040)]
)
def test_missing(subtests, n: int, seed: int):
    rng = np.random.RandomState(seed)
    X = _generate_X(n, True, rng)[['bool', 'num_2', 'num_1']].copy()
    X['bool'] = X.loc[0, 'bool']

    mi = compose.ColumnTransformer(
        [
            ('orig', 'passthrough', list(X.columns)),
            ('mi', impute.MissingIndicator(), ['num_1', 'num_2', 'bool'])
        ]
    )
    si = impute.SimpleImputer(strategy='mean')
    vt = feature_selection.VarianceThreshold()
    pip = pipeline.make_pipeline(mi, si, vt, 'passthrough')

    pip.fit(X)

    explainer = sklearn_explainer.sklearn_explainer_factory(pip)
    X_trans = explainer.fit_forward(X, None)

    assert X_trans.shape == (n, 4)
    X_trans = pd.DataFrame(data=X_trans, columns=['num_2', 'num_1', 'num_1_missing', 'num_2_missing'])

    with subtests.test(msg='local_explanation', i=0):
        importance_0 = rng.uniform(-1, 1, size=X_trans.shape)
        importance = explainer.backward(importance_0)

        assert importance.shape == X.shape
        for i, c in enumerate(X.columns):
            idx = [j for j, c0 in enumerate(X_trans.columns) if c0 == c or c0.startswith(c + '_')]
            assert np.abs(importance[:, i] - importance_0[:, idx].sum(axis=1)).max() < 1e-7
    
    with subtests.test(msg='global_explanation', i=1):
        importance_0 = rng.uniform(-1, 1, size=X_trans.shape[1])
        importance = explainer.backward_global(importance_0)
        
        assert importance.shape == (X.shape[1],)
        for i, c in enumerate(X.columns):
            idx = [j for j, c0 in enumerate(X_trans.columns) if c0 == c or c0.startswith(c + '_')]
            assert np.abs(importance[i] - importance_0[idx].sum()) < 1e-7


@pytest.mark.parametrize(
    argnames=['n', 'seed'],
    argvalues=[(100, 856401), (100, 856377), (50, 7758), (500, 20000)]
)
def test_agglomeration(subtests, n: int, seed: int):
    rng = np.random.RandomState(seed)
    X = _generate_X(n, False, rng)[['num_2', 'num_1']].copy()
    X['num_3'] = X['num_1'] + rng.randn(n) * 0.1 * X['num_1'].std()
    X['num_4'] = X['num_1'] + rng.uniform(-1, 1, size=n) * 0.1 * X['num_1'].std()
    X['num_5'] = X['num_2'] + rng.randn(n) * 0.5 * X['num_2'].std()
    X['num_6'] = X['num_1'] + X['num_2']
    X['num_7'] = X['num_1'] * X['num_2']

    fa = cluster.FeatureAgglomeration(n_clusters=2, pooling_func=np.mean).fit(X)
    assert len(fa.labels_) == X.shape[1]

    explainer = sklearn_explainer.sklearn_explainer_factory(fa)
    X_trans = explainer.fit_forward(X, None)

    assert X_trans.shape == (n, 2)

    with subtests.test(msg='local_explanation', i=0):
        importance_0 = rng.uniform(-1, 1, size=X_trans.shape)
        importance = explainer.backward(importance_0)

        assert importance.shape == X.shape
        for i in range(len(X.columns)):
            assert np.abs(
                importance[:, i] - importance_0[:, fa.labels_[i]] / (fa.labels_ == fa.labels_[i]).sum()
            ).max() < 1e-7
    
    with subtests.test(msg='global_explanation', i=1):
        importance_0 = rng.uniform(-1, 1, size=X_trans.shape[1])
        importance = explainer.backward_global(importance_0)
        
        assert importance.shape == (X.shape[1],)
        for i in range(len(X.columns)):
            assert np.abs(importance[i] - importance_0[fa.labels_[i]] / (fa.labels_ == fa.labels_[i]).sum()) < 1e-7
    
    fa = cluster.FeatureAgglomeration(n_clusters=2, pooling_func=np.max).fit(X)
    assert len(fa.labels_) == X.shape[1]

    explainer = sklearn_explainer.sklearn_explainer_factory(fa)
    X_trans = explainer.fit_forward(X, None)

    assert X_trans.shape == (n, 2)

    with subtests.test(msg='max_pooling', i=0):
        importance_0 = rng.uniform(-1, 1, size=X_trans.shape)
        importance = explainer.backward(importance_0)

        assert importance.shape == X.shape
        for i in range(len(X.columns)):
            mask = fa.labels_[i] == fa.labels_
            mask = mask[np.newaxis] & (X.values == X.values[:, mask].max(axis=1, keepdims=True))
            weights = 1 / mask.sum(axis=1)
            weights[~mask[:, i]] = 0.0
            assert np.abs(importance[:, i] - importance_0[:, fa.labels_[i]] * weights).max() < 1e-7


@pytest.mark.parametrize(
    argnames=['n', 'seed'],
    argvalues=[(100, 104658), (100, 773658), (50, 8577), (500, 2)]
)
def test_pca(n: int, seed: int):
    rng = np.random.RandomState(seed)
    X = _generate_X(n, False, rng)[['num_2', 'num_1']].copy()
    X['num_3'] = X['num_1'] + rng.randn(n) * 0.1 * X['num_1'].std()
    X['num_4'] = X['num_1'] + rng.uniform(-1, 1, size=n) * 0.1 * X['num_1'].std()
    X['num_5'] = X['num_2'] + rng.randn(n) * 0.5 * X['num_2'].std()
    X['num_6'] = X['num_1'] + X['num_2']
    X['num_7'] = X['num_1'] * X['num_2']

    pca = decomposition.PCA(n_components=3).fit(X)

    explainer = sklearn_explainer.sklearn_explainer_factory(pca)
    X_trans = explainer.fit_forward(X, None)

    assert X_trans.shape == (n, 3)

    importance_0 = rng.uniform(-1, 1, size=X_trans.shape)
    importance = explainer.backward(importance_0)
    assert importance.shape == X.shape

    # we can't really compare importance values, since _LinearTransformationExplainer might change at some point
