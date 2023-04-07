#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import glob
import importlib
import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import catabra.ood.internal as internal
from catabra.ood.base import (
    FeaturewiseOODDetector,
    OverallOODDetector,
    SamplewiseOODDetector,
)

# TODO: skaltenl include PYOD tests

# -- helpers -----------------------------------------------------------------------------------------------------------

class DataSets:
    Iris = 0,
    BreastCancer = 1,
    Diabetes = 2


def _iterate_ood_classes():
    files = glob.glob(internal.__path__[0] + '/*[!_].py')
    for file in files:
        name = file.split('/')[-1].split('.')[0]
        module = importlib.import_module('catabra.ood.internal.' + name)
        module_classes = inspect.getmembers(module, lambda x: inspect.isclass(x))

        ood_class = next(
            class_ for class_name, class_ in module_classes if class_name.lower() == name.replace('_', '')
        )
        yield ood_class


def _load_test_data(dataset: int) -> pd.DataFrame:
    if dataset == DataSets.Diabetes:
        return load_diabetes(return_X_y=True, as_frame=True)[0]
    elif dataset == DataSets.BreastCancer:
        return load_breast_cancer(return_X_y=True, as_frame=True)[0]
    else:
        return load_iris(return_X_y=True, as_frame=True)[0]

# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    argnames=['datagen', 'x_shape', 'y_shape'],
    argvalues=[
        (np.random.random, 10,20),
        (lambda size: np.zeros(shape=size), 5, 40),
        (lambda size: np.random.randint(1000, size=size), 30,5)
    ],
    ids=['normal', 'zeros', 'int']
)
def test_output_sanity(x_shape: int, y_shape: int, datagen):
    """
    Tests for each ood detector whether the output has the expected shape and values are within a valid range.

    Parameters
    ----------
    x_shape: int
        Number of rows.
    y_shape: int
        Number of cols.
    datagen:
        Function that generates random data
    """
    data = pd.DataFrame(datagen(size=(x_shape, y_shape)), columns=[str(nr) for nr in range(y_shape)])

    for ood_class in _iterate_ood_classes():
        ood = ood_class(verbose=False)
        ood.fit(data)
        proba = ood.predict_proba(data)
        pred = ood.predict(data)

        # test shape
        if isinstance(ood, FeaturewiseOODDetector):
            assert proba.shape == (ood._transform(data).shape[1],)
            assert pred.shape == (ood._transform(data).shape[1],)
        elif isinstance(ood, SamplewiseOODDetector):
            assert proba.shape == (x_shape,)
            assert pred.shape == (x_shape,)
        elif isinstance(ood, OverallOODDetector):
            assert type(proba) == float
            assert type(pred) == int
        else:
            assert False

        # test range
        assert np.all((proba >= 0) & (proba <= 1))
        assert np.all((pred == 0) | (pred == 1))


@pytest.mark.allowed_to_fail
@pytest.mark.parametrize(
    argnames=['dataset'],
    argvalues=[(DataSets.Diabetes,), (DataSets.BreastCancer,), (DataSets.Iris,)],
    ids=['diabetes', 'breast_cancer', 'iris']
)
def test_proba_sanity(dataset: int):
    """
    Tests for each OOD detector whether higher manipulation of values cause the OOD proba values to be greater or equal
    than for the less manipulated data
    :param dataset: numpy data set as defined in class DataSets
    """
    data = _load_test_data(dataset)
    train, test = train_test_split(data, random_state=10, test_size=0.4)
    scaler = MinMaxScaler()
    train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
    test = pd.DataFrame(scaler.transform(test), columns=test.columns)

    for ood_class in _iterate_ood_classes():
        ood = ood_class(verbose=True)
        ood.fit(train)

        prev = np.mean(ood.predict_proba(test))
        for offset in [10, 100, 1000]:
            test += np.abs(np.random.normal(0,offset,test.shape[0] * test.shape[1]).reshape(test.shape))
            pred = np.mean(ood.predict_proba(test))
            assert prev <= pred
            prev = pred