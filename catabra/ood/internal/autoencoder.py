#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from catabra.ood.base import SamplewiseOODDetector
from catabra.ood.utils import make_standard_transformer


class Autoencoder(SamplewiseOODDetector):
    """
    Autoencoder for out-of distribution detection.
    Uses a neural network to encode data into a lower dimensional space and reconstruct the original data from it.
    Reconstruction error determines the likelihood of a sample being out-of-distribution.
    """

    _mlp_kwargs = {
        'n_iter_no_change': 50,
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.01,
        'max_iter': 300,
        'validation_fraction': 0.4
    }

    class SeparableMLP(MLPRegressor):

        @property
        def layer_units(self):
            if not hasattr(self, '_layer_units'):
                self._layer_units = []
            return self._layer_units

        @layer_units.setter
        def layer_units(self, units: List[int]):
            self._layer_units = units

        def _fit(self, X, y, incremental=False):
            super()._fit(X, y, incremental)
            n_features = X.shape[1]
            self.layer_units = [n_features] + self.hidden_layer_sizes + [self.n_outputs_]

        def extract(self, start=0, end=None):
            extracted = deepcopy(self)

            if end is None:
                end = len(extracted.layer_units)
            extracted.layer_units = extracted.layer_units[start:end]
            extracted.coefs_ = extracted.coefs_[start:end]
            extracted.intercepts_ = extracted.intercepts_[start:end]
            extracted.n_outputs_ = extracted.layer_units[-1]
            extracted.n_features_in_ = extracted.layer_units[0]
            extracted.n_layers_ = len(extracted.layer_units)

            # Compute the number of layers
            self.n_layers_ = len(self._layer_units)

            return extracted

    @property
    def encoder(self):
        num_layers = len(self._encoder_layers)
        return self._regressor.extract(0,num_layers + 1)

    @property
    def decoder(self):
        num_layers = len(self._encoder_layers)
        return self._regressor.extract(num_layers)

    def __init__(self, subset=1, target_dim_factor=0.25, reduction_factor=0.9, thresh=0.5,
                 random_state: int=None, verbose=True, mlp_kwargs=None):
        """
        Initialization of Autoencoder
        :param: target_dim_factor: how
        :param: reduction_factor: how much each layer reduces the dimensionality
        :param: p_val: p-value to decide when a sample is out of distribution
        """
        super().__init__(subset=subset, random_state=random_state, verbose=verbose)
        self._target_dim_factor = target_dim_factor
        self._reduction_factor = reduction_factor
        self._thresh = thresh

        self._regressor = None
        self._transformer = make_standard_transformer()
        if mlp_kwargs and len(mlp_kwargs) > 0:
            self._mlp_kwargs = mlp_kwargs

    @property
    def mlp_kwargs(self):
        return self._mlp_kwargs

    @property
    def thresh(self):
        return self._thresh

    @property
    def random_state(self):
        return self._random_state

    @property
    def reduction_factor(self):
        return self._reduction_factor

    @property
    def target_dim_factor(self):
        return self._target_dim_factor

    def _fit_transformer(self, X: pd.DataFrame):
        self._transformer.fit(X)

    def _transform(self, X: pd.DataFrame):
        return self._transformer.transform(X)

    def _fit_transformed(self, X: pd.DataFrame, y: pd.Series):
        """
        :param: X: samples to fit autoencoder on
        :param y: ignored. Only for interface consistency
        """
        num_layers = np.log(self._target_dim_factor) // np.log(self._reduction_factor)
        self._encoder_layers = np.round(np.power(np.repeat(self._reduction_factor, num_layers),
                                        np.arange(1, num_layers + 1)) * X.shape[1], 2).astype(np.int32)
        self._decoder_layers = np.flip(self._encoder_layers)[1:]
        layers = list(np.append(self._encoder_layers, self._decoder_layers))

        self._regressor = self.SeparableMLP(hidden_layer_sizes=layers, random_state=self._random_state,
                                            verbose=self._verbose, **self._mlp_kwargs)
        self._regressor.fit(X, X)
        _, X_val = train_test_split(X, random_state=self._random_state,
                                    test_size=self._regressor.validation_fraction)
        # TODO: validate that X_val is same X_val as during training
        errors = np.abs(X_val - self._regressor.predict(X_val))
        self._errors = np.max(errors, axis=0)

    def _predict_transformed(self, X):
        return (self._predict_proba_transformed(X) >= self._thresh).astype(int)

    def _predict_proba_transformed(self, X):
        X_reconstructed = self._regressor.predict(X)
        errors = np.abs(X - X_reconstructed)
        probas = np.max(errors - self._errors, axis=1)
        probas[probas < 0] = 0
        probas[probas > 1] = 1
        return probas

    def predict_raw(self, X):
        return self._regressor.predict(X)


def test():
    auto = Autoencoder()
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    X = pd.DataFrame(X)
    auto.fit(X)
    print(np.mean(auto.predict_proba(X)))
    print(np.mean(auto.predict_proba(X + np.random.normal(0,0.0001,X.shape[0] * X.shape[1]).reshape(X.shape))))
    print(np.mean(auto.predict_proba(X + np.random.normal(0,0.001,X.shape[0] * X.shape[1]).reshape(X.shape))))
    print(np.mean(auto.predict_proba(X + np.random.normal(0,0.01,X.shape[0] * X.shape[1]).reshape(X.shape))))
    print(np.mean(auto.predict_proba(X + np.random.normal(0,0.1,X.shape[0] * X.shape[1]).reshape(X.shape))))
    print(np.mean(auto.predict_proba(X + np.random.normal(0,1,X.shape[0] * X.shape[1]).reshape(X.shape))))
    print(np.mean(auto.predict_proba(X + np.random.normal(0,100,X.shape[0] * X.shape[1]).reshape(X.shape))))

    train, test = train_test_split(X)
    auto.fit(train)
    print(np.mean(auto.predict_proba(test)))

