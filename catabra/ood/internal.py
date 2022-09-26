# More advanced version in PyOD - but it needs Tensorflow/Torch
from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor

from catabra.ood.base import OODDetector
from catabra.ood.utils import StandardTransformer
from catabra.util.table import train_test_split


class Autoencoder(OODDetector):
    """
    Autoencoder for out-of distribution detection.
    Uses a neural network to encode data into a lower dimensional space and reconstruct the original data from it.
    Reconstruction error determines the likelihood of a sample being out-of-distribution.
    """

    _mlp_kwargs = {
        'n_iter_no_change': 50,
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.01,
        'max_iter': 300
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

    def __init__(self, subset=1, target_dim_factor=0.25, reduction_factor=0.9, p_val=0.05,
                 random_state: int=None, verbose=True, **mlp_kwargs):
        """
        Intialization of Autoencoder
        @param target_dim_factor: how
        @param reduction_factor: how much each layer reduces the dimensionality
        @param p_val: p-value to decide when a sample is out of distribution
        """
        super().__init__(subset=subset, verbose=verbose)
        self._target_dim_factor = target_dim_factor
        self._reduction_factor = reduction_factor
        self._p_val = p_val
        self._random_state = np.random.randint(1000) if random_state is None else random_state

        self._regressor = None
        self._transformer = StandardTransformer()
        if mlp_kwargs and len(mlp_kwargs) > 0:
            self._mlp_kwargs = mlp_kwargs

    def _fit_transformer(self, X: pd.DataFrame):
        self._transformer.fit(X)

    def _transform(self, X: pd.DataFrame):
        return self._transformer.transform(X)

    def _fit_transformed(self, X: pd.DataFrame, y: pd.Series):
        """
        @param: X: samples to fit autoencoder on
        @param y: ignored. Only for interface consistency
        """
        num_layers = np.log(self._target_dim_factor) // np.log(self._reduction_factor)
        self._encoder_layers = np.round(np.power(np.repeat(self._reduction_factor, num_layers),
                                        np.arange(1, num_layers + 1)) * X.shape[1], 2).astype(np.int32)
        self._decoder_layers = np.flip(self._encoder_layers)[1:]
        layers = list(np.append(self._encoder_layers, self._decoder_layers))

        self._regressor = self.SeparableMLP(hidden_layer_sizes=layers, random_state=self._random_state,
                                            verbose=self._verbose, **self._mlp_kwargs)
        self._regressor.fit(X, X)
        _, X_val, _, _ = train_test_split(X, X, random_state=self._random_state,
                                          test_size=self._regressor.validation_fraction)
        # TODO: validate that X_val is same X_val as during training
        self._errors = X_val - self._regressor.predict(X_val)

    def _predict_pvals(self, X):
        X_reconstructed = self._regressor.predict(X)
        errors = X - X_reconstructed

        p_val = np.zeros_like(errors)
        for r in range(errors.shape[0]):
            for c, name in enumerate(errors.columns):
                p_val[r,c] = np.sum(self._errors.iloc[:,c] >= errors.iloc[r,c]) / self._errors.shape[0]

        return p_val

    def _predict_transformed(self, X):
        p_vals = self._predict_pvals(X)
        return (p_vals <= self._p_val).astype(int)

    def _predict_proba_transformed(self, X):
        p_vals = self._predict_pvals(X)
        return 1 - p_vals

    def predict_raw(self, X):
        return self._regressor.predict(X)


class SoftBrownianOffset(OODDetector):
    """
    Out-of-Distribution detector using soft brownian offset.
    Transforms samples into a lower dimensional space and generates synthetic OOD samples in this subspace.
    A classifier is trained to detect the OOD samples.
    """

    def __init__(
            self,
            subset: float = 1,
            classifier=RandomForestClassifier,
            dist_min: float = 0.5,
            dist_off: float = 0.1,
            softness: float = 0,
            samples: float = 1,
            random_state: int = None,
            verbose: bool = True,
            **kwargs
    ):

        """
        Initialize SoftBrownianOffset
        @param classifier: classifier for training to differentiate in- (ID) and out-of-distribution (OOD) samples
        @param dist_min: (Likely) Minimum distance to ID data
        @param dist_off: Offset distance used in each iteration
        @param softness: Describes softness of minimum distance. Parameter between 0 (hard) and 1 (soft)
        @param samples: Number of samples to return in proportion to original samples
        """

        super().__init__(subset, verbose)
        import sbo

        autoenc_kwargs = kwargs.get('autoencoder', {})
        classifier_kwargs = kwargs.get('classifier', {})

        self._autoencoder = Autoencoder(**autoenc_kwargs)
        self._classifier = classifier(**classifier_kwargs)
        self._transformer = StandardTransformer()

        self._dist_min = dist_min
        self._dist_off = dist_off
        self._samples = samples
        self._softness = softness
        self._random_state = random_state

    def _transform(self, X: pd.DataFrame):
        return self._autoencoder.encoder.predict(self._autoencoder._transform(X))

    def _fit_transformer(self, X: pd.DataFrame):
        self._autoencoder.fit(X)

    def _fit_transformed(self, X: pd.DataFrame, y: pd.Series):
        n_samples = X.shape[0] * self._samples
        syn_samples = sbo.soft_brownian_offset(X, d_min=self._dist_min, d_off=self._dist_off,
                                               n_samples=n_samples, random_state=self._random_state,
                                               softness=self._softness)
        self._y = [0] * X.shape[0] + [1] * n_samples
        self._classifier.fit(X=np.vstack((X, syn_samples)), y=self._y)

    def _predict_transformed(self, X):
        return self._classifier.predict(X)

    def _predict_proba_transformed(self, X):
        return self._classifier.predict_proba(X)