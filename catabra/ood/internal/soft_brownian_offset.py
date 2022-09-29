import numpy as np
import pandas as pd
import sbo
from sklearn.ensemble import RandomForestClassifier

from catabra.ood.base import OODDetector
from catabra.ood.internal.autoencoder import Autoencoder
from catabra.ood.utils import make_standard_transformer


class SoftBrownianOffset(OODDetector):
    """
    Out-of-Distribution detector using soft brownian offset.
    Transforms samples into a lower dimensional space and generates synthetic OOD samples in this subspace.
    A classifier is trained to detect the OOD samples.
    Requires sbo to be installed.
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
        :param classifier: classifier for training to differentiate in- (ID) and out-of-distribution (OOD) samples
        :param dist_min: (Likely) Minimum distance to ID data
        :param dist_off: Offset distance used in each iteration
        :param softness: Describes softness of minimum distance. Parameter between 0 (hard) and 1 (soft)
        :param samples: Number of samples to return in proportion to original samples
        """

        super().__init__(subset, verbose)

        autoenc_kwargs = kwargs.get('autoencoder', {})
        classifier_kwargs = kwargs.get('classifier', {})

        self._autoencoder = Autoencoder(**autoenc_kwargs, verbose=False)
        self._classifier = classifier(**classifier_kwargs)
        self._transformer = make_standard_transformer()

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
        n_samples = int(X.shape[0] * self._samples)
        syn_samples = sbo.soft_brownian_offset(X, d_min=self._dist_min, d_off=self._dist_off,
                                               n_samples=n_samples, random_state=self._random_state,
                                               softness=self._softness)
        self._y = [0] * X.shape[0] + [1] * n_samples
        self._classifier.fit(X=np.vstack((X, syn_samples)), y=self._y)

    def _predict_transformed(self, X):
        return self._classifier.predict(X)

    def _predict_proba_transformed(self, X):
        return self._classifier.predict_proba(X)