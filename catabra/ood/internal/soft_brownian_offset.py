#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from catabra.ood.base import SamplewiseOODDetector
from catabra.ood.utils import make_standard_transformer


def soft_brownian_offset(X, d_min=0.5, d_off=0.1, n_samples=1, show_progress=False, softness=False,
                         random_state=None):
    """Generates OOD samples using SBO on the input X and returns n_samples number of samples constrained by
    other parameters. Based on [1].

    Parameters
    ----------
    X: ndarray
        In-distribution (ID) data to form OOD samples around. First dimension contains samples.
    d_min: float, default=0.5
        (Likely) Minimum distance to ID data.
    d_off: float, default=0.1
        Offset distance used in each iteration.
    n_samples: int, default=1
        Number of samples to return.
    show_progress: bool, default=False
        Whether to show a tqdm progress bar.
    softness: float, default=False
        Describes softness of minimum distance. Parameter between 0 (hard) and 1 (soft).
    random_state: int
        RNG state used for reproducibility.

    Returns
    -------
    ndarray
        Out of distribution samples of shape (n_samples, X.shape[1])

    References
    ----------
    .. [1] https://github.com/flxai/soft-brownian-offset/blob/master/sbo/sbo.py

    """
    if softness == 0:
        softness = False
    if random_state is not None:
        np.random.seed(random_state)

    n_dim = X.shape[1]
    ys = []
    iterator = range(n_samples)
    if show_progress:
        iterator = tqdm(iterator)

    for i in iterator:
        # Sample uniformly from X
        y = X[np.random.choice(len(X))].astype(float)
        # Move out of reach of other points
        skip = False
        while True:
            dist = pairwise_distances(y[:, None].T, X)[0]
            if dist.min() > 0:
                if not softness and dist.min() > d_min:
                    skip = True
                elif softness > 0:
                    p = 1 / (1 + np.exp((-dist.min() + d_min) / softness / d_min * 7))
                    if np.random.uniform() < p:
                        skip = True
                elif not isinstance(softness, bool):
                    raise ValueError("Softness should be float greater zero")
            if skip:
                break
            y += gaussian_hyperspheric_offset(1, n_dim=n_dim)[0] * d_off
        ys.append(np.array(y))
    return np.array(ys)


def gaussian_hyperspheric_offset(n_samples, mu=4, std=.7, n_dim=3, random_state=None):
    """
    Generates OOD samples using GHO and returns n_samples number of samples constrained by other
    parameters. Inspired by [1].

    Parameters
    ----------
    n_samples: int
        Number of samples to return.
    mu: float, default=4
        Mean of distribution.
    std: float, default=0.7
        Standard deviation of distribution.
    n_dim: int, default=3
        Number of dimensions.
    random_state: int, optional
        RNG state used for reproducibility.

    Returns
    -------
    ndarray
        Out of distribution samples of shape (n_samples, n_dim)

    References
    ----------
    .. [1] https://stackoverflow.com/a/33977530/10484131
    """
    if random_state is not None:
        np.random.seed(random_state)
    vec = np.random.randn(n_dim, n_samples)
    vec /= np.linalg.norm(vec, axis=0)
    vec *= np.random.normal(loc=mu, scale=std, size=n_samples)
    vec = vec.T
    return vec


class SoftBrownianOffset(SamplewiseOODDetector):
    """
    Out-of-Distribution detector using soft brownian offset.
    Transforms samples into a lower dimensional space and generates synthetic OOD samples in this subspace.
    A classifier is trained to detect the OOD samples.

    Parameters
    ----------
    classifier: default=RandomForestClassifier
        Classifier for training to differentiate in- (ID) and out-of-distribution (OOD) samples.
    dim_reduction: default=PCA
        Dimensionality reduction algorithm to use.
    dist_min: float, default=0.2
        (Likely) Minimum distance to ID data
    dist_off: float, default=0.01
        Offset distance used in each iteration
    softness: float, default=0
        Describes softness of minimum distance. Parameter between 0 (hard) and 1 (soft)
    samples: float, default=1
        Number of samples to return in proportion to original samples
    """

    def __init__(
            self,
            subset: float = 1,
            classifier=RandomForestClassifier,
            dim_reduction=PCA,
            dist_min: float = 0.2,
            dist_off: float = 0.01,
            softness: float = 0,
            samples: float = 1,
            random_state: int = None,
            verbose: bool = True,
            **kwargs
    ):
        super().__init__(subset, random_state=random_state, verbose=verbose)

        dimred_kwargs = kwargs.get('dim_reduction', {})
        classifier_kwargs = kwargs.get('classifier', {})

        self._dim_reduction = dim_reduction(**dimred_kwargs)
        self._classifier = classifier(**classifier_kwargs)
        self._transformer = make_standard_transformer()

        self._dist_min = dist_min
        self._dist_off = dist_off
        self._samples = samples
        self._softness = softness

    def _transform(self, X: pd.DataFrame):
        return self._transformer.transform(X).values

    def _fit_transformer(self, X: pd.DataFrame):
        self._transformer.fit(X)

    def _fit_transformed(self, X: pd.DataFrame, y: pd.Series):
        n_samples = int(X.shape[0] * self._samples)
        syn_samples = soft_brownian_offset(X, d_min=self._dist_min, d_off=self._dist_off,
                                               n_samples=n_samples, random_state=self._random_state,
                                               softness=self._softness)
        self._y = [0] * X.shape[0] + [1] * n_samples
        self._classifier.fit(X=np.vstack((X, syn_samples)), y=self._y)

    def _predict_transformed(self, X):
        return self._classifier.predict(X)

    def _predict_proba_transformed(self, X):
        return self._classifier.predict_proba(X)[:,1]