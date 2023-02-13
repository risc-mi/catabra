import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from catabra.ood.base import OODDetector
from catabra.ood.internal.autoencoder import Autoencoder
from catabra.ood.utils import make_standard_transformer

import numpy as np

from sklearn.metrics import pairwise_distances
from tqdm import tqdm

# Based on. https://github.com/flxai/soft-brownian-offset/blob/master/sbo/sbo.py


def soft_brownian_offset(X, d_min=0.5, d_off=0.1, n_samples=1, show_progress=False, softness=False,
                         random_state=None):
    """Generates OOD samples using SBO on the input X and returns n_samples number of samples constrained by
    other parameters.
    Args:
        X (:obj:`numpy.array`): In-distribution (ID) data to form OOD samples around. First dimension contains samples
        d_min (float): (Likely) Minimum distance to ID data
        d_off (float): Offset distance used in each iteration
        n_samples(int): Number of samples to return
        show_progress(boolean): Whether to show a tqdm progress bar
        softness(float): Describes softness of minimum distance. Parameter between 0 (hard) and 1 (soft)
        random_state(int): RNG state used for reproducibility
    Returns:
        :obj:`numpy.array`:
            Out of distribution samples of shape (n_samples, X.shape[1])
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


# Inspired by https://stackoverflow.com/a/33977530/10484131
def gaussian_hyperspheric_offset(n_samples, mu=4, std=.7, n_dim=3, random_state=None):
    """Generates OOD samples using GHO and returns n_samples number of samples constrained by other
    parameters.
    Args:
        n_samples(int): Number of samples to return
        mu (float): Mean of distribution
        std (float): Standard deviation of distribution
        n_dim (int): Number of dimensions
        random_state(int): RNG state used for reproducibility
    Returns:
        :obj:`numpy.array`:
            Out of distribution samples of shape (n_samples, n_dim)
    """
    if random_state is not None:
        np.random.seed(random_state)
    vec = np.random.randn(n_dim, n_samples)
    vec /= np.linalg.norm(vec, axis=0)
    vec *= np.random.normal(loc=mu, scale=std, size=n_samples)
    vec = vec.T
    return vec


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
            dim_reduction=PCA,
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

        dimred_kwargs = kwargs.get('dim_reduction', {})
        classifier_kwargs = kwargs.get('classifier', {})

        self._dim_reduction = dim_reduction(**dimred_kwargs)
        self._classifier = classifier(**classifier_kwargs)
        self._transformer = make_standard_transformer()

        self._dist_min = dist_min
        self._dist_off = dist_off
        self._samples = samples
        self._softness = softness
        self._random_state = random_state

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
        return self._classifier.predict_proba(X)


def test():
    test = np.array([
        [0, 1, 2],
        [0.1, 1.1, 2.1],
        [0.2, 1.2, 2.2],
        [0.3, 1.3, 2.3],
        [0.4, 1.4, 2.4],
        [0.5, 1.5, 2.5],
        [0.6, 1.6, 2.6],
        [0.7, 1.7, 2.7],
        [0.8, 1.8, 2.8],
        [0.9, 1.9, 2.9],
    ])

    ood = soft_brownian_offset(test, n_samples=10)
    import plotly.express as px
    df = pd.DataFrame(np.vstack([test, ood]), columns=['x', 'y', 'z'])
    df['id'] = [True] * test.shape[0] + [False] * ood.shape[0]
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='id')
    fig.show()