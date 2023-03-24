#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def create_synthetic_data(task: str = 'binary_classification', n_samples: int = 50000, n_groups: int = 5,
                          object_column: bool = True, difficulty: int = 0, frac_unlabeled: float = 0.2,
                          seed: Optional[int] = None) -> pd.DataFrame:
    """
    Create a synthetic data with several feature- and label columns with different data types.
    :param task: Prediction task. Can also be "multioutput_regression", in which case two regression targets are
    constructed.
    :param n_samples: Number of samples = number of rows of resulting DataFrame.
    :param n_groups: Number of groups. Must be between 2 and `n_samples`.
    :param object_column: Whether to add a column with object data type, containing strings.
    :param difficulty: Difficulty of the prediction task:
    * 0: very easy, prediction models should be near perfect
    * 1: moderately difficult, performance of prediction models should be good
    * 2: moderately difficult unless samples are weighted by column "_sample_weight", in which case task is easy
    * 3: random labels, models should not be significantly better than random
    :param frac_unlabeled: Fraction of unlabeled samples.
    :param seed: Random seed.
    :return: DataFrame with features, label(s), and additional columns "_group", "_sample_weight" and "_test".
    """

    assert 2 <= n_groups <= n_samples

    rng = np.random.RandomState(seed=seed)

    # create random groups
    grp = np.zeros((n_samples,), dtype=np.int32)
    s = round(n_samples / n_groups)
    i = 0
    for g in range(n_groups - 1):
        j = min(max(1, rng.randint(s - 5, s + 6)), n_samples - i)
        grp[i:i + j] = g
        i += j
    grp[i:] = n_groups - 1
    grp = rng.permutation(grp)

    # construct random features
    syn_df = pd.DataFrame(
        data=dict(
            uniform=rng.uniform(-2, 10, size=n_samples),
            normal=rng.normal(5, 2.5, size=n_samples),
            exponential=rng.exponential(3, size=n_samples),
            multi_modal=_multi_modal(rng, rng.normal(0, 0.5, size=n_samples), rng.normal(7, 4, size=n_samples),
                                     rng.normal(-3, 1, size=n_samples)),
            datetime=pd.Timestamp(0) + pd.Series(
                np.linspace(0, 100, n_samples) - 2 * np.sin(np.linspace(0, 100, n_samples))
            ) * pd.Timedelta(12, unit='h'),
            timedelta=(
                rng.chisquare(3, size=n_samples) * rng.laplace(0.5, 2, size=n_samples)
            ) * pd.Timedelta(1, unit='m'),
            bool=rng.choice([False, True], p=[0.83, 0.17], size=n_samples),
            int=rng.zipf(1.5, size=n_samples),
            cat_2=pd.Categorical.from_codes(rng.choice(2, p=[0.61, 0.39], size=n_samples), categories=['X', 'Y']),
            cat_5=pd.Categorical.from_codes(rng.choice(5, p=[0.18, 0.21, 0.2, 0.3, 0.11], size=n_samples),
                                            categories=['A', 'B', 'C', 'D', 'E']),
            float=grp + rng.normal(0, 0.1, size=n_samples),
            _group=grp,
            _sample_weight=_multi_modal(rng, rng.exponential(size=n_samples) * 0.05,
                                        rng.normal(loc=1, scale=0.2, size=n_samples))
        )
    )
    if object_column:
        syn_df['object'] = ['A' + str(i) for i in rng.randint(1000000, 9999999, size=n_samples)]

    # assign random train-tests split
    split = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=rng)
    train, _ = list(split.split(syn_df.index, groups=syn_df['_group']))[0]
    syn_df['_test'] = True
    syn_df.loc[train, '_test'] = False

    if difficulty <= 0:
        label_float = np.pi * syn_df['uniform']
        label_float += rng.normal(-2, 0.1, size=n_samples)
    elif difficulty == 1:
        # can be predicted from features
        label_float = np.sin(syn_df['multi_modal']) * np.tanh(syn_df['uniform']) + \
                      np.log(np.maximum(1e-7, syn_df['normal']))
        label_float.loc[syn_df['bool']] += np.arcsinh(syn_df['exponential'] - syn_df['uniform'])
        label_float.loc[syn_df['cat_2'] == 'Y'] -= np.minimum(syn_df['int'], 10)
        label_float += rng.normal(0, 1, size=n_samples)
    elif difficulty == 2:
        # can be predicted easily from features for samples with large sample weight
        label_float = (syn_df['datetime'] - pd.Timestamp(0)) / (pd.Timedelta(1, unit='d'))
        mask = syn_df['_sample_weight'] < syn_df['_sample_weight'].median()
        label_float.loc[mask] = rng.uniform(label_float.min(), label_float.max(), size=mask.sum())
    else:
        # random
        label_float = pd.Series(index=syn_df.index, data=rng.uniform(-10, 2, size=n_samples))

    if task == 'regression':
        syn_df['_label'] = label_float

    elif task == 'multioutput_regression':
        syn_df['_label_float'] = label_float

        # can be predicted from features
        syn_df['_label_timedelta'] = (np.exp(syn_df['uniform']) * np.tanh(syn_df['multi_modal']) +
                                      np.log(syn_df['exponential'])) * pd.Timedelta(1, unit='h')
        syn_df.loc[syn_df['bool'], '_label_timedelta'] += np.sin(syn_df['normal'] * np.pi) * pd.Timedelta(30, unit='d')
        syn_df.loc[syn_df['cat_2'] == 'X', '_label_timedelta'] -= \
            pd.to_timedelta(np.minimum(syn_df['int'] * 50, 500), unit='d')
        syn_df['_label_timedelta'] += rng.normal(0, 1, size=n_samples) * pd.Timedelta(10, unit='d')

    else:
        label_binary = pd.Categorical.from_codes(
            (label_float < label_float.quantile(0.23)).astype(np.int8),
            categories=['-', '+']
        )

        if task == 'binary_classification':
            syn_df['_label'] = label_binary

        elif task == 'multiclass_classification':
            syn_df['_label'] = -10.
            syn_df.loc[(label_binary == '-') & syn_df['cat_5'].isin(['A', 'B']), '_label'] = -9.
            syn_df.loc[(label_binary == '-') & syn_df['cat_5'].isin(['C', 'D', 'E']), '_label'] = -8.
            syn_df.loc[(label_binary == '+') & (syn_df['timedelta'] < pd.Timedelta(0)), '_label'] = -7.

        elif task == 'multilabel_classification':
            syn_df['_label_random'] = 0.
            syn_df.loc[rng.choice([False, True], p=[0.73, 0.27], size=n_samples), '_label_random'] = 1.

            syn_df['_label_binary'] = label_binary

            # can be predicted easily from features, ROC-AUC on tests set is approx. 1
            syn_df['_label_bool'] = syn_df['bool'].astype(np.float32)

        else:
            raise ValueError(f'Unknown task: {task}')

    # set random features to NaN
    for c in syn_df.columns:
        if not c.startswith('_label') and (syn_df[c].dtype.kind in 'fmM' or syn_df[c].dtype.name == 'category'):
            p = rng.uniform(0.1, 0.3)
            mask = rng.choice([False, True], p=[1 - p, p], size=n_samples)
            syn_df.loc[mask, c] = None

    # set random labels to NaN
    if frac_unlabeled > 0:
        syn_df.loc[
            ~syn_df['_test'] & rng.choice([False, True], p=[1 - frac_unlabeled, frac_unlabeled], size=n_samples),
            [c for c in syn_df.columns if c.startswith('_label')]
        ] = None

    return syn_df


def _multi_modal(rng, *args: np.ndarray) -> np.ndarray:
    n_samples = len(args[0])
    return np.stack(args)[rng.randint(len(args), size=n_samples), np.arange(n_samples)]
