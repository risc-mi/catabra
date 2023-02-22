#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from sklearn.utils.validation import column_or_1d
from sklearn.utils.multiclass import type_of_target
from sklearn import model_selection
from sklearn.model_selection._split import _BaseKFold, _validate_shuffle_split


def _find_best_fold(n_splits: int, y_counts_per_fold, y_cnt, group_y_counts) -> int:
    # copied from sklearn.model_selection._split.StratifiedGroupKFold
    best_fold = None
    min_eval = np.inf
    min_samples_in_fold = np.inf
    for i in range(n_splits):
        y_counts_per_fold[i] += group_y_counts
        # Summarise the distribution over classes in each proposed fold
        std_per_class = np.std(y_counts_per_fold / y_cnt.reshape(1, -1), axis=0)
        y_counts_per_fold[i] -= group_y_counts
        fold_eval = np.mean(std_per_class)
        samples_in_fold = np.sum(y_counts_per_fold[i])
        if fold_eval < min_eval or (np.isclose(fold_eval, min_eval) and samples_in_fold < min_samples_in_fold):
            min_eval = fold_eval
            min_samples_in_fold = samples_in_fold
            best_fold = i
    return best_fold


def _stratified_group_shuffle_split(y, groups, n_test: int, rng,
                                    n_splits: int = 1, method: str = 'automatic', n_iter: int = None):
    n_samples = len(y)
    if n_test >= n_samples:
        for _ in range(n_splits):
            yield np.ones(n_samples, dtype=np.bool)
        return
    elif n_test <= 0:
        for _ in range(n_splits):
            yield np.zeros(n_samples, dtype=np.bool)
        return

    _, y_inv, y_cnt = np.unique(y, return_inverse=True, return_counts=True)
    n_classes = len(y_cnt)

    _, groups_inv, groups_cnt = np.unique(groups, return_inverse=True, return_counts=True)
    y_counts_per_group = np.zeros((len(groups_cnt), n_classes))
    for class_idx, group_idx in zip(y_inv, groups_inv):
        y_counts_per_group[group_idx, class_idx] += 1

    if method == 'automatic':
        method = 'brute_force' if len(groups_cnt) > 1000 else 'exact'

    # In the future, one could consider adding a method based on gradient descent, as described here:
    # https://github.com/joaofig/strat-group-split

    if method == 'brute_force':
        if n_iter is None:
            n_iter = min(max(5, 5 * 10 ** 6 // len(groups_cnt)), 100)
        else:
            n_iter = max(1, n_iter)

        for _ in range(n_splits):
            best_loss = np.inf
            test = None

            for _ in range(n_iter):
                p = rng.permutation(len(groups_cnt))
                cs = groups_cnt[p].cumsum()
                i = np.argmin(np.abs(cs - n_test)) + 1
                # `i` is index of first group to include in train set, wrt. permutation `p`

                # l-inf norm of class-wise relative error between overall distribution and test-set distribution
                y_test_freq = y_counts_per_group[p[:i]].sum(axis=0) / cs[i - 1]
                loss = np.max(np.abs(n_samples * y_test_freq - y_cnt) / y_cnt)

                if loss < best_loss:
                    best_loss = loss
                    test = p[:i]  # indices of groups in test set

            yield np.in1d(groups_inv, test)
    else:
        for _ in range(n_splits):
            rng.shuffle(y_counts_per_group)

            # Stable sort to keep shuffled order for groups with the same class distribution variance
            sorted_groups_idx = np.argsort(-np.std(y_counts_per_group, axis=1), kind='mergesort')

            y_counts_per_fold = np.zeros((2, n_classes))
            group_fold_mapping = -np.ones(len(groups_cnt), dtype=np.int8)

            best_test_size = -n_samples
            test_fold = 0

            # assign groups to folds until one of them roughly contains `n_test` samples => test set
            for group_idx in sorted_groups_idx:
                group_y_counts = y_counts_per_group[group_idx]
                # find best fold for current group
                best_fold = _find_best_fold(
                    2,
                    y_counts_per_fold=y_counts_per_fold,
                    y_cnt=y_cnt,
                    group_y_counts=group_y_counts,
                )
                y_counts_per_fold[best_fold] += group_y_counts
                group_fold_mapping[group_idx] = best_fold
                n = y_counts_per_fold[best_fold].sum()
                if abs(n - n_test) < abs(best_test_size - n_test):
                    best_test_size = n
                    test_fold = best_fold
                if n > n_test and y_counts_per_fold[1 - best_fold].sum() > n_test:
                    break

            yield group_fold_mapping[groups_inv] == test_fold


class StratifiedGroupShuffleSplit(model_selection.StratifiedShuffleSplit):
    """
    Stratified grouped split into train- and test set. Ensures that groups in the two sets do not overlap, and tries
    to distribute samples in such a way that class percentages are roughly maintained in each split.

    Parameters
    ----------
    n_splits : int, default=10
        Number of re-shuffling & splitting iterations.

    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.1.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int or RandomState instance, default=None
        Controls the randomness of the training and testing indices produced.
        Pass an int for reproducible output across multiple function calls.

    method : str, default="automatic"
        Resampling method to use. Can be "automatic", "exact" and "brute_force".
        If there are many small groups, "brute_force" tends to give reasonable
        results and is significantly faster than "exact". Otherwise, if there
        are only few large groups, method "exact" might be preferable.
        "automatic" tries to infer the optimal method based on the number of
        groups.

    n_iter : int, default=None
        Number of brute-force iterations. The larger the number, the more
        splits are tried, and hence the better the results get. If None, the
        number of iterations is determined automatically.
    """

    def __init__(self, n_splits=10, *, test_size=None, train_size=None, random_state=None, method='automatic',
                 n_iter=None):
        super().__init__(n_splits=n_splits, test_size=test_size, train_size=train_size, random_state=random_state)
        self.n_iter = n_iter
        self.method = method

    def _iter_indices(self, X, y, groups=None):
        y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ('binary', 'multiclass')
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                'Supported target types are: {}. Got {!r} instead.'.format(allowed_target_types, type_of_target_y)
            )
        y = column_or_1d(y)
        n_samples = len(y)

        n_train, n_test = _validate_shuffle_split(n_samples, self.test_size, self.train_size,
                                                  default_test_size=self._default_test_size)
        rng = check_random_state(self.random_state)
        for test_mask in _stratified_group_shuffle_split(y, groups, n_test, rng, n_splits=self.n_splits,
                                                         method=self.method, n_iter=self.n_iter):
            yield np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


class StratifiedGroupKFold(_BaseKFold):
    """
    Copied and adapted from sklearn version 1.0.2, because auto-sklearn
    requires an older version without this class.

    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

    Changelist:
    - Removed warning if some class has fewer than `n_splits` instances.
    - Do not throw error if all classes have fewer than `n_splits` instances.
    - Added method "brute_force".

    Parameters
    ----------
    n_splits : int, default=10
        Number of re-shuffling & splitting iterations.

    shuffle : bool, default=False
        Whether to shuffle samples before splitting.

    random_state : int or RandomState instance, default=None
        Controls the randomness of the training and testing indices produced.
        Pass an int for reproducible output across multiple function calls.

    method : str, default="automatic"
        Resampling method to use. Can be "automatic", "exact" and "brute_force".
        If there are many small groups, "brute_force" tends to give reasonable
        results and is significantly faster than "exact". Otherwise, if there
        are only few large groups, method "exact" might be preferable.
        "automatic" tries to infer the optimal method based on the number of
        groups.
        Note that "brute_force" is only possible if `shuffle` is set to True.

    n_iter : int, default=None
        Number of brute-force iterations. The larger the number, the more
        splits are tried, and hence the better the results get. If None, the
        number of iterations is determined automatically.
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None, method: str = 'automatic', n_iter: int = None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.n_iter = n_iter
        self.method = method

    def _iter_test_indices(self, X=None, y=None, groups=None):
        rng = check_random_state(self.random_state)
        y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ('binary', 'multiclass')
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                'Supported target types are: {}. Got {!r} instead.'.format(allowed_target_types, type_of_target_y)
            )

        y = column_or_1d(y)

        if self.method == 'automatic':
            self.method = 'brute_force' if self.shuffle and self.n_splits <= 10 \
                                           and len(np.unique(groups)) > 1000 else 'exact'

        if self.shuffle and self.method == 'brute_force':
            n_test = int(len(y) / self.n_splits + 0.5)
            train_mask = np.ones(len(y), dtype=np.bool)
            for _ in range(self.n_splits):
                for local_test_mask in _stratified_group_shuffle_split(y[train_mask], groups[train_mask], n_test, rng,
                                                                       method='brute_force', n_iter=self.n_iter):
                    test_mask = np.zeros(len(y), dtype=np.bool)
                    test_mask[train_mask] = local_test_mask
                    train_mask &= ~test_mask
                    yield np.flatnonzero(test_mask)

        else:
            _, y_inv, y_cnt = np.unique(y, return_inverse=True, return_counts=True)
            n_classes = len(y_cnt)

            _, groups_inv, groups_cnt = np.unique(groups, return_inverse=True, return_counts=True)
            y_counts_per_group = np.zeros((len(groups_cnt), n_classes))
            for class_idx, group_idx in zip(y_inv, groups_inv):
                y_counts_per_group[group_idx, class_idx] += 1

            y_counts_per_fold = np.zeros((self.n_splits, n_classes))
            group_fold_mapping = -np.ones(len(groups_cnt), dtype=np.int8)

            if self.shuffle:
                rng.shuffle(y_counts_per_group)

            # Stable sort to keep shuffled order for groups with the same class distribution variance
            sorted_groups_idx = np.argsort(-np.std(y_counts_per_group, axis=1), kind='mergesort')

            for group_idx in sorted_groups_idx:
                group_y_counts = y_counts_per_group[group_idx]
                # find best fold for current group
                best_fold = _find_best_fold(
                    self.n_splits,
                    y_counts_per_fold=y_counts_per_fold,
                    y_cnt=y_cnt,
                    group_y_counts=group_y_counts,
                )
                y_counts_per_fold[best_fold] += group_y_counts
                group_fold_mapping[group_idx] = best_fold

            for i in range(self.n_splits):
                yield np.flatnonzero(group_fold_mapping[groups_inv] == i)


class CustomPredefinedSplit(model_selection.BaseCrossValidator):
    """
    Predefined split cross-validator. Provides train/test indices to split data into train/test sets using a predefined
    scheme specified by explicit test indices.

    In contrast to `sklearn.model_selection.PredefinedSplit`, samples can be in the test set of more than one split.

    In methods `split()` etc., parameters `X`, `y` and `groups` only exist for compatibility, but are always ignored.

    Parameters
    ----------
    test_folds : list of array-like
        Indices of test samples for each split. The number of splits equals the length of the list.
        Note that the test sets do not have to be mutually disjoint.
    """

    def __init__(self, test_folds=None):
        assert test_folds is not None
        if isinstance(test_folds, (np.ndarray, pd.Series)):
            test_folds = [test_folds]
        self.test_folds = test_folds

    @classmethod
    def from_data(cls, X: pd.DataFrame, columns) -> 'CustomPredefinedSplit':
        if isinstance(columns, (str, int)):
            columns = [columns]
        assert all(X[c].dtype.kind == 'b' for c in columns)
        return cls(test_folds=[np.flatnonzero(X[c].values) for c in columns])

    def _iter_test_indices(self, X=None, y=None, groups=None):
        for indices in self.test_folds:
            yield indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.test_folds)
