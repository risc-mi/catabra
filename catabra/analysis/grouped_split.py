from collections import defaultdict
import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.validation import column_or_1d
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import StratifiedShuffleSplit
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


class StratifiedGroupShuffleSplit(StratifiedShuffleSplit):
    """
    Stratified grouped split into train- and test set. Ensures that groups in the two sets do not overlap, and tries
    to distribute samples in such a way that class percentages are roughly maintained in each split.
    """

    def __init__(self, n_splits=10, *, test_size=None, train_size=None, random_state=None):
        super().__init__(n_splits=n_splits, test_size=test_size, train_size=train_size, random_state=random_state)

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
        if n_train <= 0:
            for _ in range(self.n_splits):
                yield [], list(range(n_samples))
            return
        elif n_test <= 0:
            for _ in range(self.n_splits):
                yield list(range(n_samples)), []
            return

        _, y_inv, y_cnt = np.unique(y, return_inverse=True, return_counts=True)
        n_classes = len(y_cnt)

        _, groups_inv, groups_cnt = np.unique(groups, return_inverse=True, return_counts=True)
        y_counts_per_group = np.zeros((len(groups_cnt), n_classes))
        for class_idx, group_idx in zip(y_inv, groups_inv):
            y_counts_per_group[group_idx, class_idx] += 1

        rng = check_random_state(self.random_state)

        for _ in range(self.n_splits):
            rng.shuffle(y_counts_per_group)

            # Stable sort to keep shuffled order for groups with the same class distribution variance
            sorted_groups_idx = np.argsort(-np.std(y_counts_per_group, axis=1), kind='mergesort')

            y_counts_per_fold = np.zeros((2, n_classes))
            groups_per_fold = defaultdict(set)

            best_test_size = -n_samples
            test = set()

            # assign groups to folds until one of them roughly contains `n_test` samples => this will be the test set
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
                groups_per_fold[best_fold].add(group_idx)
                n = y_counts_per_fold[best_fold].sum()
                if abs(n - n_test) < abs(best_test_size - n_test):
                    best_test_size = n
                    test = {idx for idx, group_idx in enumerate(groups_inv) if group_idx in groups_per_fold[best_fold]}
                elif n > n_test and y_counts_per_fold[1 - best_fold].sum() > n_test:
                    break

            train = np.array([idx for idx in range(n_samples) if idx not in test])
            yield train, np.array(list(test))


class StratifiedGroupKFold(_BaseKFold):
    """
    Copied and adapted from sklearn version 1.0.2, because autosklearn requires an older version without this class.

    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html#sklearn.model_selection.StratifiedGroupKFold

    Changelist:
    - Removed warning if some class has fewer than `n_splits` instances.
    - Do not throw error if all classes have fewer than `n_splits` instances.
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

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
        _, y_inv, y_cnt = np.unique(y, return_inverse=True, return_counts=True)
        n_classes = len(y_cnt)

        _, groups_inv, groups_cnt = np.unique(groups, return_inverse=True, return_counts=True)
        y_counts_per_group = np.zeros((len(groups_cnt), n_classes))
        for class_idx, group_idx in zip(y_inv, groups_inv):
            y_counts_per_group[group_idx, class_idx] += 1

        y_counts_per_fold = np.zeros((self.n_splits, n_classes))
        groups_per_fold = defaultdict(set)

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
            groups_per_fold[best_fold].add(group_idx)

        for i in range(self.n_splits):
            test_indices = [idx for idx, group_idx in enumerate(groups_inv) if group_idx in groups_per_fold[i]]
            yield test_indices
