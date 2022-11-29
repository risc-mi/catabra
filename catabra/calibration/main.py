from typing import Union, Optional, Type
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import _SigmoidCalibration     # noqa

from ..util import io, logging
from ..util import table as tu
from ..core import CaTabRaBase, Invocation, CaTabRaPaths


def calibrate(*table: Union[str, Path, pd.DataFrame], folder: Union[str, Path] = None,
              split: Optional[str] = None, subset=None, sample_weight: Optional[str] = None,
              method: Optional[str] = None, out: Union[str, Path, None] = None,
              from_invocation: Union[str, Path, dict, None] = None):
    """
    Calibrate existing CaTabRa classification models. Calibration ensures that the probability estimates returned by
    the model are an indicator for the "true" confidence of the model, such that for instance a probability of 0.5
    means that the model is unsure about its prediction. Citing scikit-learn:

        Well calibrated classifiers are probabilistic classifiers for which the output of the `predict_proba()`
        method can be directly interpreted as a confidence level. For instance, a well calibrated (binary)
        classifier should classify the samples such that among the samples to which it gave a `predict_proba`-value
        close to 0.8, approximately 80% actually belong to the positive class.

    :param table: The table(s) to calibrate the CaTabRa classifier on. If multiple are given, their columns are merged
    into a single table. Must have the same format as the table(s) initially passed to function `analyze()`.
    :param folder: The folder containing the CaTabRa classifier to calibrate.
    :param split: Optional, column used for splitting the data into disjoint subsets and calibrating the classifier on
    only one of them (given by parameter `subset`). Ignored if `subset` is None.
    :param subset: Optional, value in column `split` to consider for calibration. For instance, if the column specified
    by `split` contains values "train", "val" and "test", and `subset` is set to "val", the classifier is calibrated
    only on the "val"-entries.
    In general, note that classifiers should be neither calibrated on the data used for training nor on the data used
    for evaluating them.
    :param sample_weight: Optional, column with sample weights. If specified and not "", must have numeric data type.
    :param method: Calibration method. Must be one of "sigmoid", "isotonic" or "auto". "sigmoid" should be used for
    small sample sizes (<< 1000 samples) to avoid overfitting; otherwise, "isotonic" is preferable.
    "auto" automatically selects the calibration method based on the sample size.
    :param out: Optional, directory where to save all generated artifacts. Defaults to a directory located in `folder`,
    with a name following a fixed naming pattern. If `out` already exists, the user is prompted to specify whether it
    should be replaced; otherwise, it is automatically created.
    :param from_invocation: Optional, dict or path to an invocation.json file. All arguments of this function not
    explicitly specified are taken from this dict; this also includes the table to use for calibration.
    """

    calib = CaTabRaCalibration(invocation=from_invocation)
    calib(
        *table,
        folder=folder,
        split=split,
        subset=subset,
        sample_weight=sample_weight,
        method=method,
        out=out
    )


class CaTabRaCalibration(CaTabRaBase):

    @property
    def invocation_class(self) -> Type['CalibrationInvocation']:
        return CalibrationInvocation

    def _call(self):
        loader = io.CaTabRaLoader(self._invocation.folder, check_exists=True)
        self._config = loader.get_config()

        out_ok = self._invocation.resolve_output_dir(
            prompt=f'Calibration folder "{self._invocation.out.as_posix()}" already exists. Delete?'
        )
        if not out_ok:
            logging.log('### Aborting')
            return

        model = loader.get_model()
        fe = loader.get_fitted_ensemble(from_model=False)
        if model is None and fe is None:
            logging.log('### Aborting: no trained prediction model found')
            return

        with logging.LogMirror((self._invocation.out / CaTabRaPaths.ConsoleLogs).as_posix()):
            logging.log(f'### Calibration started at {self._invocation.start}')
            io.dump(io.to_json(self._invocation), self._invocation.out / CaTabRaPaths.Invocation)

            encoder = loader.get_encoder()
            if encoder.task_ == 'regression':
                raise ValueError('Only classification models can be calibrated, but found task ' + encoder.task_)

            # merge tables
            df, _ = tu.merge_tables(self._invocation.table)
            if df.columns.nlevels != 1:
                raise ValueError(f'Table must have 1 column level, but found {df.columns.nlevels}.')

            # restrict to subset
            df = self._invocation.restrict_to_subset(df)

            x_train, y_train = encoder.transform(data=df)

            sample_weights = self._invocation.get_sample_weights(df)
            if self._invocation.method is None:
                self._invocation.method = 'auto'

            if model is None:
                y_hat = fe.predict_proba(x_train, calibrated=False)
            else:
                y_hat = model.predict_proba(x_train, calibrated=False)

            calibrator = Calibrator(method=self._invocation.method).fit(y_hat, y=y_train, sample_weight=sample_weights)

            if model is not None:
                model.calibrator_ = calibrator
                io.dump(model, loader.path / CaTabRaPaths.Model)
            if fe is not None:
                fe.calibrator_ = calibrator
                fe.dump(loader.path / CaTabRaPaths.FittedEnsemble, as_dict=True)

            end = pd.Timestamp.now()
            logging.log(f'### Calibration finished at {end}')
            logging.log(f'### Elapsed time: {end - self._invocation.start}')
            logging.log(f'### Output saved in {self._invocation.out.as_posix()}')


class CalibrationInvocation(Invocation):

    @property
    def folder(self) -> Union[str, Path]:
        return self._folder

    @property
    def subset(self):
        return self._subset

    @property
    def method(self) -> Optional[str]:
        return self._method

    @method.setter
    def method(self, value: str):
        self._method = value

    def __init__(
        self,
        *table,
        split: Optional[str] = None,
        subset=None,
        sample_weight: Optional[str] = None,
        method: Optional[str] = None,
        out: Union[str, Path, None] = None,
        jobs: Optional[int] = None,
        folder: Union[str, Path] = None
    ):
        super().__init__(*table, split=split, sample_weight=sample_weight, out=out, jobs=jobs)
        self._folder = folder
        self._subset = subset
        self._method = method

    def update(self, src: dict = None):
        super().update(src)
        if src:
            if self._folder is None:
                self._folder = src.get('folder')
            if self._subset is None:
                self._subset = src.get('subset')
            if self._method is None:
                self._method = src.get('method')

    def resolve(self):
        super().resolve()

        if self._subset == '':
            self._subset = None
        if self._method == '':
            self._method = None

        if self._folder is None:
            raise ValueError('No CaTabRa directory specified.')
        else:
            self._folder = io.make_path(self._folder, absolute=True)

        if self._out is None:
            self._out = self._table[0]
            if isinstance(self._out, pd.DataFrame):
                self._out = self._folder / ('calibrate_' + self._start.strftime('%Y-%m-%d_%H-%M-%S'))
            else:
                self._out = \
                    self._folder / ('calibrate_' + self._out.stem + '_' + self._start.strftime('%Y-%m-%d_%H-%M-%S'))
        else:
            self._out = io.make_path(self._out, absolute=True)
        if self._out == self._folder:
            raise ValueError(
                f'Output directory must differ from CaTabRa directory, but both are "{self._out.as_posix()}".'
            )

    def to_dict(self) -> dict:
        dct = super().to_dict()
        dct.update(
            table=['<DataFrame>' if isinstance(tbl, pd.DataFrame) else tbl for tbl in self._table],
            folder=self._folder,
            subset=self._subset,
            method=self._method
        )
        return dct

    def restrict_to_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        subset = self._subset
        split = self._split
        if split is None:
            if subset is not None:
                logging.warn(f'Cannot restrict table to calibration subset "{subset}",'
                             ' because no split has been specified.')
                subset = None
        elif subset is None:
            logging.log(f'Ignoring split "{split}", because no calibration subset has been specified.')
            split = None
        if split is not None:
            if isinstance(subset, str):
                if df[split].dtype.kind in 'iu':
                    subset = int(subset)
                elif df[split].dtype.kind == 'f':
                    subset = float(subset)
                elif df[split].dtype.kind == 'b':
                    subset = subset.lower() in ('true', '1')
                elif df[split].dtype.kind == 'M':
                    subset = pd.to_datetime(subset)
                elif df[split].dtype.kind == 'm':
                    subset = pd.to_timedelta(subset)
            df = df[df[split] == subset]
            logging.log(f'Restricting table to calibration subset {split} = {subset} ({len(df)} entries)')

        return df


class Calibrator(BaseEstimator, TransformerMixin):

    def __init__(self, method: str = 'auto'):
        """
        Calibrator, which transforms uncalibrated predictions of classification problems to calibrated class
        probabilities.
        :param method: The method to use for calibration. Can be "sigmoid" which corresponds to Platt's method (i.e. a
        logistic regression model) or "isotonic" which is a non-parametric approach. Can also be "auto", which defaults
        to "sigmoid" if less than 900 samples are provided in `fit()` and to "isotonic" otherwise.
        """
        self.method = method

    def fit(self, X, y=None, sample_weight=None) -> 'Calibrator':
        """
        Fit this Calibrator instance based on ground truth labels and uncalibrated prediction probabilities (or scores).
        :param X: Uncalibrated predictions, array-like of shape `(n_samples,)`, `(n_samples, 1)` or
        `(n_samples, n_classes)`. In the first two cases, the problem is assumed to be a binary classification problem
        with `X` containing the probabilities/scores of the positive class.
        :param y: Ground truth, array-like of shape `(n_samples,)` or `(n_samples, n_labels)` with values among
        0, ..., `n_classes` - 1 and NaN.
        :param sample_weight: Sample weight, optional. If given must have shape `(n_samples,)`.
        :return: This Calibrator instance.
        """

        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        assert 1 <= y.ndim <= 2
        if y.ndim == 2 and y.shape[1] == 1:
            # binary or multiclass classification
            y = y[:, 0]
        self.multilabel_ = y.ndim == 2
        mask = np.isfinite(y)

        assert len(X) == len(y)
        X = Calibrator._check_X(X, n_estimators=None, multilabel=self.multilabel_)
        if self.multilabel_:
            assert X.shape[1] == y.shape[1]
            mask &= np.isfinite(X)
        else:
            mask &= np.isfinite(X).all(axis=1)
        if sample_weight is not None:
            assert sample_weight.shape == (len(y),)

        if self.method == 'auto':
            self.method_ = 'sigmoid' if mask.sum() < 900 else 'isotonic'
        else:
            self.method_ = self.method

        if self.method_ == 'isotonic':
            self.estimators_ = [IsotonicRegression(out_of_bounds='clip') for _ in range(X.shape[1])]
        elif self.method_ == 'sigmoid':
            self.estimators_ = [_SigmoidCalibration() for _ in range(X.shape[1])]
        else:
            raise ValueError(f'method should be "sigmoid" or "isotonic", but got "{self.method_}".')

        if self.multilabel_:
            for i, e in enumerate(self.estimators_):
                e.fit(X[mask[:, i], i], y[mask[:, i], i],
                      sample_weight=None if sample_weight is None else sample_weight[mask[:, i]])
        else:
            if sample_weight is not None:
                sample_weight = sample_weight[mask]
            for i, e in enumerate(self.estimators_):
                e.fit(X[mask, i], y[mask] == (i + 1 if X.shape[1] == 1 else i), sample_weight=sample_weight)

        return self

    def predict(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        """
        Apply this Calibrator instance to uncalibrated predictions.
        :param X: Uncalibrated predictions, array-like of shape `(n_samples,)`, `(n_samples, 1)` or
        `(n_samples, n_classes)`.
        :return: Calibrated class probabilities, of the same type and shape as `X`.
        """

        y = Calibrator._check_X(X, n_estimators=len(self.estimators_), multilabel=self.multilabel_).copy()
        for i, e in enumerate(self.estimators_):
            y[:, i] = e.predict(y[:, i])
        np.clip(y, 0, 1, out=y)

        if y.shape[1] > 1 and not self.multilabel_:
            div = y.sum(axis=1, keepdims=True)
            np.divide(y, div, out=y, where=div > 0)

        # return same type and shape as input
        if X.ndim == 1:
            y = y[:, 0]
        elif X.shape[1] == 2 and y.shape[1] == 1:
            y = np.hstack([1 - y, y])
        if isinstance(X, pd.DataFrame):
            y = pd.DataFrame(index=X.index, columns=X.columns, data=y)
        elif isinstance(X, pd.Series):
            y = pd.Series(index=X.index, data=y, name=X.name)
        return y

    def predict_proba(self, X):
        """Alias for method `predict()`."""
        return self.predict(X)

    def transform(self, X, y=None):
        """Alias for method `predict()`."""
        return self.predict(X)

    @staticmethod
    def _check_X(X: Union[pd.DataFrame, pd.Series, np.ndarray], n_estimators: Optional[int] = None,
                 multilabel: bool = False) -> np.ndarray:
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if X.ndim == 2:
            if X.shape[1] == 2 and not multilabel:
                # binary classification => restrict to positive class
                X = X[:, -1:]
        else:
            # binary classification => add axis
            assert not multilabel
            assert X.ndim == 1
            X = X[..., np.newaxis]
        if n_estimators is not None and X.shape[1] != n_estimators:
            raise ValueError(f'X has wrong number of columns: expected {n_estimators}, got {X.shape[1]}')
        return X
