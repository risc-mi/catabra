from typing import Union, Optional, Type
from pathlib import Path
import pandas as pd

from ..util import table as tu, io, logging, metrics
from ..core import CaTabRaBase, Invocation
from ..core import CaTabRaPaths


def apply(*table: Union[str, Path, pd.DataFrame], folder: Union[str, Path] = None, model_id=None, explain=None,
          check_ood=None, out: Union[str, Path, None] = None, jobs: Optional[int] = None,
          batch_size: Optional[int] = None, from_invocation: Union[str, Path, dict, None] = None,
          print_results: Union[bool, str] = False):
    """
    Apply an existing CaTabRa object (prediction model, OOD-detector, ...) to given data.
    :param table: The table(s) to apply the CaTabRa object to. If multiple are given, their columns are merged into
    a single table. Must have the same format as the table(s) initially passed to function `analyze()`.
    :param folder: The folder containing the CaTabRa object to apply.
    :param model_id: Optional, ID of the prediction model to apply. If None or "__ensemble__", the sole trained
    model or the entire ensemble are applied.
    :param explain: Explain prediction model(s) on the given data. If "__all__", all models specified by `model_id` are
    explained; otherwise, must be a list of the model ID(s) to explain, which must be a subset of the models that are
    applied.
    :param check_ood: Whether to apply the OOD-detector to the given data. If True, the results of OOD-detection are
    added to the table containing the model predictions.
    :param out: Optional, directory where to save all generated artifacts. Defaults to a directory located in `folder`,
    with a name following a fixed naming pattern. If `out` already exists, the user is prompted to specify whether it
    should be replaced; otherwise, it is automatically created.
    :param jobs: Optional, number of jobs to use. Overwrites the "jobs" config param.
    :param batch_size: Optional, batch size used for applying the prediction model.
    :param from_invocation: Optional, dict or path to an invocation.json file. All arguments of this function not
    explicitly specified are taken from this dict; this also includes the table to apply the CaTabRa object to.
    :param print_results: Whether to print prediction results. If "auto", results are only printed if the number of
    samples does not exceed 30.
    """
    applier = CaTabRaApplication(invocation=from_invocation)
    applier(
        *table,
        folder=folder,
        model_id=model_id,
        explain=explain,
        check_ood=check_ood,
        out=out,
        jobs=jobs,
        batch_size=batch_size,
        print_results=print_results
    )


class CaTabRaApplication(CaTabRaBase):

    @property
    def invocation_class(self) -> Type['ApplicationInvocation']:
        return ApplicationInvocation

    def _call(self):
        loader = io.CaTabRaLoader(self._invocation.folder, check_exists=True)
        self._config = loader.get_config()

        out_ok = self._invocation.resolve_output_dir(
            prompt=f'Application folder "{self._invocation.out.as_posix()}" already exists. Delete?'
        )
        if not out_ok:
            logging.log('### Aborting')
            return

        with logging.LogMirror((self._invocation.out / CaTabRaPaths.ConsoleLogs).as_posix()):
            logging.log(f'### Application started at {self._invocation.start}')
            io.dump(io.to_json(self._invocation), self._invocation.out / CaTabRaPaths.Invocation)

            # merge tables
            df, _ = tu.merge_tables(self._invocation.table)
            if df.columns.nlevels != 1:
                raise ValueError(f'Table must have 1 column level, but found {df.columns.nlevels}.')

            # encode data
            encoder = loader.get_encoder()
            X = encoder.transform(x=df)

            model = loader.get_model_or_fitted_ensemble()
            ood = loader.get_ood()
            if self._invocation.check_ood and ood is not None:
                ood_predictions = pd.DataFrame(ood.predict_proba(X), columns=['__ood_proba'])
                ood_predictions['__ood_decision'] = pd.DataFrame(ood.predict(X))
            else:
                ood_predictions = None

            if encoder.task_ is None or model is None:
                self._invocation.explain = []
                model_predictions = None
            else:
                self._invocation.set_models_to_explain(model)
                # `explain` is now either None or a list: None => explain all models; list => explain only these models

                if encoder.task_ == 'regression':
                    model_predictions = self._apply_regression(encoder, model, X)
                else:
                    model_predictions = self._apply_classification(encoder, model, X)

            # combine `model_predictions` and `ood_predictions`
            if model_predictions is None:
                model_predictions = ood_predictions
            elif ood_predictions is not None:
                assert (model_predictions.index == ood_predictions.index).all()
                model_predictions = pd.concat([model_predictions, ood_predictions], axis=1, sort=False)

            if model_predictions is not None:
                io.write_df(model_predictions, self._invocation.out / CaTabRaPaths.Predictions)
                if self._invocation.print_results is True \
                        or (self._invocation.print_results == 'auto' and model_predictions.shape[0] <= 30
                            and model_predictions.shape[1] <= 8):
                    logging.log('Predictions:\n' + repr(model_predictions))

            end = pd.Timestamp.now()
            logging.log(f'### Application finished at {end}')
            logging.log(f'### Elapsed time: {end - self._invocation.start}')
            logging.log(f'### Output saved in {self._invocation.out.as_posix()}')

        if self._invocation.explain is None or len(self._invocation.explain) > 0:
            from ..explanation import explain as explain_fn
            explain_fn(df, folder=self._invocation.folder, model_id=self._invocation.explain,
                       out=self._invocation.out / 'explanations', glob=False,
                       batch_size=self._invocation.batch_size, jobs=self._invocation.jobs)

    def _apply_classification(self, encoder, model, X: pd.DataFrame) -> pd.DataFrame:
        y_hat = model.predict_proba(X, jobs=self._invocation.jobs, batch_size=self._invocation.batch_size,
                                    model_id=self._invocation.model_id)

        if encoder.task_ == 'multilabel_classification':
            detailed = encoder.inverse_transform(y=y_hat, inplace=True)
            detailed.index = X.index
            detailed.columns = [f'{c}_proba' for c in detailed.columns]
            return detailed
        else:
            detailed = encoder.inverse_transform(y=y_hat, inplace=True)
            detailed.index = X.index
            detailed.columns = [f'{c}_proba' for c in detailed.columns]
            if encoder.task_ == 'multiclass_classification':
                detailed.insert(
                    0,
                    encoder.target_names_[0] + '_pred',
                    encoder.inverse_transform(y=metrics.multiclass_proba_to_pred(y_hat)).values[:, 0]
                )
            return detailed

    def _apply_regression(self, encoder, model, X: pd.DataFrame) -> pd.DataFrame:
        y_hat = model.predict(X, jobs=self._invocation.jobs, batch_size=self._invocation.batch_size,
                              model_id=self._invocation.model_id)
        if y_hat.ndim == 1:
            y_hat = y_hat.reshape(-1, 1)

        detailed = encoder.inverse_transform(y=y_hat, inplace=True)
        detailed.index = X.index
        detailed.columns = [f'{c}_pred' for c in detailed.columns]
        return detailed


class ApplicationInvocation(Invocation):

    @property
    def folder(self) -> Union[str, Path]:
        return self._folder

    @property
    def model_id(self) -> Optional[str]:
        return self._model_id

    @property
    def explain(self):
        return self._explain

    @explain.setter
    def explain(self, value):
        self._explain = value

    @property
    def check_ood(self) -> Optional[bool]:
        return self._check_ood

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    @property
    def print_results(self):
        return self._print_results

    def __init__(
        self,
        *table,
        out: Union[str, Path, None] = None,
        jobs: Optional[int] = None,
        folder: Union[str, Path] = None,
        model_id=None,
        explain=None,
        check_ood: Optional[bool] = None,
        batch_size: Optional[int] = None,
        print_results=None
    ):
        super().__init__(*table, out=out, jobs=jobs)
        self._folder = folder
        self._model_id = model_id
        self._explain = explain
        self._check_ood = check_ood
        self._batch_size = batch_size
        self._print_results = print_results

    def update(self, src: dict = None):
        super().update(src)
        if src:
            if self._folder is None:
                self._folder = src.get('folder')
            if self._model_id is None:
                self._model_id = src.get('model_id')
            if self._explain is None:
                self._explain = src.get('explain')
            if self._check_ood is None:
                self._check_ood = src.get('check_ood')
            if self._batch_size is None:
                self._batch_size = src.get('batch_size')
            if self._print_results is None:
                self._print_results = src.get('print_results')

    def resolve(self):
        super().resolve()

        if self._folder is None:
            raise ValueError('No CaTabRa directory specified.')
        else:
            self._folder = io.make_path(self._folder, absolute=True)

        if self._out is None:
            self._out = self._table[0]
            if isinstance(self._out, pd.DataFrame):
                self._out = self._folder / ('apply_' + self._start.strftime('%Y-%m-%d_%H-%M-%S'))
            else:
                self._out = self._folder / ('apply_' + self._out.stem + '_' + self._start.strftime('%Y-%m-%d_%H-%M-%S'))
        else:
            self._out = io.make_path(self._out, absolute=True)
        if self._out == self._folder:
            raise ValueError(
                f'Output directory must differ from CaTabRa directory, but both are "{self._out.as_posix()}".'
            )

    def to_dict(self) -> dict:
        dic = super().to_dict()
        dic.update(dict(
            table=['<DataFrame>' if isinstance(tbl, pd.DataFrame) else tbl for tbl in self._table],
            folder=self._folder,
            model_id=self._model_id,
            explain=self._explain,
            check_ood=self._check_ood,
            batch_size=self._batch_size,
            print_results=self._print_results
        ))
        return dic

    def set_models_to_explain(self, model):
        if self._model_id == '__ensemble__':
            self._model_id = None

        if self._explain is None:
            self._explain = set()
        elif isinstance(self._explain, list):

            self._explain = set(self._explain)
        elif not isinstance(self._explain, set):
            self._explain = {self._explain}
        if '__ensemble__' in self._explain:
            if self._model_id is None:
                self._explain = None
            else:
                self._explain.remove('__ensemble__')
                self._explain.update(model.model_ids_)
        elif '__all__' in self._explain:
            if self._model_id is None:
                self._explain = None
            else:
                self._explain.remove('__all__')
                self._explain.add(self._model_id)
        if isinstance(self._explain, set):
            if self._model_id is not None and any(e != self._model_id for e in self._explain):
                raise ValueError('Cannot explain models that are not being applied.')
            self._explain = list(self._explain)

    @staticmethod
    def requires_table() -> bool:
        return True
