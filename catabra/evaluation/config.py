from pathlib import Path
from typing import Dict, Union, Optional, List, Sized

import pandas as pd

from ..core import Invocation
from ..util import io


class EvaluationInvocation(Invocation):

    @property
    def folder(self) -> Union[str, Path]:
        return self._folder

    @property
    def model_id(self) -> Optional[str]:
        return self._model_id

    @property
    def explain(self) -> Sized:
        return self._explain

    @explain.setter
    def explain(self, value: Sized):
        self._explain = value

    @property
    def glob(self) -> Optional[bool]:
        return self._glob

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    @property
    def threshold(self) -> Optional[float]:
        return self._threshold

    @property
    def bootstrapping_repetitions(self) -> Optional[int]:
        return self._bootstrapping_repetitions

    @bootstrapping_repetitions.setter
    def bootstrapping_repetitions(self, value: int):
        self._bootstrapping_repetitions = value

    @property
    def bootstrapping_metrics(self) -> Optional[List]:
        return self._bootstrapping_metrics

    def __init__(
        self,
        *table,
        split: Optional[str] = None,
        sample_weight: Optional[str] = None,
        out: Union[str, Path, None] = None,
        jobs: Optional[int] = None,
        folder: Union[str, Path] = None,
        model_id=None,
        explain=None,
        glob: Optional[bool] = False,
        batch_size: Optional[int] = None,
        threshold: Optional[float] = None,
        bootstrapping_repetitions: Optional[int] = None,
        bootstrapping_metrics: Optional[list] = None,
    ):

        super().__init__(*table, split=split, sample_weight=sample_weight, out=out, jobs=jobs)
        self._folder = folder
        self._model_id = model_id
        self._explain = explain
        self._glob = glob
        self._batch_size = batch_size
        self._threshold = threshold
        self._bootstrapping_repetitions = bootstrapping_repetitions
        self._bootstrapping_metrics = bootstrapping_metrics

    def update(self, src: Dict = None):
        super().update(src)
        if src:
            if self._folder is None:
                self._folder = src.get('folder')
            if self._model_id is None:
                self._model_id = src.get('model_id')
            if self._explain is None:
                self._explain = src.get('explain')
            if self._glob is None:
                self._glob = src.get('glob')
            if self._threshold is None:
                self._threshold = src.get('threshold')
            if self._bootstrapping_repetitions is None:
                self._bootstrapping_repetitions = src.get('bootstrapping_repetitions')
            if self._bootstrapping_metrics is None:
                self._bootstrapping_metrics = src.get('bootstrapping_metrics')
            if self._batch_size is None:
                self._batch_size = src.get('bootstrapping_metrics')

    def resolve(self):
        if self._threshold is None:
            self._threshold = 0.5

        if self._out is None:
            self._out = self._table[0]
            if isinstance(self._out, pd.DataFrame):
                self._out = self._folder / ('eval_' + self._start.strftime('%Y-%m-%d_%H-%M-%S'))
            else:
                self._out = self._folder / ('eval_' + self._out.stem + '_' + self._start.strftime('%Y-%m-%d_%H-%M-%S'))
        else:
            self._out = io.make_path(self._out, absolute=True)
        if self._out == self._folder:
            raise ValueError(f'Output directory must differ from CaTabRa directory, but both are "{out.as_posix()}".')

    def to_dict(self) -> Dict:
        dic = super().to_dict()
        dic.update(dict(
            table=['<DataFrame>' if isinstance(tbl, pd.DataFrame) else tbl for tbl in self._table],
            folder=self._folder,
            model_id=self._model_id,
            explain=self._explain,
            glob=self._glob,
            split=self._split,
            sample_weight=self._sample_weight,
            out=self._out,
            jobs=self._jobs,
            threshold=self._threshold,
            bootstrapping_repetitions=self._bootstrapping_repetitions,
            bootstrapping_metrics=self._bootstrapping_metrics,
            timestamp=self._start
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
            if self._model_id is not None and any(e != self._model_id for e in self._invocation._explain):
                raise ValueError('Cannot explain models that are not being evaluated.')
            self._explain = list(self._explain)

